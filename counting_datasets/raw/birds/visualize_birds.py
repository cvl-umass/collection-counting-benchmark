"""
Visualize bird annotations (points + HBBs) on a random crop of a bird .tif image.

Loads from the built index at counting/data/index.sqlite.
Red rectangles = Otsu pseudo-bboxes (role="instance"), canonical counted format.
Green dots = annotator points (role="point"), auxiliary.

Usage:
    python visualize_birds.py
    python visualize_birds.py --scene sky           # sky image (_DSC5214.tif)
    python visualize_birds.py --scene reeds         # reeds image (_DSC5295.tif)
    python visualize_birds.py --crop-size 800       # crop width/height in px (default 600)
    python visualize_birds.py --seed 42             # reproducible random crop
    python visualize_birds.py --out crop.png        # save to file instead of showing
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

HERE = Path(__file__).parent.parent.parent
INDEX_PATH = HERE / "counting" / "data" / "index.sqlite"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize bird annotations on a crop."
    )
    parser.add_argument("--scene", choices=["sky", "reeds"], default=None)
    parser.add_argument(
        "--crop-size",
        type=int,
        default=600,
        help="Side length of the crop in pixels (default: 600)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Overlay tile grid lines to verify coordinate alignment",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Save output to this path instead of displaying",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if not INDEX_PATH.exists():
        print(f"ERROR: index not found at {INDEX_PATH}", file=sys.stderr)
        sys.exit(1)

    conn = _connect(INDEX_PATH)

    # --- pick image ---
    rows = conn.execute(
        "SELECT image_id, path, width, height, meta_json FROM images WHERE dataset = 'birds'"
    ).fetchall()

    if not rows:
        print("No bird images found in the index.", file=sys.stderr)
        sys.exit(1)

    if args.scene:
        rows = [
            r for r in rows if json.loads(r["meta_json"]).get("scene") == args.scene
        ]
        if not rows:
            print(f"No bird images with scene='{args.scene}' found.", file=sys.stderr)
            sys.exit(1)

    row = random.choice(rows)
    image_id = row["image_id"]
    image_path = row["path"]
    img_w, img_h = row["width"], row["height"]
    scene = json.loads(row["meta_json"]).get("scene", "unknown")
    print(f"Image : {Path(image_path).name}  ({img_w}×{img_h} px)  scene={scene}")

    # --- load annotations ---
    ann_rows = conn.execute(
        "SELECT ann_type, geometry_json, role FROM annotations WHERE image_id = ? ORDER BY role, instance_index",
        [image_id],
    ).fetchall()
    conn.close()

    points: list[tuple[float, float]] = []
    hbbs: list[tuple[float, float, float, float]] = []

    for a in ann_rows:
        geom = json.loads(a["geometry_json"])
        if a["role"] == "point" and a["ann_type"] == "point":
            points.append((geom["x"], geom["y"]))
        elif a["role"] == "instance" and a["ann_type"] == "hbb":
            hbbs.append((geom["x"], geom["y"], geom["w"], geom["h"]))

    print(f"  {len(points):,} points  |  {len(hbbs):,} HBBs  (in full image)")

    if not points:
        print(
            "No point annotations found — cannot choose a crop center.", file=sys.stderr
        )
        sys.exit(1)

    # --- choose crop centered on a random point ---
    cx, cy = random.choice(points)
    half = args.crop_size // 2
    x0 = int(cx) - half
    y0 = int(cy) - half
    # clamp to image bounds
    x0 = max(0, min(x0, img_w - args.crop_size))
    y0 = max(0, min(y0, img_h - args.crop_size))
    x1 = x0 + args.crop_size
    y1 = y0 + args.crop_size
    print(f"  crop : ({x0}, {y0}) → ({x1}, {y1})")

    # --- crop the .tif ---
    with Image.open(image_path) as im:
        crop = im.crop((x0, y0, x1, y1)).convert("RGB")

    # --- filter annotations to crop-local coordinates ---
    crop_points = [(x - x0, y - y0) for x, y in points if x0 <= x < x1 and y0 <= y < y1]
    crop_hbbs = [
        (bx - x0, by - y0, bw, bh)
        for bx, by, bw, bh in hbbs
        if bx < x1 and bx + bw > x0 and by < y1 and by + bh > y0
    ]
    print(f"  in crop: {len(crop_points)} points  |  {len(crop_hbbs)} HBBs")

    # --- draw ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(crop)

    # Tile grid overlay — draw lines at each tile boundary inside the crop
    if args.grid:
        tile_sizes = {"sky": 200, "reeds": 160}
        tile_px = tile_sizes.get(scene, 200)
        # vertical lines
        col_start = x0 // tile_px
        col_end = math.ceil(x1 / tile_px)
        for col in range(col_start, col_end + 1):
            lx = col * tile_px - x0
            ax.axvline(lx, color="yellow", linewidth=0.5, alpha=0.6)
        # horizontal lines
        row_start = y0 // tile_px
        row_end = math.ceil(y1 / tile_px)
        for row in range(row_start, row_end + 1):
            ly = row * tile_px - y0
            ax.axhline(ly, color="yellow", linewidth=0.5, alpha=0.6)

    for bx, by, bw, bh in crop_hbbs:
        ax.add_patch(
            patches.Rectangle(
                (bx, by),
                bw,
                bh,
                linewidth=1.5,
                edgecolor="red",
                facecolor="none",
                alpha=0.85,
            )
        )

    if crop_points:
        xs, ys = zip(*crop_points)
        ax.scatter(xs, ys, s=25, c="lime", edgecolors="black", linewidths=0.5, zorder=5)

    ax.set_title(
        f"{Path(image_path).name}  |  scene={scene}\n"
        f"crop ({x0},{y0})→({x1},{y1})  |  "
        f"{len(crop_points)} points  {len(crop_hbbs)} HBBs",
        fontsize=9,
    )
    ax.axis("off")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="lime",
            markeredgecolor="black",
            markersize=8,
            label="point (original)",
        ),
    ]
    if hbbs:
        legend_handles.append(
            patches.Patch(edgecolor="red", facecolor="none", label="HBB (pseudo-bbox)")
        )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved → {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
