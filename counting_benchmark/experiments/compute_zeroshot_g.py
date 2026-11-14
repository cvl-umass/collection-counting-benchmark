"""Compute a frozen model's g-vector over a class's whole pool, offline.

Writes a ``*__g_fixed`` store that the estimators replay without a sampling loop.
"""

from __future__ import annotations

import os

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch

# Make the repo root importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.experiment_recorder import ExperimentRecorder  # noqa: E402
from experiments.run import build_dataset, compute_run_dir, make_eval_loader  # noqa: E402
from models import _resolve_text_prompt, build_model  # noqa: E402

# Root of the counting-datasets checkout.  Override with --index-root /
# --raw-root, or set COUNTING_DATA_ROOT; the fallback assumes the two
# repositories are checked out side by side.
_DATA_ROOT = os.environ.get(
    "COUNTING_DATA_ROOT",
    next((p for p in ("../counting_datasets", "../counting-datasets")
          if os.path.isdir(p)), "../counting_datasets"))
_ZERO_SHOT_MODELS = ("grounding_dino", "owlv2", "florence2")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)

    # Identical flags/defaults to experiments/run.py's "Dataset / pool" group,
    # so --image-pool-list (or --pool-size) picks out exactly the same tiles.
    g = ap.add_argument_group("Dataset / pool")
    g.add_argument("--index-root", default=f"{_DATA_ROOT}/counting/data")
    g.add_argument("--raw-root", default=f"{_DATA_ROOT}/raw")
    g.add_argument(
        "--class-key", required=True, help="e.g. dota/small_vehicle or birds/bird"
    )
    g.add_argument(
        "--tile-sizes",
        default="outputs/suggested_tile_sizes.json",
        help="Suggestion store mapping class_key -> {tile_size, tile_threshold, "
        "...}; sets per-class tiling so the loop matches the DINOv3 cache. "
        'Pass "" to disable and rely on --tile-size/--tile-threshold.',
    )
    g.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="override the store's tile_size for this class (default: from "
        "--tile-sizes).",
    )
    g.add_argument("--max-whole-image-side", type=int, default=1536)
    g.add_argument(
        "--tile-threshold",
        type=int,
        default=None,
        help="override the store's tile_threshold (default: from --tile-sizes, "
        "else falls back to --max-whole-image-side).",
    )
    g.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help="if set, randomly subsample this many source images",
    )
    g.add_argument("--pool-sample-seed", type=int, default=0)
    g.add_argument(
        "--image-pool-list",
        default=None,
        help="path to a text file listing image filenames to include -- pass "
        "the SAME pool file (e.g. the frozen budget5000 list) used by any "
        "experiments/run.py trajectory for this class, so tile_index lines up "
        "(mutually exclusive with --pool-size)",
    )

    g = ap.add_argument_group("Model")
    g.add_argument("--model", required=True, choices=_ZERO_SHOT_MODELS)
    g.add_argument(
        "--text-prompt",
        default=None,
        help="defaults to a prettified class_key (see models._resolve_text_prompt)",
    )
    g.add_argument("--score-thresh", type=float, default=0.5)
    g.add_argument(
        "--capture-score-floor",
        type=float,
        default=0.05,
        help="record every detection with confidence >= this floor to "
        "detections.npz; does not affect the counts in g. 0.05 = the "
        "detector's natural emission floor.",
    )
    g.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="DataLoader batch size for inference. There is no training here "
        "(unlike experiments/run.py, where --batch-size also feeds fit()), so "
        "this is the only batch-size knob.",
    )
    g.add_argument("--num-workers", type=int, default=0)

    g = ap.add_argument_group("I/O")
    g.add_argument("--out-root", default="outputs")
    g.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute even if this (class, model, prompt) already has a "
        "cached g (default: skip -- inference is not free)",
    )

    args = ap.parse_args(argv)
    # make_eval_loader (reused from experiments.run) reads args.eval_batch_size
    # as an optional override of args.batch_size; there's no separate eval
    # mode here, so it's always unset and batch_size alone decides.
    args.eval_batch_size = None
    return args


def derive_run_id(args: argparse.Namespace) -> str:
    """``<model>__g_fixed[__prompt-<slug>]`` -- no sampler/seed: one g per
    (class, model, prompt), shared by every simulated trajectory."""
    parts = [args.model, "g_fixed"]
    if args.text_prompt:
        slug = re.sub(r"[^a-z0-9]+", "-", args.text_prompt.strip().lower()).strip("-")
        parts.append(f"prompt-{slug}")
    return "__".join(parts)


def build_config(args: argparse.Namespace, run_id: str, ds) -> dict:
    return {
        "run_id": run_id,
        "regime": "g_fixed",
        "class_key": args.class_key,
        "model": args.model,
        "text_prompt": args.text_prompt,
        "text_prompt_resolved": _resolve_text_prompt(args),
        "score_thresh": args.score_thresh,
        "capture_score_floor": args.capture_score_floor,
        "tile_size": int(ds.tile_size),
        "tile_threshold": int(ds.tile_threshold),
        "max_whole_image_side": int(ds.max_whole_image_side),
        "image_pool_list": args.image_pool_list,
        "pool_size": args.pool_size,
        "pool_sample_seed": args.pool_sample_seed,
        "F_true": int(ds.F_true),
        "N": int(len(ds)),
        "n_images": int(ds.num_images),
    }


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    ds = build_dataset(args)
    loader = make_eval_loader(ds, args)
    print(
        f"dataset: {len(ds)} units from {ds.num_images} images; F_true={ds.F_true}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = build_model(args, device)
    if model.trainable:
        raise SystemExit(
            f"{args.model} is trainable; this script is for zero-shot models "
            "only -- use experiments/run.py instead."
        )

    run_id = derive_run_id(args)
    run_dir = compute_run_dir(args, run_id)
    print(f"run_dir: {run_dir}")
    recorder = ExperimentRecorder(run_dir)

    if recorder.has_step_data() and not args.overwrite:
        print(f"{run_dir} already has a cached g; pass --overwrite to recompute. Skipping.")
        return
    if args.overwrite:
        deleted = recorder.wipe()
        if deleted:
            print(f"--overwrite: deleted {sorted(deleted)}")

    recorder.write_config(build_config(args, run_id, ds))
    recorder.write_manifest(ds.manifest())

    print("running zero-shot inference (single pass, greedy/deterministic)...")
    t0 = time.perf_counter()
    g, det_scores = model.predict(
        loader,
        device,
        score_thresh=args.score_thresh,
        capture_floor=args.capture_score_floor,
    )
    elapsed = time.perf_counter() - t0

    snap_id = recorder.append_snapshot(g, step=0)
    recorder.append_detections(snap_id, det_scores)
    recorder.finalize()
    print(
        f"g computed: F_model={float(g.sum()):.1f} in {elapsed:.1f}s -> {run_dir}"
    )


if __name__ == "__main__":
    main()
