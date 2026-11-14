"""How widely off-the-shelf MLLMs disagree on the raw collection total."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import (MODEL_COLOR, FAMILY, NAME, INK2, MUTED, AXIS, GRID,  # noqa: E402
                     SURFACE, class_meta, _shades)

# mllm_total_counts.csv display column -> the model key the palette is keyed on.
COL2MODEL = {"Aya-8B": "aya-vision-8b", "Gemini-3.1": "gemini-3.1-pro-preview",
             "GPT-5.2": "gpt-5.2", "InternVL3.5": "internvl3-5-8b-hf",
             "Molmo-7B": "molmo-7b-d-0924", "Molmo2-8B": "molmo2-8b",
             "Qwen3-VL-8B": "qwen3-vl-8b-instruct", "Qwen3.7+": "qwen3.7-plus"}
MODEL2COL = {v: k for k, v in COL2MODEL.items()}

# Models grouped by family (open MLLMs first, then proprietary), so the legend reads as two
# blocks and each block's tints run light -> dark in the palette's member order.
MLLM_ORDER = ["aya-vision-8b", "internvl3-5-8b-hf", "qwen3-vl-8b-instruct",
              "molmo-7b-d-0924", "molmo2-8b",
              "gemini-3.1-pro-preview", "gpt-5.2", "qwen3.7-plus"]
FAM_MARKER = {"open_mllm": "o", "prop_mllm": "D"}   # second channel for the two families

# Proprietary models are near-white and narrowly spaced in the canonical per-model palette,
# so the three are nearly indistinguishable here.  Give them their own wider light->dark
# ramp off a mid-rose RdPu anchor (open MLLMs already separate well, so they keep MODEL_COLOR).
PROP_COLOR = dict(zip(["gemini-3.1-pro-preview", "gpt-5.2", "qwen3.7-plus"],
                      _shades("#d94b8f", 3, light=0.72, hue=0.0)))   # light -> dark


def fig_disagreement(counts: pd.DataFrame, meta: pd.DataFrame, out: Path) -> None:
    order = list(meta.class_key)                       # full -> rare
    y_of = {c: len(order) - 1 - i for i, c in enumerate(order)}  # full at top
    short = dict(zip(meta.class_key, meta.short))
    row = counts.set_index("class")

    fig, ax = plt.subplots(figsize=(7.0, 3.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for c in order:                                          # faint row guides
        ax.axhline(y_of[c], color=GRID, linewidth=0.7, zorder=0)
    ax.axvline(1.0, color=AXIS, linewidth=1.3, zorder=1)     # truth

    for model in MLLM_ORDER:
        col, fam = MODEL2COL[model], FAMILY[model]
        ratios = [row.loc[c, col] / row.loc[c, "F_true"] for c in order]
        ax.scatter(ratios, [y_of[c] for c in order], s=60, marker=FAM_MARKER[fam],
                   color=PROP_COLOR.get(model, MODEL_COLOR[model]), edgecolors=INK2,
                   linewidths=0.5, alpha=0.9, zorder=3, label=NAME[model])

    ax.set_xscale("log")
    ax.set_xlim(3e-3, 1e2)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_yticks([y_of[c] for c in order], [short[c] for c in order])
    ax.set_xlabel("predicted total / true total (log)",
                  fontsize=14, color=INK2)
    ax.text(1.0, len(order) - 0.35, "truth", ha="center", va="bottom", fontsize=10,
            color=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    ax.legend(loc="center left", frameon=False, fontsize=12, labelcolor=INK2,
              handletextpad=0.3, labelspacing=0.35, borderaxespad=0.0,
              bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"mllm_disagreement.{ext}", dpi=300, bbox_inches="tight",
                    facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="outputs/analysis/mllm_total_counts.csv")
    ap.add_argument("--summary", default="outputs/analysis/estimates_summary.csv")
    ap.add_argument("--out-dir", default="outputs/analysis/figures/results_init")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = pd.read_csv(args.input)
    summ = pd.read_csv(args.summary)
    summ.columns = [c.lstrip("﻿") for c in summ.columns]
    meta = class_meta(summ)
    fig_disagreement(counts, meta, out)
    print(f"wrote {out / 'mllm_disagreement.png'}")


if __name__ == "__main__":
    main()
