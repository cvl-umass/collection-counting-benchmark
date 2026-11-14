"""Model-assisted counting versus the model-blind Monte Carlo baseline.

Per occupancy tier, each family's best and worst member bound a band; below 1
means using the model beats labeling blindly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import (FAM_ORDER, FAM_COLOR, FAM_LABEL, TIERS, INK, INK2,  # noqa: E402
                     MUTED, AXIS, SURFACE, style)
from plot_mllm_summary import model_best, mc_floor  # noqa: E402


def ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Per (class, budget, family): the family's best- and worst-member error as a
    ratio to that cell's Monte Carlo error."""
    best = model_best(df)                                  # each model's best estimator
    mc = mc_floor(df)[["class_key", "checkpoint", "err"]].rename(columns={"err": "mc"})
    b = best.merge(mc, on=["class_key", "checkpoint"])
    b["ratio"] = b.err / b.mc
    g = b.groupby(["tier", "class_key", "checkpoint", "family"]).ratio
    fam = g.agg(best="min", worst="max").reset_index()     # over members within a family
    return fam


def fig_vs_mc(fam: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), sharey=True, facecolor=SURFACE)
    for ax, tier in zip(axes, TIERS):
        style(ax)
        ax.axhline(1.0, color=AXIS, linewidth=1.3, linestyle="--", zorder=2)
        sub = fam[fam.tier == tier]
        for f in FAM_ORDER:
            s = sub[sub.family == f]
            if s.empty:
                continue
            # median over the tier's tasks of the family's best / worst member ratio
            med = s.groupby("checkpoint")[["best", "worst"]].median()
            x = 100 * med.index.to_numpy()
            ax.plot(x, med["best"], "-", color=FAM_COLOR[f], linewidth=2.2,
                    marker="o", markersize=3.5, zorder=4)
            ax.plot(x, med["worst"], "--", color=FAM_COLOR[f], linewidth=1.4,
                    alpha=0.85, zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(0.12, 4)
        ax.set_yticks([0.2, 0.5, 1, 2], ["0.2x", "0.5x", "1x", "2x"])
        ax.set_xlabel("% labeled", fontsize=9, color=INK2)
        ax.set_title(f"{tier} tier", fontsize=10.5, color=INK)
    axes[0].set_ylabel("error  /  Monte Carlo error", fontsize=9.5, color=INK2)

    fam_h = [plt.Line2D([], [], color=FAM_COLOR[f], linewidth=2.4,
                        label=FAM_LABEL[f].split(",")[0].split("(")[0].strip())
             for f in FAM_ORDER]
    style_h = [plt.Line2D([], [], color=MUTED, linewidth=2.2, linestyle="-",
                          label="best member"),
               plt.Line2D([], [], color=MUTED, linewidth=1.4, linestyle="--",
                          label="worst member")]
    fig.legend(handles=fam_h + style_h, loc="upper center", ncol=6, frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Model-assisted counting beats model-blind Monte Carlo (below 1x)",
                 y=1.14, fontsize=13, color=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"vs_monte_carlo.{ext}", dpi=200, bbox_inches="tight",
                    facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="outputs/analysis/estimates_summary.csv")
    ap.add_argument("--out-dir", default="outputs/analysis/figures/results_init")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    df.columns = [c.lstrip("﻿") for c in df.columns]
    fig_vs_mc(ratios(df), out)
    print(f"wrote {out / 'vs_monte_carlo.png'}")


if __name__ == "__main__":
    main()
