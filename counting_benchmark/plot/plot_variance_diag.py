"""Variance diagnostics: reported interval width against realized spread."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import (EST_COLOR, INK, INK2, MUTED, AXIS, GRID, SURFACE,  # noqa: E402
                     FAM_COLOR, style)

TIERS = ["full", "partial", "sparse", "rare"]
DETECTOR = FAM_COLOR["detector"]        # purple, for the FRCNN model-variance tail
FROZEN = EST_COLOR["dis"]               # blue, for the frozen-model cloud


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lstrip("﻿") for c in df.columns]
    df["an_sd"] = np.sqrt(df["variance_mean"]) / df["F_true"]   # analytic, as SD of frac_error
    df["emp_sd"] = df["sd_frac_error"]                          # empirical cross-trajectory
    df["is_frcnn"] = df.model.str.startswith("frcnn")
    df["tier"] = pd.Categorical(df.tier, TIERS, ordered=True)
    return df


# ---------------------------------------------------------------- calibration
def fig_calibration(df: pd.DataFrame, out: Path) -> None:
    """Analytic vs empirical SD, one point per (model,class,sampler,estimator,budget).

    model_only has no within-run variance and is dropped.  Points on the diagonal are
    well-calibrated; the FRCNN tail where the analytic variance collapses to ~0 (the
    model-retraining variance no within-run formula sees) is floored and marked."""
    v = df[(df.estimator != "model_only_corrected") & df.an_sd.notna() & (df.emp_sd > 0)].copy()
    floor = 1e-3
    v["an_disp"] = v.an_sd.clip(lower=floor)
    collapsed = v.an_sd < floor

    fig, ax = plt.subplots(figsize=(4.6, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    lo, hi = 8e-4, 3.0
    ax.plot([lo, hi], [lo, hi], color=AXIS, linewidth=1.2, zorder=1)          # y = x
    for lab, sub, col in [("frozen model", v[~v.is_frcnn], FROZEN),
                          ("FRCNN, on diagonal", v[v.is_frcnn & ~collapsed], DETECTOR)]:
        ax.scatter(sub.emp_sd, sub.an_disp, s=9, color=col, alpha=0.35,
                   edgecolors="none", zorder=3, label=lab)
    fc = v[v.is_frcnn & collapsed]                                            # analytic var -> 0
    ax.scatter(fc.emp_sd, fc.an_disp, s=26, marker="x", color=DETECTOR,
               linewidths=1.1, zorder=4, label="FRCNN, analytic var $\\to$ 0")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.axhline(floor, color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel("empirical SD of frac. error (cross-trajectory)", fontsize=10, color=INK2)
    ax.set_ylabel("analytic SD of frac. error (plug-in $\\sqrt{\\hat V}/F$)", fontsize=10, color=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2,
              handletextpad=0.3, borderaxespad=0.2)
    med = (v.an_sd / v.emp_sd).median()
    ax.text(0.97, 0.06, f"median analytic/empirical = {med:.2f}\n"
            f"{((v.an_sd/v.emp_sd).between(0.9,1.11)).mean()*100:.0f}% within $\\pm$10%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"variance_calibration.{ext}", dpi=200, bbox_inches="tight",
                    facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------- per-tier bars
# frozen-model deployment routing: how the paper credits each estimator.
def _elabel(r):
    e, s = r.estimator, r.sampler
    if e == "monte_carlo":
        return "Monte Carlo"
    if e == "dis":
        return "DIS"
    if e == "spotdis":
        return "spotDIS"
    if e in ("difference", "ratio", "regression"):
        return f'{e}·{"correct" if s == "dis" else "uniform"}'
    return None


# top -> bottom: the model-directed / correct group, then Monte Carlo, then the
# uniform (model-blind proposal) group; fixed across panels so the crossover shows.
EST_ROWS = ["DIS", "spotDIS", "difference·correct", "ratio·correct",
            "regression·correct", "Monte Carlo",
            "difference·uniform", "ratio·uniform", "regression·uniform"]
ROW_BASE = {"DIS": "dis", "spotDIS": "spotdis", "Monte Carlo": "monte_carlo",
            "difference·correct": "difference", "ratio·correct": "ratio",
            "regression·correct": "regression", "difference·uniform": "difference",
            "ratio·uniform": "ratio", "regression·uniform": "regression"}


def fig_by_tier(df: pd.DataFrame, out: Path) -> None:
    fr = df[~df.is_frcnn].copy()
    fr["elab"] = fr.apply(_elabel, axis=1)
    fr = fr[fr.elab.notna()]
    piv = (fr.groupby(["elab", "tier"], observed=True).sd_frac_error.median()
           .unstack().reindex(index=EST_ROWS, columns=TIERS))

    y = np.arange(len(EST_ROWS))[::-1]                       # first row at top
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.4), sharex=True, facecolor=SURFACE)
    xmax = float(np.nanmax(piv.to_numpy())) * 1.08
    for ax, tier in zip(axes, TIERS):
        style(ax)
        vals = piv[tier].to_numpy()
        for yi, lab, val in zip(y, EST_ROWS, vals):
            uniform = lab.endswith("uniform")
            ax.barh(yi, val, height=0.72, color=EST_COLOR[ROW_BASE[lab]],
                    edgecolor=SURFACE, linewidth=0.6, zorder=2,
                    hatch="////" if uniform else None,
                    alpha=0.55 if uniform else 1.0)
        best = np.nanargmin(vals)                            # mark the lowest-variance estimator
        ax.scatter(vals[best], y[best], marker="*", s=90, color=INK, zorder=4)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, len(EST_ROWS) - 0.4)
        ax.set_title(tier, fontsize=11, color=INK)
        ax.set_xlabel("median SD of frac. error", fontsize=9.5, color=INK2)
        if ax is axes[0]:
            ax.set_yticks(y, EST_ROWS, fontsize=9)
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=8.5)

    plt.rcParams["hatch.linewidth"] = 0.5
    leg = [Patch(facecolor=MUTED, edgecolor=SURFACE, label="correct / model-directed proposal"),
           Patch(facecolor=MUTED, edgecolor=SURFACE, alpha=0.55, hatch="////",
                 label="uniform (model-blind) proposal"),
           Line2D([], [], marker="*", color=INK, linestyle="none", markersize=9,
                  label="lowest variance in tier")]
    fig.legend(handles=leg, loc="lower center", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"variance_by_tier.{ext}", dpi=200, bbox_inches="tight",
                    facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default="outputs/analysis/estimates_summary.csv")
    ap.add_argument("--out-dir", default="outputs/analysis/figures/diagnostic")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = prep(pd.read_csv(args.summary))
    fig_calibration(df, out)
    fig_by_tier(df, out)
    print(f"wrote {out / 'variance_calibration.png'} and {out / 'variance_by_tier.png'}")


if __name__ == "__main__":
    main()
