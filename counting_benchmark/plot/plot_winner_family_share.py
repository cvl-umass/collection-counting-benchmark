"""Which model family wins the most tasks, as a function of labeling budget."""

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
from palette import (FAM_COLOR, FAM_LABEL, INK, INK2, MUTED, SURFACE,  # noqa: E402
                     style)

# stack order, bottom -> top: the shrinking family (open MLLM) at the base and the
# growing one (detector) on top, so the crossover reads as the bands trading height.
STACK = ["open_mllm", "prop_mllm", "vlm", "detector"]
TIE_COLOR = MUTED                        # diagonal hatch lines / outline of the tie band
TIE_FILL = "#e9e8e2"                     # light gray fill behind the hatch
TIE_HATCH = "////"                       # diagonal shade for the "no clear winner" band
STACK_DRAW = STACK + ["tie"]             # tie band sits on top, above the families


def _hatch_tie(poly):
    """Turn the top (tie) stackplot band into a diagonally shaded gray patch."""
    poly.set_facecolor(TIE_FILL)
    poly.set_edgecolor(TIE_COLOR)
    poly.set_hatch(TIE_HATCH)
    poly.set_linewidth(0.5)


def family_share_table(grid: pd.DataFrame):
    """Returns (share, cps, n_tasks): share of ALL tasks each family wins at each budget
    (rows = pct_labeled, columns = STACK order plus a trailing ``tie`` band).  A task
    credits its family only when one model is uniquely most accurate; degenerate ties --
    several models with a flat, uninformative g that return the identical estimate (all
    malaria/difficult) -- credit no family and are pooled into the ``tie`` band, so no task
    is dropped, no model is credited for a flat prediction, and the bands fill to 100%."""
    cps = sorted(grid.pct_labeled.unique())
    n_tasks = grid.class_key.nunique()                       # all 12 tasks
    decisive = grid[~grid.winner_degenerate]                 # ties credit no family
    count = (decisive.groupby(["pct_labeled", "winner_family"]).size()
             .unstack(fill_value=0).reindex(index=cps, columns=STACK, fill_value=0))
    share = count / n_tasks
    share["tie"] = 1.0 - share[STACK].sum(axis=1)            # remainder = the tied tasks
    return share, cps, n_tasks


def fig_family_share(grid: pd.DataFrame, out: Path) -> None:
    share, cps, n_tasks = family_share_table(grid)

    plt.rcParams["hatch.linewidth"] = 0.5
    fig, ax = plt.subplots(figsize=(8.4, 5), facecolor=SURFACE)
    style(ax)
    polys = ax.stackplot(cps, [share[f].to_numpy() for f in STACK_DRAW],
                         colors=[FAM_COLOR.get(f, TIE_COLOR) for f in STACK_DRAW],
                         edgecolor=SURFACE, linewidth=0.8, zorder=2)
    _hatch_tie(polys[-1])
    # end-of-band labels: family name at the right edge, vertically at its band center
    cum = np.zeros(len(cps))
    for f in STACK_DRAW:
        h = share[f].to_numpy()
        mid = cum[-1] + h[-1] / 2
        if h[-1] > 0:
            lab = "tie (no clear winner)" if f == "tie" else \
                FAM_LABEL[f].split(",")[0].split("(")[0].strip()
            ax.text(cps[-1] + 0.7, mid, lab, va="center", ha="left",
                    fontsize=8.5, color=FAM_COLOR.get(f, TIE_COLOR))
        cum += h

    ax.set_xlim(cps[0], cps[-1] + 0.2)
    ax.set_ylim(0, 1)
    ax.set_xticks(cps, [f"{c:g}" for c in cps])
    ax.set_yticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100"])
    ax.set_xlabel("% of collection labeled", fontsize=10, color=INK2)
    ax.set_ylabel(f"share of {n_tasks} tasks won (%)", fontsize=10, color=INK2)
    ax.set_title("Off-the-shelf MLLMs win early; the fine-tuned detector overtakes",
                 fontsize=12, color=INK, pad=22)
    ax.text(0.5, 1.02, "each task's winner = its best model+estimator at that budget",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"winner_family_share.{ext}", dpi=200, bbox_inches="tight",
                    facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="outputs/analysis/winner_model_grid.csv")
    ap.add_argument("--out-dir", default="outputs/analysis/figures/results_init")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_family_share(pd.read_csv(args.input), out)
    print(f"wrote {out / 'winner_family_share.png'}")


if __name__ == "__main__":
    main()
