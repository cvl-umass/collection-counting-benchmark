"""Which estimator wins, as a share of cells, at each labeling budget."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import (EST_COLOR, EST_TEXT, EST_LABEL, EST_ORDER,  # noqa: E402
                     INK, INK2, MUTED, SURFACE, TIERS, style)

# winner labels in estimator_wins.csv -> the estimator key the palette is keyed on.
WIN2EST = {"model-only": "model_only_corrected", "monte_carlo": "monte_carlo",
           "difference": "difference", "ratio": "ratio", "regression": "regression",
           "DIS": "dis", "AIS": "ais", "spotDIS": "spotdis", "spotAIS": "spotais"}


def share_table(cell: pd.DataFrame) -> pd.DataFrame:
    """Per budget, the fraction of that budget's cells won by each estimator
    (rows = checkpoint, columns = EST_ORDER, each row sums to 1)."""
    cell = cell.copy()
    cell["est"] = cell.winner.map(WIN2EST)
    cps = sorted(cell.checkpoint.unique())
    share = (cell.groupby(["checkpoint", "est"]).size()
             .unstack(fill_value=0).reindex(columns=EST_ORDER, fill_value=0))
    return share.div(share.sum(axis=1), axis=0).reindex(cps)


def tier_share_table(cell: pd.DataFrame) -> pd.DataFrame:
    """The same win counts marginalized the other way: per occupancy tier, the fraction of
    that tier's cells (models x tasks x budgets) won by each estimator (rows = TIERS,
    columns = EST_ORDER, each row sums to 1).  This is the "3) WINS by occupancy tier"
    block of ``analysis/estimator_wins.py`` as shares."""
    cell = cell.copy()
    cell["est"] = cell.winner.map(WIN2EST)
    share = (cell.groupby(["tier", "est"]).size()
             .unstack(fill_value=0).reindex(columns=EST_ORDER, fill_value=0))
    return share.div(share.sum(axis=1), axis=0).reindex(TIERS)


def fig_share(cell: pd.DataFrame, out: Path) -> None:
    cps = sorted(cell.checkpoint.unique())
    share = share_table(cell)

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
    style(ax)
    x = [100 * c for c in cps]
    bottom = [0.0] * len(cps)
    for est in EST_ORDER:
        h = share[est].to_numpy()
        if h.sum() == 0:
            continue
        ax.bar(x, h, bottom=bottom, width=3.6, color=EST_COLOR[est],
               edgecolor=SURFACE, linewidth=0.6, label=EST_LABEL[est], zorder=2)
        # label a segment only where it is tall enough to hold text
        for xi, hi, bi in zip(x, h, bottom):
            if hi >= 0.05:
                ax.text(xi, bi + hi / 2, f"{100 * hi:.0f}", ha="center",
                        va="center", fontsize=6.5, color=EST_TEXT[est])
        bottom = [b + hi for b, hi in zip(bottom, h)]

    ax.set_xlim(2.5, 52.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(x, [f"{c:g}" for c in x])
    ax.set_yticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100"])
    ax.set_xlabel("% of collection labeled", fontsize=10, color=INK2)
    ax.set_ylabel("share of cells won (%)", fontsize=10, color=INK2)
    ax.set_title("Which estimator wins, by labeling budget", fontsize=12,
                 color=INK, pad=26)
    ax.text(0.5, 1.02, f"share of {cell.model.nunique()} models x "
            f"{cell.class_key.nunique()} tasks = {len(cell) // len(cps)} cells per budget; "
            "lowest mean |frac error| wins the cell",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color=MUTED)
    # legend outside, estimator order top-to-bottom matches the stack top-to-bottom
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="center left", frameon=False,
              fontsize=8.5, labelcolor=INK2, bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"estimator_share.{ext}", dpi=200, bbox_inches="tight",
                    facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="outputs/analysis/estimator_wins.csv")
    ap.add_argument("--out-dir", default="outputs/analysis/figures/results_init")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_share(pd.read_csv(args.input), out)
    print(f"wrote {out / 'estimator_share.png'}")


if __name__ == "__main__":
    main()
