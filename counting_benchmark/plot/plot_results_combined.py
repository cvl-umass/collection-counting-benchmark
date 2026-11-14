"""Combined two-column results overview."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import (FAM_ORDER, FAM_COLOR, FAM_LABEL, TIERS, INK, INK2, MUTED,  # noqa: E402
                     AXIS, SURFACE, EST_COLOR, EST_TEXT, EST_LABEL, EST_ORDER, style)
from plot_vs_mc import ratios  # noqa: E402
from plot_estimator_share import share_table, tier_share_table  # noqa: E402
from plot_winner_family_share import (family_share_table, STACK,  # noqa: E402
                                       STACK_DRAW, TIE_COLOR, TIE_FILL,
                                       TIE_HATCH, _hatch_tie)

# enlarge type: the figure is a wide figure* scaled down to \textwidth.
LBL, TCK, PANEL, LEG = 13, 10, 12, 10
EST_BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def _fam_name(f: str) -> str:
    return FAM_LABEL[f].split(",")[0].split("(")[0].strip()


def panel_vs_mc(ax, fam: pd.DataFrame, tier: str, left_col: bool, bottom_row: bool):
    style(ax)
    ax.tick_params(labelsize=TCK)
    ax.axhline(1.0, color=AXIS, linewidth=1.3, linestyle="--", zorder=2)
    sub = fam[fam.tier == tier]
    for f in FAM_ORDER:
        s = sub[sub.family == f]
        if s.empty:
            continue
        med = s.groupby("checkpoint")[["best", "worst"]].median()
        x = 100 * med.index.to_numpy()
        ax.plot(x, med["best"], "-", color=FAM_COLOR[f], linewidth=2.2,
                marker="o", markersize=3.2, zorder=4)
        ax.plot(x, med["worst"], "--", color=FAM_COLOR[f], linewidth=1.4,
                alpha=0.85, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.12, 4)
    ax.set_yticks([0.2, 0.5, 1, 2], ["0.2x", "0.5x", "1x", "2x"])
    ax.set_xlim(2, 52)
    ax.set_xticks([5, 20, 35, 50])
    ax.text(0.5, 0.965, f"{tier}", transform=ax.transAxes, ha="center", va="top",
            fontsize=PANEL, color=INK,
            bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
    # axis titles are hoisted to one per block by block_labels(); the panels only decide
    # which of them carry tick labels.
    ax.tick_params(labelbottom=bottom_row, labelleft=left_col)


def panel_est_share(ax, cell: pd.DataFrame):
    style(ax)
    ax.tick_params(labelsize=TCK)
    share = share_table(cell).reindex(EST_BUDGETS)
    x = [100 * c for c in EST_BUDGETS]
    bottom = np.zeros(len(x))
    for est in EST_ORDER:
        h = share[est].to_numpy()
        if np.nansum(h) == 0:
            continue
        ax.bar(x, h, bottom=bottom, width=3.7, color=EST_COLOR[est],
               edgecolor=SURFACE, linewidth=0.6, label=EST_LABEL[est], zorder=2)
        for xi, hi, bi in zip(x, h, bottom):
            if hi >= 0.07:
                ax.text(xi, bi + hi / 2, f"{100 * hi:.0f}", ha="center", va="center",
                        fontsize=7, color=EST_TEXT[est])
        bottom += np.nan_to_num(h)
    ax.set_xlim(2, 52)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .5, 1], ["0", "50", "100"])
    ax.set_ylabel("cases won\n(% of 156)", fontsize=LBL, color=INK2)
    ax.tick_params(labelbottom=False)


def panel_family_share(ax, grid: pd.DataFrame):
    style(ax)
    ax.tick_params(labelsize=TCK)
    share, cps, n_tasks = family_share_table(grid)
    polys = ax.stackplot(cps, [share[f].to_numpy() for f in STACK_DRAW],
                         colors=[FAM_COLOR.get(f, TIE_COLOR) for f in STACK_DRAW],
                         edgecolor=SURFACE, linewidth=0.8, zorder=2)
    _hatch_tie(polys[-1])
    ax.set_xlim(2, 52)
    ax.set_ylim(0, 1)
    ax.set_xticks([5, 10, 20, 30, 40, 50])
    ax.set_yticks([0, .5, 1], ["0", "50", "100"])
    ax.set_ylabel("tasks won\n(% of 12)", fontsize=LBL, color=INK2)


def panel_tier_bar(ax, share: pd.DataFrame, tier: str, right_col: bool):
    """One 100%-stacked bar of that tier's estimator wins, parked just past the 50%-labeled
    end of its vs-MC panel.  Same colors and same bottom-to-top order as the by-budget bars
    on the right, so the two read as one encoding marginalized two ways; this is the "WINS
    by occupancy tier" table of analysis/estimator_wins.py.  Only the right column carries
    the scale, matching how the vs-MC panels label only their left column."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["right"].set_visible(right_col)
    ax.spines["right"].set_color(AXIS)
    bottom = 0.0
    for est in EST_ORDER:
        h = float(share.loc[tier, est])
        if h == 0:
            continue
        ax.bar([0], [h], bottom=[bottom], width=1.0, color=EST_COLOR[est],
               edgecolor=SURFACE, linewidth=0.6, zorder=2)
        if h >= 0.09:
            ax.text(0, bottom + h / 2, f"{100 * h:.0f}", ha="center", va="center",
                    fontsize=7, color=EST_TEXT[est])
        bottom += h
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([0, .5, 1], ["0", "50", "100"])
    if right_col:
        ax.yaxis.tick_right()
        ax.tick_params(colors=MUTED, labelsize=TCK - 2, length=2.5, pad=1.0)
    else:
        ax.tick_params(left=False, right=False, labelleft=False, labelright=False)


def block_labels(fig, lines, bars, budget_ax) -> None:
    """One axis title per block instead of one per panel.  The 2x2 grid shares all three
    of its axes, so ``error / MC error``, ``% labeled`` and ``winning estimators`` are
    placed once, centered on the extent of the axes they describe: the two vertical titles
    span both rows, and ``% labeled`` is centered on the trajectory panels only (the win
    bars have no x axis).  Positions come from the drawn axes, so they follow the gridspec
    if the layout is retuned.

    Both halves get their ``% labeled`` from here rather than from ``set_xlabel`` on the
    right one: an xlabel is offset from its own tick labels by labelpad, which lands a
    hair below a fig.text placed off the axes box, and the two halves then sit at visibly
    different heights.  Sharing one baseline -- the lowest axes of either half, which is
    the same y since both blocks bottom out on the outer gridspec -- keeps them level."""
    box = [ax.get_position() for ax in lines]
    bb = budget_ax.get_position()
    x0, x1 = min(b.x0 for b in box), max(b.x1 for b in box)
    y0, y1 = min(b.y0 for b in box), max(b.y1 for b in box)
    xb = max(ax.get_position().x1 for ax in bars)
    y_xlabel = min(y0, bb.y0) - 0.085
    kw = dict(fontsize=LBL, color=INK2, ha="center", va="center")
    fig.text(x0 - 0.038, (y0 + y1) / 2, "error / MC error", rotation=90, **kw)
    fig.text(xb + 0.028, (y0 + y1) / 2, "winning estimators (%)", rotation=270, **kw)
    fig.text((x0 + x1) / 2, y_xlabel, "% labeled", **kw)
    fig.text((bb.x0 + bb.x1) / 2, y_xlabel, "% labeled", **kw)


def fig_combined(fam: pd.DataFrame, cell: pd.DataFrame, grid: pd.DataFrame,
                 out: Path) -> None:
    plt.rcParams["hatch.linewidth"] = 0.5
    fig = plt.figure(figsize=(13.2, 3.7), facecolor=SURFACE)
    # a wide, flat layout: 2x2 vs-MC on the left, the two share panels stacked on the
    # right (the estimator panel gets more height), and both legends down a shared right
    # margin.  The generous wspace keeps the right y-labels off the left panels.
    gs = fig.add_gridspec(1, 2, width_ratios=[1.22, 0.77], left=0.055, right=0.83,
                          top=0.96, bottom=0.12, wspace=0.4)

    # every vs-MC panel is a pair: the trajectories, then the tier's win bar hard against
    # their right edge (a hair past the 50%-labeled tick).
    left = gs[0].subgridspec(2, 2, hspace=0.20, wspace=0.13)
    right = gs[1].subgridspec(2, 1, height_ratios=[2, 1.0], hspace=0.16)

    tier_share = tier_share_table(cell)
    lines, bars = [], []
    for tier, (r, c) in zip(TIERS, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        pair = left[r, c].subgridspec(1, 2, width_ratios=[1, 0.145], wspace=0.045)
        lines.append(fig.add_subplot(pair[0, 0]))
        bars.append(fig.add_subplot(pair[0, 1]))
        panel_vs_mc(lines[-1], fam, tier, left_col=(c == 0), bottom_row=(r == 1))
        panel_tier_bar(bars[-1], tier_share, tier, right_col=(c == 1))

    ax_top = fig.add_subplot(right[0, 0])
    ax_bot = fig.add_subplot(right[1, 0])
    panel_est_share(ax_top, cell)
    panel_family_share(ax_bot, grid)
    block_labels(fig, lines, bars, ax_bot)

    # both legends on the right margin, each vertically centered on the panel it serves.
    # reversed so the legend reads top->bottom in the same order the bar stacks top->bottom.
    est_h = [plt.Line2D([], [], color=EST_COLOR[e], linewidth=6, label=EST_LABEL[e])
             for e in reversed(EST_ORDER)]
    fig.legend(handles=est_h, loc="center left", bbox_to_anchor=(0.85, 0.72),
               ncol=1, frameon=False, fontsize=LEG, labelcolor=INK2, handlelength=1.1,
               labelspacing=0.4, title="ESTIMATOR", title_fontsize=LEG, alignment="left")

    fam_h = [plt.Line2D([], [], color=FAM_COLOR[f], linewidth=2.8, label=_fam_name(f))
             for f in FAM_ORDER]
    fam_h.append(Patch(facecolor=TIE_FILL, edgecolor=TIE_COLOR, hatch=TIE_HATCH,
                       linewidth=0.6, label="tie (no clear winner)"))
    style_h = [plt.Line2D([], [], color=MUTED, linewidth=2.4, linestyle="-",
                          label="best member"),
               plt.Line2D([], [], color=MUTED, linewidth=1.4, linestyle="--",
                          label="worst member")]
    fig.legend(handles=fam_h + style_h, loc="center left", bbox_to_anchor=(0.85, 0.23),
               ncol=1, frameon=False, fontsize=LEG, labelcolor=INK2, handlelength=1.6,
               labelspacing=0.4, title="MODEL FAMILY", title_fontsize=LEG,
               alignment="left")

    for ext in ("png", "pdf"):
        fig.savefig(out / f"results_overview.{ext}", dpi=300, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default="outputs/analysis/estimates_summary.csv")
    ap.add_argument("--wins", default="outputs/analysis/estimator_wins.csv")
    ap.add_argument("--grid", default="outputs/analysis/winner_model_grid.csv")
    ap.add_argument("--out-dir", default="outputs/analysis/figures/results_init")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summ = pd.read_csv(args.summary)
    summ.columns = [c.lstrip("﻿") for c in summ.columns]
    fig_combined(ratios(summ), pd.read_csv(args.wins), pd.read_csv(args.grid), out)
    print(f"wrote {out / 'results_overview.png'}")


if __name__ == "__main__":
    main()
