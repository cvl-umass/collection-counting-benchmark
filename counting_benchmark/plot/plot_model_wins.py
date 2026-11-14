"""Which model is most accurate on the most (task, budget) settings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import (FAMILY, FAM_COLOR, FAM_LABEL, FAM_ORDER, NAME,  # noqa: E402
                     INK2, MUTED, SURFACE, style)

LOW = 15   # % labeled threshold for the "low budget" segment


def fig_model_wins(grid: pd.DataFrame, out: Path) -> None:
    outright = grid[~grid.winner_degenerate]                 # drop exact-tie cells
    won_all = outright.groupby("winner_model").size()
    won_low = outright[outright.pct_labeled <= LOW].groupby("winner_model").size()
    df = pd.DataFrame(
        [{"family": f, "name": NAME.get(m, m),
          "all": int(won_all.get(m, 0)), "low": int(won_low.get(m, 0))}
         for m, f in FAMILY.items()])
    never = df[df["all"] == 0].name.tolist()
    df = df[df["all"] > 0].sort_values("all")                # keep winners; largest on top
    print(f"  never win outright ({len(never)}): {', '.join(never)}")

    fig, ax = plt.subplots(figsize=(6.6, 2.8), facecolor=SURFACE)
    style(ax)
    ax.tick_params(labelsize=14)
    for i, r in enumerate(df.itertuples()):
        c = FAM_COLOR[r.family]
        ax.barh(i, r.low, color=c, edgecolor=SURFACE, linewidth=0.5, zorder=3)
        ax.barh(i, r.all - r.low, left=r.low, color=c, alpha=0.4,
                edgecolor=SURFACE, linewidth=0.5, zorder=3)
        if r.all:
            ax.text(r.all + 0.5, i, str(r.all), va="center", ha="left",
                    fontsize=11, color=INK2)
    ax.set_yticks(range(len(df)), df.name)
    ax.set_xlim(0, df["all"].max() * 1.13)
    ax.set_xlabel("# of (task, budget) where the model\nis most accurate (of 120)",
                  fontsize=13, color=INK2)

    present = set(df.family)
    fam_h = [Patch(color=FAM_COLOR[f],
                   label=FAM_LABEL[f].split(",")[0].split("(")[0].strip())
             for f in FAM_ORDER if f in present]
    shade_h = [Patch(color=MUTED, label=rf"$\leq${LOW}% labeled"),
               Patch(color=MUTED, alpha=0.4, label=rf"$>${LOW}% labeled")]
    ax.legend(handles=fam_h + shade_h, loc="lower right", frameon=False,
              fontsize=12, labelcolor=INK2, labelspacing=0.3, handlelength=1.2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"model_wins.{ext}", dpi=150, bbox_inches="tight",
                    facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out / 'model_wins.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", default="outputs/analysis/winner_model_grid.csv")
    ap.add_argument("--out-dir", default="outputs/analysis/figures/results_init")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_model_wins(pd.read_csv(args.grid), out)


if __name__ == "__main__":
    main()
