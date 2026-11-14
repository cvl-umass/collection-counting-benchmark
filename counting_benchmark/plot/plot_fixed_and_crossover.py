"""One task's estimator curves stacked over the task-averaged detector-vs-MLLM crossover."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import (EST_COLOR, EST_LABEL, EST_ORDER, FAM_COLOR, INK, INK2,  # noqa: E402
                     SURFACE, style)
from plot_fixed_model_error import MODELS, draw_model_panel, valid_rows  # noqa: E402
from plot_detector_vs_mllm_loglog import (MEAN_LABEL, MEAN_STYLE, dense_summary,  # noqa: E402
                                          draw_means, family_means)
from plot_mllm_summary import _family_envelope, mc_floor, model_best  # noqa: E402

# Single-column type: small but never below ~7pt, a common floor for figure text.
LBL, TCK, TITLE, LEG = 14, 10, 14, 12.5
LEG_X = 0.745          # left edge of the shared legend margin, in figure coords
# Both rows end at the same budget *and* put it on their right spine, so the 50% mark of the
# top panels sits at the same figure x as the 50% mark of the bottom one (the rows share a
# gridspec column, so their axes boxes already coincide -- only the data limits had to match).
XMAX = 50.0            # % labeled at the right spine of every panel


def _row_center(axes) -> float:
    """Vertical center of a row of axes, in figure coords (for anchoring its legend)."""
    boxes = [ax.get_position() for ax in axes]
    return (max(b.y1 for b in boxes) + min(b.y0 for b in boxes)) / 2


def fig_combined(perstep: pd.DataFrame, fam: pd.DataFrame, budgets: np.ndarray,
                 class_key: str, meta_row: pd.Series, out: Path) -> None:
    df = valid_rows(perstep, class_key)
    means = family_means(fam)

    fig = plt.figure(figsize=(9, 5.5), facecolor=SURFACE)   # wide, not square
    # One column of two rows inside a fixed box, with the right margin reserved for both
    # legends.  Explicit left/right (no tight_layout) so the two rows stay flush with each
    # other and the legends can be anchored to each row's center.
    gs = fig.add_gridspec(2, 1, left=0.1, right=0.745, top=0.94, bottom=0.10,
                          height_ratios=[1.0, 1.2], hspace=0.42)
    top = gs[0].subgridspec(1, 2, wspace=0.09)

    # --- (a) one task, model fixed: the estimators ---------------------------------
    ax_a = [fig.add_subplot(top[0, 0])]
    ax_a.append(fig.add_subplot(top[0, 1], sharey=ax_a[0]))
    present: list[str] = []
    for ax, (model, label) in zip(ax_a, MODELS):
        style(ax)
        ax.tick_params(labelsize=TCK)
        present += draw_model_panel(ax, df, model, linewidth=1.6)
        ax.set_yscale("log")
        ax.set_xlim(0, XMAX)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_xlabel("% labeled", fontsize=LBL + 4, color=INK2)
        ax.set_title(label, fontsize=TITLE, color=INK, pad=4)
    ax_a[1].tick_params(labelleft=False)
    ax_a[0].set_ylabel("mean\n|fractional error|", fontsize=LBL, color=INK2)


    # --- (b) all tasks, family free to pick its best member: the crossover ----------
    ax_b = fig.add_subplot(gs[1])
    style(ax_b)
    ax_b.tick_params(labelsize=TCK)
    draw_means(ax_b, means, markersize=2.8, quiet=True)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    xt = [t for t in (1, 2, 5, 10, 20, 50) if t >= 100 * budgets[0] - 1e-9]
    ax_b.set_xticks(xt)
    ax_b.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_b.set_xlim(100 * budgets[0] * 0.9, XMAX)
    # XMAX is now the right spine, so the 50% markers would be sliced in half by it -- let
    # the curves draw past the axes instead (nothing but margin lies to the right).
    for line in ax_b.get_lines():
        line.set_clip_on(False)
    ax_b.set_xlabel("% labeled (log)", fontsize=LBL + 4, color=INK2)
    ax_b.set_ylabel("mean best-in-family\n|fractional error|", fontsize=LBL, color=INK2)

    # --- the two legends -------------------------------------------------------------
    # No row headers: the caption carries which task (a) shows and that (b) is the
    # task average, so the rows are only separated by white space.
    est_h = [plt.Line2D([], [], color=EST_COLOR[e], linewidth=5, label=EST_LABEL[e])
             for e in reversed(EST_ORDER) if e in present]
    fig.legend(handles=est_h, loc="center left", bbox_to_anchor=(LEG_X+0.01, _row_center(ax_a)),
               frameon=False, fontsize=LEG, labelcolor=INK2, handlelength=1.1,
               labelspacing=0.35, title="ESTIMATOR", title_fontsize=LEG, alignment="left")

    fam_h = [plt.Line2D([], [], color=FAM_COLOR[f], linewidth=2.2,
                        linestyle=MEAN_STYLE[f][0], label=MEAN_LABEL[f])
             for f, s in means.items() if not s.empty]
    fig.legend(handles=fam_h, loc="center left", bbox_to_anchor=(LEG_X, _row_center([ax_b])),
               frameon=False, fontsize=LEG, labelcolor=INK2, handlelength=1.1,
               labelspacing=0.35, title="MODEL FAMILY", title_fontsize=LEG, alignment="left")

    safe = class_key.replace("/", "_")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fixed_and_crossover_{safe}.{ext}", dpi=300, facecolor=SURFACE)
    plt.close(fig)
    # The caption needs the task's tier/emptiness and the panel scopes, none of which are
    # drawn in the figure any more -- so print them.
    print(f"wrote {out / f'fixed_and_crossover_{safe}.png'}\n"
          f"  (a) {meta_row.short} ({meta_row.tier} tier, {meta_row.frac_empty:.1%} empty): "
          f"{len(present)} estimator curves over 2 models\n"
          f"  (b) {fam.class_key.nunique()} tasks, {len(fam_h)} families")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--perstep", default="outputs/analysis/estimates_perstep.parquet")
    ap.add_argument("--classes", default="outputs/selected_classes.json")
    ap.add_argument("--class", dest="class_key", default="birds/bird",
                    help="task shown in panel (a); any class key in the benchmark")
    ap.add_argument("--out-dir", default="outputs/analysis/figures/results_init")
    ap.add_argument("--budgets", nargs="+", type=float, default=None,
                    help="budget grid for panel (b); default = 16 log-spaced 1%%..50%%")
    args = ap.parse_args()

    budgets = (np.array(sorted(args.budgets)) if args.budgets
               else np.geomspace(0.01, 0.50, 16))
    meta_full = pd.DataFrame([
        {"class_key": c["class_key"], "tier": c["level"], "frac_empty": c["frac_empty"]}
        for c in json.load(open(args.classes))["classes"]])
    if args.class_key not in set(meta_full.class_key):
        raise SystemExit(f"unknown class {args.class_key!r}; choose one of "
                         f"{', '.join(sorted(meta_full.class_key))}")
    meta_row = meta_full[meta_full.class_key == args.class_key].iloc[0].copy()
    meta_row["short"] = (args.class_key.split("/")[1].replace("_", " ").capitalize())

    cols = ["model", "class_key", "sampler", "frozen", "estimator", "frac_labeled",
            "mean_abs_frac_error"]
    perstep = pd.read_parquet(args.perstep, columns=cols)
    dense = dense_summary(perstep, meta_full, budgets)
    fam = _family_envelope(pd.concat([model_best(dense), mc_floor(dense)],
                                     ignore_index=True))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_combined(perstep, fam, budgets, args.class_key, meta_row, out)


if __name__ == "__main__":
    main()
