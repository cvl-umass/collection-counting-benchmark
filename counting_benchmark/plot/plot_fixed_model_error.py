"""Raw estimation error for a fixed model and task -- an un-aggregated view."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import (EST_COLOR, EST_LABEL, EST_ORDER, FAMILY, INK, INK2,  # noqa: E402
                     SURFACE, style)

# The two models the paper contrasts: best of the fine-tuned and the zero-shot worlds.
MODELS = [("frcnn_resnet50", "Faster R-CNN (ResNet-50)"),
          ("internvl3-5-8b-hf", "InternVL3.5-8B")]

# (sampler, estimator) in the per-step store -> the palette/estimator key.  This is the
# valid roster: the five uniform estimators plus each model's importance samplers.
VALID = {("random", "model_only_corrected"): "model_only_corrected",
         ("random", "monte_carlo"): "monte_carlo",
         ("random", "difference"): "difference",
         ("random", "ratio"): "ratio",
         ("random", "regression"): "regression",
         ("importance", "ais"): "ais", ("importance", "dis"): "dis",
         ("spotcheck", "spotais"): "spotais", ("spotcheck", "spotdis"): "spotdis"}
# AIS/spotAIS only make sense for a retraining FRCNN detector; DIS/spotDIS are the
# frozen-model estimator (see the method roster in Table 1).  FRCNN also
# carries a "frozen at snapshot 0" DIS/spotDIS diagnostic cell (a deliberate,
# separate comparison), which VALID would otherwise
# also pick up here -- excluded per model below, same as plot_mllm_summary.model_best.
ADAPTIVE_ESTIMATORS = {"ais", "spotais"}
FROZEN_ESTIMATORS = {"dis", "spotdis"}

LBL, TCK = 16, 14


def valid_rows(perstep: pd.DataFrame, class_key: str) -> pd.DataFrame:
    """One task's rows, tagged with the palette estimator key (invalid pairs dropped)."""
    df = perstep[perstep.class_key == class_key].copy()
    df["est"] = [VALID.get((s, e)) for s, e in zip(df.sampler, df.estimator)]
    return df[df.est.notna()]


def draw_model_panel(ax, df: pd.DataFrame, model: str, *,
                     linewidth: float = 2.0) -> list[str]:
    """One model's estimator curves on ``ax``; returns the estimators actually drawn.

    Axis scale and labels are left to the caller so the combined figure can size them
    for a single column."""
    sub = df[df.model == model]
    wrong_regime = FROZEN_ESTIMATORS if FAMILY[model] == "detector" else ADAPTIVE_ESTIMATORS
    sub = sub[~sub.est.isin(wrong_regime)]
    drawn = []
    for est in EST_ORDER:
        s = sub[sub.est == est].sort_values("frac_labeled")
        if s.empty:
            continue
        drawn.append(est)
        ax.plot(100 * s.frac_labeled, s.mean_abs_frac_error, "-",
                color=EST_COLOR[est], linewidth=linewidth, zorder=3)
    return drawn


def fig_fixed(perstep: pd.DataFrame, class_key: str, out: Path) -> None:
    df = valid_rows(perstep, class_key)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharey=True, facecolor=SURFACE)
    present = []
    for ax, (model, label) in zip(axes, MODELS):
        style(ax)
        ax.tick_params(labelsize=TCK)
        present += draw_model_panel(ax, df, model)
        ax.set_yscale("log")
        ax.set_xlim(0, 50)
        ax.set_xlabel("% labeled", fontsize=LBL + 4, color=INK2)
        ax.set_title(label, fontsize=LBL, color=INK)
    axes[0].set_ylabel("mean |fractional error|", fontsize=LBL, color=INK2)

    # reversed(EST_ORDER) so the legend reads model-only -> spotAIS top-to-bottom,
    # matching the estimator legend in the overview figure.
    order = [e for e in reversed(EST_ORDER) if e in present]
    handles = [plt.Line2D([], [], color=EST_COLOR[e], linewidth=6, label=EST_LABEL[e])
               for e in order]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.87, 0.5),
               frameon=False, fontsize=TCK, labelcolor=INK2, handlelength=1.3,
               labelspacing=0.5, title="ESTIMATOR", title_fontsize=TCK, alignment="left")
    fig.subplots_adjust(left=0.10, right=0.86, top=0.90, bottom=0.16, wspace=0.08)
    safe = class_key.replace("/", "_")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fixed_model_error_{safe}.{ext}", dpi=150, facecolor=SURFACE,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / f'fixed_model_error_{safe}.png'}  "
          f"({df.model.nunique()} models, {len(order)} estimators shown)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--perstep", default="outputs/analysis/estimates_perstep.parquet")
    ap.add_argument("--class", dest="class_key", default="birds/bird")
    ap.add_argument("--out-dir", default="outputs/analysis/figures/results_init")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["model", "class_key", "sampler", "estimator", "frac_labeled",
            "mean_abs_frac_error"]
    fig_fixed(pd.read_parquet(args.perstep, columns=cols), args.class_key, out)


if __name__ == "__main__":
    main()
