"""Log-log detector-vs-MLLM error from the per-step parquet (dense low-budget grid)."""

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
from plot_mllm_summary import (  # noqa: E402
    FAM_ORDER, FAM_COLOR, FAM_LABEL, INK, INK2, SURFACE, style, class_meta,
    model_best, mc_floor, _family_envelope)

# Raw (sampler, frozen) -> the single sampler label used everywhere else
# (mirrors summarize_checkpoints.CELL).
CELL = {("random", False): "random", ("importance", False): "ais",
        ("importance", True): "dis", ("spotcheck", False): "spotais",
        ("spotcheck", True): "spotdis"}


def dense_summary(perstep: pd.DataFrame, meta: pd.DataFrame,
                  budgets: np.ndarray) -> pd.DataFrame:
    """Resample the per-step ensemble onto ``budgets``.

    For each (model, class, sampler, estimator) trajectory, take the first step
    at or past each target budget (same rule as ``collect_sims.checkpoint_rows``),
    dropping budgets below the trajectory's first available step so we never
    extrapolate past the data.  Returns a summary-shaped frame with the columns
    ``model_best``/``mc_floor`` expect."""
    df = perstep.copy()
    df["sampler"] = [CELL[(s, f)] for s, f in zip(df["sampler"], df["frozen"])]
    parts = []
    for key, g in df.groupby(["model", "class_key", "sampler", "estimator"],
                             sort=False):
        g = g.sort_values("frac_labeled")
        fl = g["frac_labeled"].to_numpy()
        keep = budgets >= fl[0]          # no extrapolation below the first step
        idx = np.searchsorted(fl, budgets[keep], side="left")
        idx = np.clip(idx, 0, len(g) - 1)
        rows = g.iloc[idx][["mean_abs_frac_error", "frac_labeled"]].copy()
        rows["checkpoint"] = budgets[keep]
        for k, v in zip(["model", "class_key", "sampler", "estimator"], key):
            rows[k] = v
        parts.append(rows)
    out = pd.concat(parts, ignore_index=True).merge(meta, on="class_key")
    return out


def fig_loglog(fam: pd.DataFrame, meta: pd.DataFrame, budgets: np.ndarray,
               out: Path, nrow: int = 4, ncol: int = 3,
               fname: str = "detector_vs_mllm_loglog.png",
               figsize: tuple = (14.5, 13)) -> None:
    style_kw = {"detector": ("-", 2.4, 1.0), "open_mllm": ("-", 2.2, 1.0),
                "prop_mllm": ("-", 2.2, 1.0), "vlm": ("--", 1.4, 0.75)}
    xt = [t for t in (1, 2, 5, 10, 20, 50) if t >= 100 * budgets[0] - 1e-9]
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, facecolor=SURFACE,
                             squeeze=False)
    for ax in axes.flat[len(meta):]:      # hide any panels past the class count
        ax.axis("off")
    for ax, (_, m) in zip(axes.flat, meta.iterrows()):
        style(ax)
        sub = fam[fam.class_key == m.class_key]
        mc = sub[sub.family == "mc_floor"].sort_values("checkpoint")
        if len(mc):
            ax.plot(100 * mc.checkpoint, mc.err, ":", color=FAM_COLOR["mc_floor"],
                    linewidth=1.8, label=FAM_LABEL["mc_floor"], zorder=1)
        for f in FAM_ORDER:
            s = sub[sub.family == f].sort_values("checkpoint")
            if not len(s):
                continue
            ls, lw, alpha = style_kw[f]
            ax.plot(100 * s.checkpoint, s.err, ls, color=FAM_COLOR[f],
                    linewidth=lw, alpha=alpha, marker="o", markersize=3,
                    label=FAM_LABEL[f], zorder=3 if f != "vlm" else 2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(xt)
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlim(100 * budgets[0] * 0.9, 58)
        ax.set_title(f"{m.class_key}  ({m.tier}, {m.frac_empty:.0%} empty)",
                     fontsize=9, color=INK)
    for ax in axes[-1]:
        ax.set_xlabel("% labeled (log)", fontsize=9, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("best mean |frac error| in family", fontsize=9, color=INK2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("When does the fine-tuned detector overtake the MLLMs?  "
                 "(log-log)\n(each point = the best model+estimator in that family "
                 "at that budget; the winning model and estimator combo can change across budgets)",
                 y=1.05, fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(out / fname, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


MIN_FRAC = 0.67          # >= 2/3 of the tasks must be present to draw a point
# Line style per family in the task-averaged view; mc_floor is dotted (a floor, not a
# competitor).  Shared with the combined figure so both draw the same curves.
MEAN_STYLE = {"mc_floor": (":", 2.0), "detector": ("-", 2.6), "open_mllm": ("-", 2.2),
              "prop_mllm": ("-", 2.2), "vlm": ("-", 2.2)}
MEAN_LABEL = {"mc_floor": "MC (no model)",
              **{f: FAM_LABEL[f].split(",")[0].split("(")[0].strip() for f in FAM_ORDER}}


def family_means(fam: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per family, the task-averaged best-member error over budget.

    At each budget, the mean over tasks of that family's best-member error (a mean of the
    per-panel minimums), kept only where at least ``MIN_FRAC`` of the tasks have data.  The
    MLLM/OVD families cover all tasks from ~3% on; the fine-tuned detector needs enough
    labels before its first checkpoint, so at the lowest budgets its mean is over fewer
    tasks (the ``n`` column, whose range belongs in the caption)."""
    floor = max(1, round(MIN_FRAC * fam.class_key.nunique()))
    out = {}
    for f in ["mc_floor"] + FAM_ORDER:
        g = (fam[fam.family == f].groupby("checkpoint").err
             .agg(mean="mean", n="size").reset_index())
        out[f] = g[g.n >= floor].sort_values("checkpoint")
    return out


# Painting order alone would bury the detector (it is drawn first of the families), and it is
# the curve the figure is about -- so it gets its own layer above the rest.  mc_floor is a
# floor, so it stays at the back.
MEAN_Z = {"mc_floor": 1, "detector": 5}
MEAN_Z_DEFAULT = 3


def draw_means(ax, means: dict[str, pd.DataFrame], *, markersize: float = 3.5,
               quiet: bool = False, n_tasks: int | None = None) -> None:
    """Draw the family-mean curves onto ``ax`` (axis scales/labels left to the caller)."""
    for f, s in means.items():
        if s.empty:
            continue
        ls, lw = MEAN_STYLE[f]
        ax.plot(100 * s.checkpoint, s["mean"], ls, color=FAM_COLOR[f], linewidth=lw,
                marker=None if f == "mc_floor" else "o", markersize=markersize,
                label=MEAN_LABEL[f], zorder=MEAN_Z.get(f, MEAN_Z_DEFAULT))
        if not quiet:
            print(f"  {f:10s}: {len(s)} budgets from {100*s.checkpoint.min():.1f}% "
                  f"(tasks/point: {s.n.min()}-{s.n.max()} of {n_tasks})")


def fig_mean(fam: pd.DataFrame, budgets: np.ndarray, out: Path,
             fname: str = "detector_vs_mllm_loglog_mean.png") -> None:
    """Collapse the 12 per-task panels into one task-averaged panel (see ``family_means``)."""
    xt = [t for t in (1, 2, 5, 10, 20, 50) if t >= 100 * budgets[0] - 1e-9]

    fig, ax = plt.subplots(figsize=(6.4, 3.5), facecolor=SURFACE)   # flat, single-column
    style(ax)
    ax.tick_params(labelsize=14)
    draw_means(ax, family_means(fam), n_tasks=fam.class_key.nunique())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(xt)
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(100 * budgets[0] * 0.9, 58)
    ax.set_xlabel("% labeled (log)", fontsize=16, color=INK2)
    ax.set_ylabel("mean best-in-family\n|fractional error|", fontsize=16, color=INK2)
    ax.legend(frameon=False, fontsize=10, labelcolor=INK2, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / fname.replace(".png", f".{ext}"), dpi=300,
                    bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--perstep", default="outputs/analysis/estimates_perstep.parquet")
    ap.add_argument("--classes", default="outputs/selected_classes.json")
    ap.add_argument("--out-dir", default="outputs/analysis/figures/diagnostic",
                    help="where the full 12-panel goes")
    ap.add_argument("--subset-out-dir", default="outputs/analysis/figures/results_init",
                    help="where the --only 2x2 subset (a paper figure) goes")
    ap.add_argument("--csv-dir", default="outputs/analysis")
    ap.add_argument("--budgets", nargs="+", type=float, default=None,
                    help="budget grid (fractions); default = 16 log-spaced 1%%..50%%")
    ap.add_argument("--only", nargs="+", default=None, metavar="CLASS_KEY",
                    help="also emit a compact 2x2 subset (detector_vs_mllm_loglog_4panel.png) "
                         "for these class keys, e.g. --only malaria/trophozoite dota/harbor "
                         "dota/small_vehicle dota/storage_tank")
    args = ap.parse_args()

    budgets = (np.array(sorted(args.budgets)) if args.budgets
               else np.geomspace(0.01, 0.50, 16))
    meta_full = pd.DataFrame([
        {"class_key": c["class_key"], "tier": c["level"],
         "frac_empty": c["frac_empty"]}
        for c in json.load(open(args.classes))["classes"]])

    cols = ["model", "class_key", "sampler", "frozen", "estimator",
            "frac_labeled", "mean_abs_frac_error"]
    perstep = pd.read_parquet(args.perstep, columns=cols)
    dense = dense_summary(perstep, meta_full, budgets)

    combined = pd.concat([model_best(dense), mc_floor(dense)], ignore_index=True)
    fam = _family_envelope(combined)
    fam[["class_key", "tier", "frac_empty", "pct_labeled", "family",
         "model", "estimator", "sampler", "err", "family_rank"]].replace(
        {"__mc__": "(uniform_monte_carlo)"}).to_csv(
        Path(args.csv_dir) / "detector_vs_mllm_loglog.csv", index=False)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = class_meta(dense)
    fig_loglog(fam, meta, budgets, out)
    print(f"wrote {out/'detector_vs_mllm_loglog.png'} "
          f"({len(budgets)} budgets: {', '.join(f'{100*b:g}' for b in budgets)}%)")

    mean_out = Path(args.subset_out_dir)
    mean_out.mkdir(parents=True, exist_ok=True)
    fig_mean(fam, budgets, mean_out)
    print(f"wrote {mean_out/'detector_vs_mllm_loglog_mean.png'} (12 tasks averaged)")
    if args.only:
        sub_out = Path(args.subset_out_dir)
        sub_out.mkdir(parents=True, exist_ok=True)
        sub = meta[meta.class_key.isin(args.only)]
        fig_loglog(fam, sub, budgets, sub_out, nrow=2, ncol=2, figsize=(9.5, 8.6),
                   fname="detector_vs_mllm_loglog_4panel.png")
        print(f"wrote {sub_out/'detector_vs_mllm_loglog_4panel.png'} "
              f"({len(sub)} classes: {', '.join(sub.class_key)})")


if __name__ == "__main__":
    main()
