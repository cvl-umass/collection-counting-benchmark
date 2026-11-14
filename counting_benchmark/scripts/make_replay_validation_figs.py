"""Figures and readable tables for the replay-validation results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot import palette as pal  # noqa: E402

# path-invariance raw sampler names -> paper sampler names, and each sampler's own
# estimator (the one the central table reports).
PATH_INVARIANCE_SAMPLER = {"importance": "ais", "random": "random", "spotcheck": "spotais"}
CORE_EST = {"ais": "ais", "spotais": "spotais", "random": "monte_carlo",
            "dis": "dis", "spotdis": "spotdis"}
TEST_CPS = ["0.10", "0.30", "0.50"]

MODEL_LABEL = {"frcnn_mobilenet": "Faster R-CNN (MobileNetV3)",
               "frcnn_resnet50": "Faster R-CNN (ResNet-50)"}

# The sampler each estimator is reported on, for the estimator-sampler pairs the
# benchmark actually evaluates: the importance-sampling estimators run on their
# own proposals, everything else on uniform draws.  DIS/spotDIS are absent
# because they have no genuine runs to compare against -- conditional on the
# cache their replay IS the true process.  model_only is kept on purpose: it is
# the failure exhibit.
CANONICAL_SAMPLER = {"ais": "ais", "spotais": "spotais", "regression": "random",
                     "ratio": "random", "difference": "random",
                     "monte_carlo": "random", "model_only_corrected": "random"}
# Rows shown in the two overview figures.  reversed(EST_ORDER) is the house
# top-to-bottom order (model-only first, spotAIS last), matching the results
# figures' legends.
CORE = [(CANONICAL_SAMPLER[e], e) for e in reversed(pal.EST_ORDER)
        if e in CANONICAL_SAMPLER]
# Local, figure-only encoding: the dot fill carries the estimator (house
# palette), while a single warm accent flags landings outside the +/-2 band.
# EST_COLOR is an all-cool ramp plus grays, so the accent cannot be confused
# with an estimator.
FLAG = "#d1495b"


def fig_landing_positions(pb: pd.DataFrame, cp: str, out: Path) -> None:
    """Landing positions per (model, canonical sampler-estimator pair).

    No in-plot title: the figure is a paper float and its caption carries the
    reading instructions.  Renders both .png (screen) and .pdf (print).
    """
    sub = pb[pb["checkpoint"] == cp]
    models = sorted(sub["model"].unique())
    rows = [(s, e) for s, e in CORE
            if ((sub["sampler"] == s) & (sub["estimator"] == e)).any()]
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 0.30 * len(rows) + 1.0),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models):
        pal.style(ax)
        ax.grid(False, axis="y")
        ax.axvspan(-2, 2, color=pal.GRID, alpha=0.55, lw=0, zorder=0)
        ax.axvline(0, color=pal.AXIS, lw=0.9, zorder=1)
        for y, (s, e) in enumerate(rows):
            u = sub[(sub["model"] == model) & (sub["sampler"] == s)
                    & (sub["estimator"] == e)]["u"].to_numpy(float)
            u = u[np.isfinite(u)]
            jitter = (np.random.default_rng(0).uniform(-0.11, 0.11, u.size))
            out_of = np.abs(u) > 2
            ax.scatter(u[~out_of], y + jitter[~out_of], s=9, color=pal.EST_COLOR[e],
                       alpha=0.85, lw=0.25, edgecolor=pal.SURFACE, zorder=2)
            ax.scatter(u[out_of], y + jitter[out_of], s=12, color=FLAG,
                       alpha=0.9, lw=0.25, edgecolor=pal.SURFACE, zorder=3)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([pal.EST_LABEL[e] for _, e in rows], fontsize=8,
                           color=pal.INK)
        ax.set_title(MODEL_LABEL.get(model, model), fontsize=9, color=pal.INK, pad=4)
        ax.set_xlim(-6, 6)
    # The y axis is shared, so set its limits once: per-axes invert_yaxis calls
    # would cancel out and silently reverse CORE's display order.
    axes[0].set_ylim(len(rows) - 0.4, -0.6)
    # Both panels plot the same quantity, so the axis is labelled once, centred.
    fig.tight_layout(pad=0.4, w_pad=1.0, rect=(0, 0.055, 1, 1))
    fig.supxlabel("Genuine run's estimate, in standard deviations from its own replays' mean",
                  fontsize=9, color=pal.INK, y=0.015)
    stem = f"landing_positions_{cp.replace('.', '')}"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor=pal.SURFACE)
    plt.close(fig)


def fig_rank_histograms(pb: pd.DataFrame, cp: str, out: Path) -> None:
    sub = pb[pb["checkpoint"] == cp]
    models = sorted(sub["model"].unique())
    rows = [(s, e) for s, e in CORE
            if ((sub["sampler"] == s) & (sub["estimator"] == e)).any()]
    ncol = len(rows)
    fig, axes = plt.subplots(len(models), ncol, figsize=(1.75 * ncol, 2.0 * len(models)),
                             sharex=True, sharey=True, squeeze=False)
    for i, model in enumerate(models):
        for j, (s, e) in enumerate(rows):
            ax = axes[i][j]
            pit = sub[(sub["model"] == model) & (sub["sampler"] == s)
                      & (sub["estimator"] == e)]["pit"].to_numpy(float)
            pit = pit[np.isfinite(pit)]
            ax.hist(pit, bins=5, range=(0, 1), color="tab:blue", alpha=0.75)
            ax.axhline(pit.size / 5.0, color="0.3", lw=1, ls="--")
            if i == 0:
                ax.set_title(pal.EST_LABEL[e], fontsize=7)
            if j == 0:
                ax.set_ylabel(MODEL_LABEL.get(model, model), fontsize=7)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"Rank positions of real runs inside their replay clouds "
                 f"(checkpoint {cp}; flat at the dashed line = pass; "
                 f"bars piled left = real runs sit below their clouds)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out / f"rank_histograms_{cp.replace('.', '')}.png", dpi=150)
    plt.close(fig)


def fig_bands(band: pd.DataFrame, model: str, sampler: str, est: str,
              base: int, out: Path) -> None:
    sub = band[(band["model"] == model) & (band["sampler"] == sampler)
               & (band["estimator"] == est) & (band["base_seed"] == base)]
    if sub.empty:
        return
    classes = sorted(sub["class_key"].unique())
    ncol = 4
    nrow = int(np.ceil(len(classes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.3 * nrow),
                             sharex=True, squeeze=False)
    for k, cls in enumerate(classes):
        ax = axes[k // ncol][k % ncol]
        g = sub[sub["class_key"] == cls].sort_values("frac_labeled")
        ax.fill_between(g["frac_labeled"], g["fe_p05"], g["fe_p95"],
                        color="tab:blue", alpha=0.25, lw=0,
                        label="replay middle 90%")
        ax.plot(g["frac_labeled"], g["fe_p50"], color="tab:blue", lw=0.9,
                ls="--", label="replay median")
        ax.plot(g["frac_labeled"], g["fe_real"], color="tab:red", lw=1.2,
                label="real run")
        ax.axhline(0, color="0.4", lw=0.6)
        ax.set_title(cls, fontsize=8)
        ax.tick_params(labelsize=7)
        lim = max(0.05, np.percentile(np.abs(g[["fe_p05", "fe_p95", "fe_real"]]
                                             .to_numpy()), 99) * 1.2)
        ax.set_ylim(-lim, lim)
    for k in range(len(classes), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    axes[0][0].legend(fontsize=7, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("fraction labeled", fontsize=8)
    fig.suptitle(f"Real run vs its replay band -- {model}, {sampler} sampler, "
                 f"{est} estimator, base {base} (fractional error vs budget)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / f"bands__{model}__{sampler}__{est}.png", dpi=150)
    plt.close(fig)


def build_central_table(in_dir: Path, path_invariance_csv: Path, out: Path) -> pd.DataFrame:
    pooled = pd.read_csv(in_dir / "pooled.csv")
    stab = pd.read_csv(in_dir / "path_stability.csv")

    la = None
    if path_invariance_csv.exists():
        la = pd.read_csv(path_invariance_csv)
        la["model"] = la["regime"].str.split("__stride32__").str[0]
        la["sampler"] = la["sampler"].map(PATH_INVARIANCE_SAMPLER)
        la = la.assign(dist=(la["frac_labeled"] - 0.30).abs())
        la = la.loc[la.groupby(["model", "sampler", "dataset", "cls"])["dist"].idxmin()]

    rows = []
    pairs = (stab[["model", "sampler"]].drop_duplicates().itertuples(index=False))
    for model, sampler in sorted(pairs):
        est = CORE_EST.get(sampler)
        row = dict(model=model, sampler=sampler, estimator=est)
        if la is not None:
            g = la[(la["model"] == model) & (la["sampler"] == sampler)]
            if len(g):
                row["prop_tv_med30"] = float(g["prop_tv_mean"].median())
                row["pearson_g_med30"] = float(g["pearson_mean"].median())
        s = stab[(stab["model"] == model) & (stab["sampler"] == sampler)
                 & (stab["estimator"] == est) & (stab["checkpoint"] == "0.30")]
        if len(s):
            row["path_mean_shift_med30"] = float(s["range_m_sigma"].median())
            row["path_var_ratio_med30"] = float(s["vratio_maxmin"].median())
        p = pooled[(pooled["model"] == model) & (pooled["sampler"] == sampler)
                   & (pooled["estimator"] == est)
                   & (pooled["checkpoint"].isin(TEST_CPS))]
        if len(p):
            row["worst_abs_z_pool"] = float(p["z_pool"].abs().max())
            row["d_pool_min"] = float(p["d_pool"].min())
            row["d_pool_max"] = float(p["d_pool"].max())
            row["gof_p_min"] = float(p["gof_chi2_p"].min())
            row["verdicts"] = "/".join(p.sort_values("checkpoint")["verdict"])
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.round(3).to_csv(out / "tables" / "central_table.csv", index=False)
    return tab


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default="outputs/analysis/replay_validation")
    ap.add_argument("--out-dir", default="outputs/analysis/figures/results_init")
    ap.add_argument("--path-invariance", default="outputs/analysis/path_invariance/per_checkpoint.csv",
                    help="path-invariance per-checkpoint CSV for the central table's "
                         "path-interchangeability block (skipped if absent)")
    ap.add_argument("--checkpoint", default="0.30",
                    help="checkpoint for the two overview figures")
    ap.add_argument("--band-base", type=int, default=0,
                    help="which base's band/real pair to draw in the band figures")
    args = ap.parse_args()

    in_dir, out = Path(args.in_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)

    pb = pd.read_csv(in_dir / "per_base.csv")
    pooled = pd.read_csv(in_dir / "pooled.csv")

    fig_landing_positions(pb, args.checkpoint, out)
    fig_rank_histograms(pb, args.checkpoint, out)

    band_path = in_dir / "band_data.parquet"
    if band_path.exists():
        band = pd.read_parquet(band_path)
        for model in sorted(band["model"].unique()):
            fig_bands(band, model, "ais", "ais", args.band_base, out)
            fig_bands(band, model, "spotais", "spotais", args.band_base, out)
        # the failure exhibit: model_only riding outside its bands
        fig_bands(band, "frcnn_mobilenet", "random", "model_only_corrected",
                  args.band_base, out)

    head = pooled[pooled["checkpoint"].isin(["0.10", "0.30", "0.50", "auc"])]
    idx = ["model", "sampler", "estimator"]
    (head.pivot_table(index=idx, columns="checkpoint", values="verdict",
                      aggfunc="first")
         .to_csv(out / "tables" / "verdicts.csv"))
    zd = head.pivot_table(index=idx, columns="checkpoint",
                          values=["z_pool", "d_pool"])
    zd.columns = [f"{a}@{b}" for a, b in zd.columns]
    zd.round(2).to_csv(out / "tables" / "z_and_d.csv")

    tab = build_central_table(in_dir, Path(args.path_invariance), out)
    with pd.option_context("display.width", 200,
                           "display.float_format", lambda v: f"{v:.3f}"):
        print("\n=== central table (tables/central_table.csv) ===")
        print(tab.to_string(index=False))
    print(f"\nwrote figures + tables under {out}")


if __name__ == "__main__":
    main()
