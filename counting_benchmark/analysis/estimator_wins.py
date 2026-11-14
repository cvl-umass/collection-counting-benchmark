"""Which estimator should a model default to?  Win counts over estimator configs."""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "outputs", "analysis", "estimates_summary.csv")

FROZEN = ["aya-vision-8b", "florence2", "gemini-3.1-pro-preview", "gpt-5.2",
          "grounding_dino", "internvl3-5-8b-hf", "molmo-7b-d-0924", "molmo2-8b",
          "owlv2", "qwen3-vl-8b-instruct", "qwen3.7-plus"]
FRCNN = ["frcnn_resnet50", "frcnn_mobilenet"]

# label -> (sampler, estimator).  AIS / spotAIS exist only for the retraining FRCNNs.
CONFIGS = {
    "model-only":  ("random", "model_only_corrected"),
    "monte_carlo": ("random", "monte_carlo"),
    "difference":  ("random", "difference"),
    "ratio":       ("random", "ratio"),
    "regression":  ("random", "regression"),
    "DIS":         ("dis", "dis"),
    "AIS":         ("ais", "ais"),
    "spotDIS":     ("spotdis", "spotdis"),
    "spotAIS":     ("spotais", "spotais"),
}
LABELS = list(CONFIGS)
TOL = 0.05          # top-tier: within 5% relative of the cell best
TIER_ORDER = ["full", "partial", "sparse", "rare"]


def load_cells(models):
    """One row per (model, task, budget) cell, one column of mean-abs-frac-error per
    config, plus the config's SE (sd / sqrt(n_traj)) for the 1-SE cross-check.  Configs a
    model never ran (AIS/spotAIS for frozen-g models) come back as NaN columns.

    The pipeline also carries a FRCNN "frozen at snapshot 0" DIS/spotDIS diagnostic
    cell (a deliberate side comparison, not a candidate estimator per the
    method roster in Table 1), so
    DIS/spotDIS rows are dropped for the two FRCNN models before pivoting -- same
    exclusion model_best() in plot_mllm_summary.py applies."""
    df = pd.read_csv(SUMMARY)
    df.columns = [c.lstrip("﻿") for c in df.columns]
    df = df[df.model.isin(models)].copy()
    df = df[~(df.model.isin(FRCNN) & df.estimator.isin({"dis", "spotdis"}))]
    df["cfg"] = list(zip(df.sampler, df.estimator))
    inv = {v: k for k, v in CONFIGS.items()}
    df = df[df.cfg.isin(inv)].copy()
    df["config"] = df.cfg.map(inv)
    df["se"] = df.sd_abs_frac_error / np.sqrt(df.n_traj)

    key = ["model", "class_key", "tier", "checkpoint"]
    err = df.pivot_table(index=key, columns="config",
                         values="mean_abs_frac_error").reindex(columns=LABELS)
    se = df.pivot_table(index=key, columns="config",
                        values="se").reindex(columns=LABELS)
    return err.reset_index(), se.reset_index()


def per_cell(err):
    """Attach winner, best error, and each config's regret/rank/top-tier flags."""
    E = err[LABELS].to_numpy(float)
    best = np.nanmin(E, axis=1)
    out = err[["model", "class_key", "tier", "checkpoint"]].copy()
    out["winner"] = [LABELS[i] for i in np.nanargmin(E, axis=1)]
    out["best"] = best
    regret = E / best[:, None]
    ranks = pd.DataFrame(E, columns=LABELS).rank(axis=1, method="min")  # 1 = best
    toptier = E <= best[:, None] * (1 + TOL)
    for j, lab in enumerate(LABELS):
        out[f"regret__{lab}"] = regret[:, j]
        out[f"rank__{lab}"] = ranks[lab].to_numpy()
        out[f"top__{lab}"] = toptier[:, j]
    return out


def counts(cell, by=None):
    grp = cell.groupby(by).winner.value_counts().unstack(fill_value=0) if by \
        else cell.winner.value_counts().to_frame().T
    return grp.reindex(columns=LABELS, fill_value=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope", choices=["all", "offtheshelf"], default="all",
                    help="all = 13 models (adds the retraining FRCNNs, 9 configs); "
                         "offtheshelf = the 11 frozen-g models only (7 configs)")
    args = ap.parse_args()
    models = FROZEN if args.scope == "offtheshelf" else FROZEN + FRCNN
    out = os.path.join(ROOT, "outputs", "analysis",
                       "estimator_wins.csv" if args.scope == "all"
                       else "estimator_wins_offtheshelf.csv")

    err, se = load_cells(models)
    cell = per_cell(err)
    cell.to_csv(out, index=False)
    n = len(cell)
    nan_cfg = err[LABELS].isna().sum()
    print(f"[setup] {err.model.nunique()} models ({args.scope}) x {err.class_key.nunique()} tasks "
          f"x {err.checkpoint.nunique()} budgets = {n} cells | metric = mean_abs_frac_error")
    if nan_cfg.any():
        print("  configs absent for some models (frozen-g lack AIS/spotAIS):",
              {k: int(v) for k, v in nan_cfg.items() if v})

    # 1. pooled argmin wins
    w = cell.winner.value_counts().reindex(LABELS, fill_value=0)
    print("\n1) WIN COUNT (argmin over a cell's available configs), pooled over all cells:")
    for lab in sorted(LABELS, key=lambda l: -w[l]):
        print(f"    {lab:12s} {w[lab]:4d}  ({w[lab]/n:5.1%})")

    # 2. wins by budget
    print("\n2) WINS by budget (rows = config, cols = labeling budget):")
    bb = counts(cell, "checkpoint").T
    bb.columns = [f"{c:.0%}" for c in bb.columns]
    print(bb.loc[[l for l in LABELS if bb.loc[l].sum() > 0]].to_string())

    # 3. wins by tier
    print("\n3) WINS by occupancy tier:")
    tb = counts(cell, "tier").reindex(TIER_ORDER)
    tb["(cells)"] = cell.groupby("tier").size().reindex(TIER_ORDER)
    print(tb[[l for l in LABELS if tb[l].sum() > 0] + ["(cells)"]].to_string())

    # 4. top-tier membership (within 5% of best) + 1-SE cross-check
    within_se = (err[LABELS].to_numpy() <= (cell.best.to_numpy()[:, None] + se[LABELS].to_numpy()))
    print(f"\n4) TOP-TIER membership rate (config is within {TOL:.0%} of the cell best):")
    print("   (1-SE column = within one config-SE of best; anti-conservative, trajectories are correlated)")
    for lab in LABELS:
        r5 = cell[f"top__{lab}"].mean()
        rse = within_se[:, LABELS.index(lab)].mean()
        print(f"    {lab:12s} within5%={r5:5.1%}   within1SE={rse:5.1%}")

    # 5. regret and mean rank
    print("\n5) REGRET (config_err / cell_best) and MEAN RANK (1=best; among a model's available configs):")
    print(f"    {'config':12s} {'median':>7} {'mean':>7} {'p90':>7} {'meanrank':>9}")
    for lab in sorted(LABELS, key=lambda l: np.nanmedian(cell[f'regret__{l}'])):
        rg = cell[f"regret__{lab}"]
        print(f"    {lab:12s} {np.nanmedian(rg):7.3f} {np.nanmean(rg):7.3f} "
              f"{np.nanpercentile(rg, 90):7.3f} {cell[f'rank__{lab}'].mean():9.2f}")

    # 6. DIS regret by tier -- is DIS a safe default where the pilot proxy is usable?
    print("\n6) DIS regret by tier (median | p90):")
    for t in TIER_ORDER:
        rg = cell.loc[cell.tier == t, "regret__DIS"]
        print(f"    {t:8s} {np.nanmedian(rg):.3f} | {np.nanpercentile(rg, 90):.3f}")
    print(f"\n[wrote] {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
