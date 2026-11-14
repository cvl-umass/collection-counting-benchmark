"""Compile the checkpoint ensembles into the reported summary CSVs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TIERS = ["full", "partial", "sparse", "rare"]
CELL = {("random", False): "random", ("importance", False): "ais",
        ("importance", True): "dis", ("spotcheck", False): "spotais",
        ("spotcheck", True): "spotdis"}


def _read_any(f: Path) -> pd.DataFrame:
    return pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)


def load(input_path: Path) -> pd.DataFrame:
    if input_path.is_dir():  # prefer parquet, fall back to csv
        files = sorted(input_path.glob("*.parquet")) or sorted(input_path.glob("*.csv"))
    else:
        files = [input_path]
    df = pd.concat([_read_any(f) for f in files], ignore_index=True)
    df["sampler"] = [CELL[(s, f)] for s, f in zip(df["sampler"], df["frozen"])]
    return df


def audit(df: pd.DataFrame) -> None:
    problems = []
    if not np.isfinite(df["F_hat"]).all():
        problems.append("non-finite F_hat rows")
    n_traj = df.groupby(["model", "class_key", "sampler"])["seed"].nunique()
    short = n_traj[n_traj != 1000]
    if len(short):
        problems.append(f"groups without 1000 trajectories:\n{short.to_string()}")
    cp = df.groupby(["model", "class_key", "sampler"])["checkpoint"].nunique()
    if (cp != cp.max()).any():
        problems.append("groups with missing checkpoints")
    print(f"audit: {df['seed'].nunique()} distinct seeds, "
          f"{len(n_traj)} model x class x sampler groups, {len(df):,} rows")
    if problems:
        print("AUDIT PROBLEMS:\n" + "\n".join(problems))
        sys.exit(1)
    print("audit: OK")


def summarize(df: pd.DataFrame, classes_meta: pd.DataFrame) -> pd.DataFrame:
    def _agg(g: pd.DataFrame) -> pd.Series:
        fe, real = g["frac_error"], g[g["source"] == "real"]["frac_error"]
        by_base = g.dropna(subset=["base_seed"]).groupby("base_seed")["frac_error"]
        return pd.Series({
            "n_traj": g["seed"].nunique(),
            "n_real": real.size,
            "mean_frac_error": fe.mean(),
            "sd_frac_error": fe.std(),
            "median_frac_error": fe.median(),
            "q25_frac_error": fe.quantile(0.25),
            "q75_frac_error": fe.quantile(0.75),
            "mean_abs_frac_error": fe.abs().mean(),
            "sd_abs_frac_error": fe.abs().std(),
            "real_mean_frac_error": real.mean() if real.size else np.nan,
            "sd_between_base": by_base.mean().std() if by_base.ngroups > 1 else np.nan,
            "sd_within_base": by_base.std().mean() if by_base.ngroups > 1 else np.nan,
            "variance_mean": g["variance"].mean(),
            "F_true": g["F_true"].iloc[0],
        })

    keys = ["model", "class_key", "sampler", "estimator", "checkpoint"]
    try:
        out = df.groupby(keys).apply(_agg, include_groups=False).reset_index()
    except TypeError:  # pandas < 2.2 has no include_groups
        out = df.groupby(keys).apply(_agg).reset_index()
    out[["n_traj", "n_real"]] = out[["n_traj", "n_real"]].astype(int)
    return out.merge(classes_meta, on="class_key").sort_values(
        ["model", "tier_rank", "class_key", "sampler", "estimator", "checkpoint"],
        ignore_index=True)


# Ladder-ordered columns of the wide views: (estimator, sampler, column name);
# sampler None = the directed trajectory (ais for FRCNN, dis for zero-shot).
PICKS = [("model_only_pure", "random", "model_only_pure"),
         ("model_only_corrected", "random", "model_only_corrected"),
         ("monte_carlo", "random", "monte_carlo"),
         ("difference", "random", "difference_random"),
         ("ratio", "random", "ratio_random"),
         ("regression", "random", "regression_random"),
         ("dis", "dis", "dis"),
         ("ais", "ais", "ais"),
         ("difference", None, "difference_directed"),
         ("ratio", None, "ratio_directed"),
         ("regression", None, "regression_directed"),
         ("spotdis", "spotdis", "spotdis"),
         ("spotais", "spotais", "spotais")]


def _pick_values(summary: pd.DataFrame, est: str, sampler: str | None) -> pd.DataFrame:
    """(model, checkpoint, class_key, value) rows for one ladder column;
    value = 'mean ± sd' of |frac_error|, 4 decimals."""
    directed = np.where(summary["model"].str.startswith("frcnn"), "ais", "dis")
    wanted = np.full(len(summary), sampler) if sampler else directed
    mask = (summary["estimator"].to_numpy() == est) & \
           (summary["sampler"].to_numpy() == wanted)
    rows = summary.loc[mask]
    vals = [f"{m:.4f} ± {s:.4f}" if np.isfinite(m) else ""
            for m, s in zip(rows["mean_abs_frac_error"], rows["sd_abs_frac_error"])]
    return rows[["model", "checkpoint", "class_key"]].assign(value=vals)


def widen(summary: pd.DataFrame) -> pd.DataFrame:
    """Ladder-ordered wide view, one row per (model, checkpoint, class)."""
    base = summary[["model", "checkpoint", "class_key", "tier", "tier_rank",
                    "frac_empty", "N", "F_true"]].drop_duplicates()
    for est, sampler, col in PICKS:
        vals = _pick_values(summary, est, sampler).rename(columns={"value": col})
        base = base.merge(vals, on=["model", "checkpoint", "class_key"], how="left")
    return (base.sort_values(["model", "checkpoint", "tier_rank", "class_key"])
                .drop(columns="tier_rank"))


def per_estimator(summary: pd.DataFrame, dest: Path) -> int:
    """Time-centric views: one CSV per ladder column, one row per
    (model, class), checkpoints as columns.  Models without the column's
    sampler (e.g. ais for zero-shot) are omitted rather than left blank."""
    dest.mkdir(parents=True, exist_ok=True)
    meta = summary[["model", "class_key", "tier", "tier_rank",
                    "frac_empty", "N", "F_true"]].drop_duplicates()
    for est, sampler, col in PICKS:
        vals = _pick_values(summary, est, sampler)
        wide = vals.pivot(index=["model", "class_key"], columns="checkpoint",
                          values="value").reset_index()
        wide.columns = ["model", "class_key"] + \
                       [f"at_{100 * c:g}pct" for c in wide.columns[2:]]
        (meta.merge(wide, on=["model", "class_key"], how="inner")
             .sort_values(["model", "tier_rank", "class_key"])
             .drop(columns="tier_rank")
             .to_csv(dest / f"{col}.csv", index=False, encoding="utf-8-sig"))
    return len(PICKS)


def retraining_gain(summary: pd.DataFrame) -> pd.DataFrame:
    """Frozen vs retrained directing per (FRCNN model, class, checkpoint).

    gain = frozen error / retrained error; > 1 means retraining pays.
    """
    frc = summary[summary["model"].str.startswith("frcnn")]
    keys = ["model", "class_key", "checkpoint"]
    out = frc[keys + ["tier", "frac_empty"]].drop_duplicates()
    for est in ["dis", "ais", "spotdis", "spotais"]:
        m = (frc["estimator"] == est) & (frc["sampler"] == est)
        vals = frc.loc[m, keys + ["mean_abs_frac_error"]].rename(
            columns={"mean_abs_frac_error": f"err_{est}"})
        out = out.merge(vals, on=keys, how="left")
    out["gain_plain"] = out["err_dis"] / out["err_ais"]
    out["gain_spot"] = out["err_spotdis"] / out["err_spotais"]
    order = frc[["class_key", "tier_rank"]].drop_duplicates()
    return (out.merge(order, on="class_key")
               .sort_values(["model", "tier_rank", "class_key", "checkpoint"])
               .drop(columns="tier_rank").round(4))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="outputs/analysis/estimates_checkpoints.parquet",
                    help="merged parquet/csv, or a directory of per-(class,model) parquet cells")
    ap.add_argument("--classes", default="outputs/selected_classes.json")
    ap.add_argument("--out-dir", default="outputs/analysis")
    ap.add_argument("--no-audit", action="store_true",
                    help="skip the 1000-trajectory audit (smoke tests)")
    args = ap.parse_args()

    meta = pd.DataFrame([
        {"class_key": c["class_key"], "tier": c["level"],
         "tier_rank": TIERS.index(c["level"]), "frac_empty": c["frac_empty"],
         "N": c["n_units"]}
        for c in json.load(open(args.classes))["classes"]])

    df = load(Path(args.input))
    if not args.no_audit:
        audit(df)

    out_dir = Path(args.out_dir)
    summary = summarize(df, meta)
    summary.to_csv(out_dir / "estimates_summary.csv", index=False,
                   encoding="utf-8-sig")
    wide = widen(summary)
    wide.to_csv(out_dir / "table_headline.csv", index=False, encoding="utf-8-sig")
    n_est = per_estimator(summary, out_dir / "by_estimator")
    gain = retraining_gain(summary)
    gain.to_csv(out_dir / "retraining_gain.csv", index=False,
                encoding="utf-8-sig")
    print(f"wrote {out_dir/'estimates_summary.csv'} ({len(summary):,} rows), "
          f"{out_dir/'table_headline.csv'} ({len(wide):,} rows), "
          f"{n_est} per-estimator files in {out_dir/'by_estimator'}/, and "
          f"{out_dir/'retraining_gain.csv'} ({len(gain):,} rows)")


if __name__ == "__main__":
    main()
