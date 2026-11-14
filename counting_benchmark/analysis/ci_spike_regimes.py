"""Why the spot-checked proposals under-cover: the two trajectory regimes.

Splits each cell's trajectories on whether one step dominates the variance,
and reports coverage and landing separately for the two groups.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

FRCNN = ("frcnn_resnet50", "frcnn_mobilenet")
TIERS = ("sparse", "rare")
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def cell_rows(path: Path, tier_of: dict, z: float, spike_share: float):
    d = np.load(path, allow_pickle=True)
    cls, model = str(d["meta__class_key"]), str(d["meta__model"])
    tier = tier_of.get(cls)
    if tier not in TIERS:
        return []
    # roster: spotAIS is the adaptive detector variant, spotDIS the frozen-model one.
    if model in FRCNN:
        key, est, sqrt_index = "f__spotcheck__0__spotais", "spotAIS", "main"
    else:
        key, est, sqrt_index = "f__spotcheck__1__spotdis", "spotDIS", "none"
    if key not in d:
        return []

    per = d[key].astype(np.float64)
    N, w, F_true = int(d["meta__N"]), int(d["meta__w"]), float(d["meta__F_true"])
    if F_true <= 0:
        return []
    n, T = per.shape

    i = np.arange(1, T + 1, dtype=np.float64)
    a = float(N) - float(w) - i
    alpha = 1.0 / (a * (a + 1.0))
    if sqrt_index == "main":
        alpha = alpha * np.sqrt(i)
    A = np.cumsum(alpha)
    F_run = np.cumsum(alpha * per, axis=1) / A          # combined estimate at every budget

    out = []
    for cp in BUDGETS:
        t = min(max(2, int(round(cp * N))), T)
        F_hat = F_run[:, t - 1]
        terms = (alpha[:t] ** 2) * (per[:, :t] - F_hat[:, None]) ** 2
        tot = terms.sum(1)
        good = tot > 0
        if not good.any():
            continue
        V = tot / A[t - 1] ** 2
        top1 = np.divide(terms.max(1), tot, out=np.zeros(n), where=good)
        half = z * np.sqrt(V) / F_true                   # CI half-width, relative to truth
        err = (F_hat - F_true) / F_true
        hit = np.abs(err) <= half
        spiked = (top1 > spike_share) & good

        for label, mask in (("spiked", spiked & good), ("not spiked", ~spiked & good)):
            if not mask.any():
                continue
            out.append(dict(
                class_key=cls, model=model, estimator=est, tier=tier, checkpoint=cp,
                group=label, n_traj=int(mask.sum()), share=float(mask.mean()),
                mean_rel_error=float(err[mask].mean()),
                median_half_width=float(np.median(half[mask])),
                coverage=float(hit[mask].mean()),
            ))
        out.append(dict(
            class_key=cls, model=model, estimator=est, tier=tier, checkpoint=cp,
            group="all", n_traj=int(good.sum()), share=1.0,
            mean_rel_error=float(err[good].mean()),
            median_half_width=float(np.median(half[good])),
            coverage=float(hit[good].mean()),
        ))
    return out


def summarize(df: pd.DataFrame) -> None:
    """Trajectory-weighted pooling: every trajectory counts once."""
    def table(sub: pd.DataFrame) -> pd.DataFrame:
        """Trajectory-weighted pooling within ``sub``; shares are relative to its own total."""
        denom = sub[sub.group == "all"].n_traj.sum()

        def pooled(g):
            wgt = g.n_traj
            return pd.Series({
                "trajectories": int(wgt.sum()),
                "share": wgt.sum() / denom,
                "mean_rel_error": np.average(g.mean_rel_error, weights=wgt),
                "median_half_width": np.average(g.median_half_width, weights=wgt),
                "coverage": np.average(g.coverage, weights=wgt),
            })

        return (sub.groupby("group", sort=False).apply(pooled, include_groups=False)
                .reindex(["not spiked", "spiked", "all"]))

    print("=== pooled over spotDIS/spotAIS cells on sparse+rare tasks, all budgets ===\n")
    print(table(df).round(4).to_string(), "\n")

    print("=== by estimator ===")
    for est in ("spotDIS", "spotAIS"):
        sub = df[df.estimator == est]
        if sub.empty:
            continue
        print(f"\n-- {est} --")
        print(table(sub)[["share", "mean_rel_error", "median_half_width",
                          "coverage"]].round(4).to_string())

    ns = df[df.group == "not spiked"]
    print("\n=== spread across cells (not-spiked group), 10th-90th pct ===")
    for col in ("share", "mean_rel_error", "median_half_width", "coverage"):
        lo, hi = np.quantile(ns[col], [0.1, 0.9])
        print(f"  {col:20s} {lo:8.3f} .. {hi:8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perstep", default="outputs/analysis/perstep_Fhat")
    ap.add_argument("--summary", default="outputs/analysis/estimates_summary.csv")
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--spike-share", type=float, default=0.9,
                    help="one step supplying more than this fraction of Var_simp")
    ap.add_argument("--out", default="outputs/analysis/ci_spike_regimes.csv")
    args = ap.parse_args()

    summ = pd.read_csv(args.summary)
    summ.columns = [c.lstrip("﻿") for c in summ.columns]
    tier_of = dict(summ[["class_key", "tier"]].drop_duplicates().itertuples(index=False))

    z = norm.ppf(0.5 + args.level / 2.0)
    files = sorted(glob.glob(str(Path(args.perstep) / "*.npz")))
    if not files:
        # Fail before writing: the shipped --out is the paper's table, and an
        # empty rebuild would silently replace it.  This cache is not shipped;
        # rebuild it with scripts/collect_sims.py (see README stage 5).
        raise SystemExit(
            f"no *.npz under {args.perstep!r} -- nothing to summarize.\n"
            "This per-step cache is regenerated by `python3 scripts/collect_sims.py`; "
            "run that first, or point --perstep at an existing cache.")
    rows = []
    for f in files:
        rows.extend(cell_rows(Path(f), tier_of, z, args.spike_share))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    summarize(df)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
