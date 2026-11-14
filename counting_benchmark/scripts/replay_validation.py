"""Replay validation: do replays reproduce genuine end-to-end runs?

Takes each cached model path as given and compares a genuine run against its
replays two ways -- coverage against the other paths' replays, and landing
shift against its own.  Frozen models and Monte Carlo need no approximation,
so only the fine-tuned detector cells are at issue.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.derive_estimates import PASSTHROUGH, combine_matrix, _sqrt_index  # noqa: E402

# (raw npz sampler, frozen) -> the sampler name used in the paper.
SAMPLER_LABEL = {
    ("random", 0): "random",
    ("importance", 0): "ais",
    ("importance", 1): "dis",
    ("spotcheck", 0): "spotais",
    ("spotcheck", 1): "spotdis",
}

_KEYS_POOLED = ["model", "sampler", "estimator", "checkpoint"]
_KEYS_CLASS = ["model", "class_key", "sampler", "estimator", "checkpoint"]


def _pit(x: float, pool: np.ndarray) -> float:
    """Mid-rank PIT of ``x`` within ``pool``: uniform on (0,1) under exchangeability."""
    M = pool.size
    rank = float((pool < x).sum()) + 0.5 * float((pool == x).sum())
    return (rank + 0.5) / (M + 1.0)


# ---------------------------------------------------------------------------
# Per-cell processing: npz -> atom rows / path-stability rows / rmse rows / band rows
# ---------------------------------------------------------------------------


def process_cell(npz_path: Path, args: argparse.Namespace
                 ) -> Dict[str, List[dict]]:
    z = np.load(npz_path, allow_pickle=True)
    N, w = int(z["meta__N"]), int(z["meta__w"])
    F_true = float(z["meta__F_true"])
    cls, model = str(z["meta__class_key"]), str(z["meta__model"])
    out: Dict[str, List[dict]] = {"base": [], "stab": [], "rmse": [], "band": []}

    groups: Dict[Tuple[str, int], List[Tuple[str, str]]] = {}
    for key in (k for k in z.files if k.startswith("f__")):
        _, sampler, frz, est = key.split("__", 3)
        groups.setdefault((sampler, int(frz)), []).append((est, key))

    for (sampler, frz), ests in sorted(groups.items()):
        rows = z[f"rows__{sampler}__{frz}"]  # (n, 3): seed, base_seed, source
        base_seeds, source = rows[:, 1], rows[:, 2]
        bases = sorted({int(b) for b in base_seeds if b >= 0})
        if len(bases) < 2:
            continue  # zero-shot store: one frozen g, replay == real by construction
        cell = SAMPLER_LABEL.get((sampler, frz), f"{sampler}__{frz}")
        sim_mask = {b: (source == 0) & (base_seeds == b) for b in bases}
        real_row = {b: int(np.flatnonzero((source == 1) & (base_seeds == b))[0])
                    for b in bases
                    if ((source == 1) & (base_seeds == b)).any()}

        for est, key in sorted(ests):
            fhat = z[key]
            n, T = fhat.shape
            t = np.arange(1, T + 1)
            frac = (w + t) / N
            if est in PASSTHROUGH:
                F = fhat.astype(np.float64)
            else:
                F, _ = combine_matrix(fhat, N, w,
                                      _sqrt_index(args.weighting, bool(frz), w))
            FE = (F - F_true) / F_true

            # checkpoint columns (+ the whole-trajectory AUC pseudo-checkpoint)
            ti = np.minimum(np.searchsorted(frac, args.checkpoints, side="left"), T - 1)
            columns = [(f"{cp:.2f}", float(frac[i]), int(t[i]), F[:, i])
                       for cp, i in zip(args.checkpoints, ti)]
            amask = frac >= args.auc_start
            if amask.any():
                columns.append(("auc", float("nan"), -1, FE[:, amask].mean(axis=1)))

            for label, frac_at, t_at, vals in columns:
                # --- replay path stability: replay-only (all multi-base cells) ---
                m_r, v_r, M_b = [], [], []
                for b in bases:
                    sv = vals[sim_mask[b]]
                    sv = sv[np.isfinite(sv)]
                    if sv.size >= args.min_sims:
                        m_r.append(float(sv.mean()))
                        v_r.append(float(sv.var(ddof=1)))
                        M_b.append(sv.size)
                if len(m_r) >= 2:
                    sig = np.sqrt(v_r)
                    med_sig = float(np.median(sig))
                    out["stab"].append(dict(
                        model=model, class_key=cls, sampler=cell, estimator=est,
                        checkpoint=label, frac_labeled=frac_at,
                        n_bases=len(m_r), M_min=int(min(M_b)),
                        range_m=float(max(m_r) - min(m_r)), med_sigma=med_sig,
                        range_m_sigma=(float(max(m_r) - min(m_r)) / med_sig
                                       if med_sig > 0 else float("nan")),
                        vratio_maxmin=(float(max(v_r) / min(v_r))
                                       if min(v_r) > 0 else float("nan")),
                    ))

                # --- the atom: real runs inside their own replay clouds ---
                if not real_row:
                    continue
                for b, rr in real_row.items():
                    X = float(vals[rr])
                    sims = vals[sim_mask[b]]
                    sims = sims[np.isfinite(sims)]
                    if not np.isfinite(X) or sims.size < args.min_sims:
                        continue
                    mu, sigma = float(sims.mean()), float(sims.std(ddof=1))
                    ho = vals[(source == 0) & (base_seeds != b)]
                    ho = ho[np.isfinite(ho)]
                    out["base"].append(dict(
                        model=model, class_key=cls, sampler=cell, estimator=est,
                        checkpoint=label, frac_labeled=frac_at, t=t_at,
                        base_seed=int(b), M=int(sims.size),
                        X_real=X, mu_replay=mu, sigma_replay=sigma,
                        u=((X - mu) / sigma if sigma > 0 else float("nan")),
                        pit=_pit(X, sims),
                        heldout_pit=(_pit(X, ho) if ho.size else float("nan")),
                        heldout_M=int(ho.size),
                    ))

                # --- Test E inputs: frac-error at this checkpoint, real vs sims ---
                if label != "auc" and len(real_row) >= 2:
                    fe = (vals - F_true) / F_true
                    fe_real = np.array([fe[rr] for rr in real_row.values()])
                    fe_sims = fe[source == 0]
                    fe_sims = fe_sims[np.isfinite(fe_sims)]
                    if np.isfinite(fe_real).all() and fe_sims.size:
                        rr_ = float(np.sqrt((fe_real ** 2).mean()))
                        rs_ = float(np.sqrt((fe_sims ** 2).mean()))
                        out["rmse"].append(dict(
                            model=model, class_key=cls, sampler=cell, estimator=est,
                            checkpoint=label, frac_labeled=frac_at,
                            n_real=len(fe_real), n_sims=int(fe_sims.size),
                            rmse_real=rr_, rmse_replay=rs_,
                            rel_gap=(rs_ / rr_ - 1.0) if rr_ > 0 else float("nan"),
                            mean_abs_real=float(np.abs(fe_real).mean()),
                            mean_abs_replay=float(np.abs(fe_sims).mean()),
                        ))

            # --- Test D figure data: per-base replay band + real trajectory ---
            if not args.no_band and real_row:
                for b, rr in real_row.items():
                    S = FE[sim_mask[b]]
                    if S.shape[0] < args.min_sims:
                        continue
                    p05, p50, p95 = np.percentile(S, [5, 50, 95], axis=0)
                    out["band"].extend(
                        dict(model=model, class_key=cls, sampler=cell, estimator=est,
                             base_seed=int(b), t=int(t[i]),
                             frac_labeled=float(frac[i]),
                             fe_p05=float(p05[i]), fe_p50=float(p50[i]),
                             fe_p95=float(p95[i]), fe_real=float(FE[rr, i]))
                        for i in range(T))
    return out


# ---------------------------------------------------------------------------
# Tests: per-class and pooled-over-classes
# ---------------------------------------------------------------------------


def per_class_tests(per_base: pd.DataFrame, alpha: float, min_real: int
                    ) -> pd.DataFrame:
    rows = []
    for keys, g in per_base.groupby(_KEYS_CLASS):
        u = g["u"].to_numpy(float)
        u = u[np.isfinite(u)]
        R = u.size
        row = dict(zip(_KEYS_CLASS, keys))
        row.update(R=R, frac_labeled=float(g["frac_labeled"].iloc[0]))
        if R >= min_real:
            ubar = float(u.mean())
            d = float(u.var(ddof=1))
            lo = stats.chi2.ppf(alpha / 2, R - 1) / (R - 1)
            hi = stats.chi2.ppf(1 - alpha / 2, R - 1) / (R - 1)
            gap = g["X_real"].to_numpy(float) - g["mu_replay"].to_numpy(float)
            mu_mean = float(g["mu_replay"].mean())
            row.update(
                u_mean=ubar, z=float(np.sqrt(R) * ubar), d=d,
                d_band_lo=float(lo), d_band_hi=float(hi),
                gap_sigma=ubar,
                gap_pct_of_mean=(100.0 * float(np.nanmean(gap)) / mu_mean
                                 if mu_mean != 0 else float("nan")),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_tests(per_class: pd.DataFrame, per_base: pd.DataFrame,
                 args: argparse.Namespace) -> pd.DataFrame:
    zq = stats.norm.ppf(0.95)  # one-sided 5% per TOST side -> 90% CI
    rows = []
    pb_groups = dict(tuple(per_base.groupby(_KEYS_POOLED)))
    for keys, g in per_class.groupby(_KEYS_POOLED):
        g = g[np.isfinite(g.get("z", np.nan))]
        row = dict(zip(_KEYS_POOLED, keys))
        C = len(g)
        row.update(C=C)
        if C < 1:
            rows.append(row)
            continue
        z_pool = float(g["z"].sum() / np.sqrt(C))
        df = int((g["R"] - 1).sum())
        d_pool = float(((g["R"] - 1) * g["d"]).sum() / df)
        lo = stats.chi2.ppf(args.alpha / 2, df) / df
        hi = stats.chi2.ppf(1 - args.alpha / 2, df) / df

        pb = pb_groups.get(keys)
        u = pb["u"].to_numpy(float)
        u = u[np.isfinite(u)]
        n_u = u.size
        u_mean, u_sd = float(u.mean()), float(u.std(ddof=1))
        ci_lo = u_mean - zq * u_sd / np.sqrt(n_u)
        ci_hi = u_mean + zq * u_sd / np.sqrt(n_u)

        pit = pb["pit"].to_numpy(float)
        pit = pit[np.isfinite(pit)]
        obs, _ = np.histogram(pit, bins=5, range=(0.0, 1.0))
        gof_p = float(stats.chisquare(obs).pvalue) if pit.size >= 10 else float("nan")

        ho = pb["heldout_pit"].to_numpy(float)
        ho = ho[np.isfinite(ho)]
        ho_in90 = float(((ho >= 0.05) & (ho <= 0.95)).mean()) if ho.size else float("nan")

        ok_z = abs(z_pool) <= args.pass_z
        ok_d = lo <= d_pool <= hi
        ok_g = (not np.isfinite(gof_p)) or gof_p > args.pass_gof
        eq = (ci_lo >= -args.tost_margin) and (ci_hi <= args.tost_margin)
        verdict = ("PASS" if (ok_z and ok_d and ok_g)
                   else "BOUNDED_BIAS" if (eq and ok_d) else "FAIL")

        row.update(
            n_u=n_u, z_pool=z_pool, u_mean=u_mean,
            u_ci90_lo=float(ci_lo), u_ci90_hi=float(ci_hi), tost_pass=eq,
            d_pool=d_pool, d_df=df, d_band_lo=float(lo), d_band_hi=float(hi),
            gof_chi2_p=gof_p, heldout_in90=ho_in90,
            pass_z=ok_z, pass_d=ok_d, pass_gof=ok_g, verdict=verdict,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def shift_table(per_base: pd.DataFrame, test_fracs: list[str]) -> pd.DataFrame:
    """The paper's *Shift* column: mean landing position, in replay SDs.

    Each run's checkpoints are averaged first, leaving one number per genuine
    run; those are averaged within a task, and then across tasks.  Faithful
    replay predicts 0.

    ``se`` clusters at the task, not the run, because the five seeds of a task
    can share a task-specific departure from faithful replay; ``se_runs`` keeps
    the unclustered SD/sqrt(n_runs) for comparison, which understates the error.
    """
    d = per_base[per_base["checkpoint"].isin(test_fracs)]
    rows = []
    for (model, sampler, est), g in d.groupby(["model", "sampler", "estimator"]):
        piv = g.pivot_table(index=["class_key", "base_seed"],
                            columns="checkpoint", values="u")
        piv = piv.reindex(columns=test_fracs).dropna()
        if piv.shape[0] < 2:
            continue
        per_run = piv.mean(axis=1)                       # one value per run
        per_task = per_run.groupby(level="class_key").mean()
        if per_task.size < 2:
            continue
        se = float(per_task.std(ddof=1) / np.sqrt(per_task.size))
        se_runs = float(per_run.std(ddof=1) / np.sqrt(per_run.size))
        shift = float(per_task.mean())
        rows.append(dict(
            model=model, sampler=sampler, estimator=est,
            n_runs=int(per_run.size), n_tasks=int(per_task.size),
            shift=shift, se=se, se_runs=se_runs,
            t=shift / se if se > 0 else np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fhat-in", default="outputs/analysis/perstep_Fhat",
                    help="dir of per-cell F_hat matrix npz (collect_sims --fhat-out)")
    ap.add_argument("--out-dir", default="outputs/analysis/replay_validation")
    ap.add_argument("--checkpoints", nargs="+", type=float,
                    default=[0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--test-fracs", default="0.1,0.2,0.3,0.4,0.5",
                    help="checkpoints to headline in the console verdict (the "
                         "full CSVs carry every checkpoint)")
    ap.add_argument("--weighting", choices=["auto", "lure", "comb"], default="auto",
                    help="combining weights, forwarded to derive_estimates logic")
    ap.add_argument("--auc-start", type=float, default=0.1,
                    help="signed-error AUC integrates frac_labeled >= this")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="two-sided level for the chi2 dispersion bands")
    ap.add_argument("--pass-z", type=float, default=1.96)
    ap.add_argument("--pass-gof", type=float, default=0.05)
    ap.add_argument("--tost-margin", type=float, default=0.5,
                    help="equivalence margin for mean(u), in replay-SD units")
    ap.add_argument("--min-sims", type=int, default=50,
                    help="min sims a base needs before its cloud is trusted")
    ap.add_argument("--min-real", type=int, default=2,
                    help="min real runs a class needs to enter the tests")
    ap.add_argument("--classes", default="", help="substring filter on npz stem")
    ap.add_argument("--models", default="", help="substring filter on npz stem")
    ap.add_argument("--no-band", action="store_true",
                    help="skip the Test D band parquet (figure data)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    fhat_in = Path(args.fhat_in)
    files = [f for f in sorted(fhat_in.glob("*.npz"))
             if args.classes in f.stem and args.models in f.stem]
    if not files:
        raise SystemExit(f"no npz in {fhat_in} matching filters")
    test_fracs = [f"{float(x):.2f}" for x in args.test_fracs.split(",") if x.strip()]

    print(f"Replay validation over {len(files)} cells | weighting={args.weighting} | "
          f"checkpoints={args.checkpoints} + auc(>= {args.auc_start:.0%})")
    acc: Dict[str, List[dict]] = {"base": [], "stab": [], "rmse": [], "band": []}
    for f in files:
        got = process_cell(f, args)
        n_atoms = len(got["base"])
        for k in acc:
            acc[k].extend(got[k])
        tag = "atoms" if n_atoms else ("path-stab only" if got["stab"] else "skipped")
        print(f"  [{tag:>14}] {f.stem}: {n_atoms} atom rows")

    if not acc["base"]:
        raise SystemExit("no real-run atoms found (no multi-base cell has real runs)")
    per_base = pd.DataFrame(acc["base"])
    per_class = per_class_tests(per_base, args.alpha, args.min_real)
    pooled = pooled_tests(per_class, per_base, args)
    stab = pd.DataFrame(acc["stab"])
    rmse = pd.DataFrame(acc["rmse"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_base.to_csv(out_dir / "per_base.csv", index=False)
    per_class.to_csv(out_dir / "per_class.csv", index=False)
    pooled.to_csv(out_dir / "pooled.csv", index=False)
    stab.to_csv(out_dir / "path_stability.csv", index=False)
    rmse.to_csv(out_dir / "rmse_gap.csv", index=False)
    shift = shift_table(per_base, test_fracs)
    shift.to_csv(out_dir / "shift.csv", index=False)
    wrote = ["per_base.csv", "per_class.csv", "pooled.csv",
             "path_stability.csv", "rmse_gap.csv", "shift.csv"]
    if not args.no_band and acc["band"]:
        pd.DataFrame(acc["band"]).to_parquet(out_dir / "band_data.parquet",
                                             index=False)
        wrote.append("band_data.parquet")

    # ---- console verdict (headline checkpoints + auc only) --------------------
    head = pooled[pooled["checkpoint"].isin(test_fracs + ["auc"])]
    idx = ["model", "sampler", "estimator"]
    with pd.option_context("display.max_rows", None, "display.width", 220,
                           "display.float_format", lambda v: f"{v:.2f}"):
        for col, label in [("z_pool", "pooled bias z (|z| <= %.2f passes)" % args.pass_z),
                           ("d_pool", "pooled dispersion d (chi2 band per row df)")]:
            piv = head.pivot_table(index=idx, columns="checkpoint", values=col)
            print(f"\n=== {label} ===")
            print(piv.to_string())
        if len(head):
            df0 = int(head["d_df"].dropna().median())
            lo = stats.chi2.ppf(args.alpha / 2, df0) / df0
            hi = stats.chi2.ppf(1 - args.alpha / 2, df0) / df0
            print(f"\ntypical d band at df={df0}: [{lo:.2f}, {hi:.2f}]  "
                  f"(resolution floor ~ +/-{100 * (hi - 1):.0f}% on variance)")
        for cp in test_fracs + ["auc"]:
            sub = head[head["checkpoint"] == cp]
            if len(sub):
                print(f"verdict @ {cp}: {sub['verdict'].value_counts().to_dict()}")
        ho = head["heldout_in90"].dropna()
        if len(ho):
            print(f"held-out: mean fraction of real runs inside the central 90% of "
                  f"the other bases' pooled sims = {ho.mean():.2f} (expect ~0.90)")
        if len(shift):
            print("\n=== Shift: mean landing position, checkpoints averaged "
                  "within a run first (replay SDs) ===\n"
                  "faithful replay predicts 0.  'se' clusters at the task "
                  "(t_{n_tasks-1} reference); 'se_runs' is the unclustered "
                  "SD/sqrt(n_runs), which understates it:")
            print(shift.set_index(["model", "sampler", "estimator"])[
                ["n_runs", "n_tasks", "shift", "se", "se_runs", "t"]].to_string(
                    float_format=lambda v: f"{v:+.3f}"))
        if len(stab):
            sub = stab[stab["checkpoint"].isin(test_fracs)]
            agg = (sub.groupby(["model", "sampler"])["vratio_maxmin"]
                   .agg(median="median", worst="max"))
            print("\nReplay path stability: max/min replay-variance ratio per (model, sampler)\n"
                  "(single-setting null at ~199 sims: ~1.26 typical, ~1.5 at 95%; "
                  "'worst' is a max over classes x estimators x checkpoints, so it "
                  "runs higher than 1.5 even under the null -- read the median):")
            print(agg.to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"\nwrote {', '.join(wrote)} in {out_dir}")


if __name__ == "__main__":
    main()
