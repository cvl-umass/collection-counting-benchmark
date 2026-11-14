"""Pilot-set model selection: can a small labeled pilot identify the frozen model
that will estimate best at full budget?

Prints a per-proxy comparison for one candidate set and one budget.  The paper's
pilot table (two candidate sets x two budgets, 10,000 pilots) comes from the faster
analysis/pilot_proxy_table.py, which reuses the helpers defined here.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "outputs", "runs")
SUMMARY = os.path.join(ROOT, "outputs", "analysis", "estimates_summary.csv")
OUT = os.path.join(ROOT, "outputs", "analysis", "pilot_model_selection.csv")

TASKS = [
    "birds/bird", "dota/harbor", "dota/helicopter", "dota/plane", "dota/ship",
    "dota/small_vehicle", "dota/storage_tank", "dota/tennis_court",
    "kenyan_wildlife/giraffe", "malaria/difficult", "malaria/red_blood_cell",
    "malaria/trophozoite",
]

OVD = {"florence2", "grounding_dino", "owlv2"}                      # open-vocab detectors
PROPRIETARY = {"gemini-3.1-pro-preview", "gpt-5.2", "qwen3.7-plus"}  # API MLLMs

# valid (sampler, estimator) configs for a frozen-g model (see analysis/estimator_wins.py)
VALID_CONFIGS = [("random", "model_only_corrected"), ("random", "monte_carlo"),
                 ("random", "difference"), ("random", "ratio"), ("random", "regression"),
                 ("dis", "dis"), ("spotdis", "spotdis")]


def frozen_models():
    """The 11 models that ship a __g_fixed store across every task."""
    names = None
    for t in TASKS:
        here = {os.path.basename(d).replace("__g_fixed", "")
                for d in glob.glob(os.path.join(RUNS, t, "*__g_fixed"))}
        names = here if names is None else (names & here)
    return sorted(names)


def load_pool(task, model):
    """Full-pool (f, g): true per-tile counts and the frozen model prediction."""
    d = os.path.join(RUNS, task, f"{model}__g_fixed")
    # Every run of a task shares one manifest (the pool is frozen before any model
    # runs), so the released data stores one copy per task directory rather than
    # one per run; fall back to it when the per-run copy is absent.
    m = os.path.join(d, "manifest.csv")
    if not os.path.exists(m):
        m = os.path.join(os.path.dirname(d), "manifest.csv")
    f = pd.read_csv(m)["gt_count"].to_numpy(float)
    g = np.load(os.path.join(d, "snapshots.npz"))["g"][0].astype(float)
    return f, g


def hellinger(f, g, floor=0.0):
    """Tile-mass Hellinger between the count-optimal proposal f/sum(f) and g/sum(g).
    0 = identical allocation of sampling mass; 1 = disjoint.  Undefined if either sums to 0.
    floor=1.0 compares against the proposal DIS samples from, max(g, floor)/sum (the
    paper's default proxy H(p, q)); floor=0.0 is the unfloored H(p, q_raw)."""
    if floor > 0:
        g = np.maximum(g, floor)
    if f.sum() <= 0 or g.sum() <= 0:
        return np.nan
    qf, qg = f / f.sum(), g / g.sum()
    return np.sqrt(max(0.0, 1.0 - np.sqrt(qf * qg).sum()))


def chi_square(f, g, floor=1.0):
    """Chi-square divergence chi2(p || q) = sum(p^2 / q) - 1 between the normalized true
    counts p = f/sum(f) and the proposal DIS samples from, q ~ max(g, floor) (the
    per-unit sampling floor, epsilon = 1 in the paper).  On the full pool, F^2 * chi2 is the
    variance of the first DIS draw, so this is the proxy matched to DIS.
    Lower = better; 0 when q equals p.  Undefined if f sums to 0.  The floor keeps it finite:
    without it, chi2 is infinite whenever a unit with objects has g = 0."""
    if f.sum() <= 0:
        return np.nan
    p = f / f.sum()
    q = np.maximum(g, floor)
    q = q / q.sum()
    return float((p ** 2 / q).sum() - 1.0)


def nanmean(x):
    """Mean ignoring nan; nan (not a warning) when every entry is nan."""
    x = np.asarray(x, float)
    return np.nan if np.all(np.isnan(x)) else np.nanmean(x)


def spearman(f, g):
    """Rank correlation; nan when either side is constant (no ranking signal on the pilot)."""
    if np.ptp(f) == 0 or np.ptp(g) == 0:
        return np.nan
    return spearmanr(f, g).statistic


def count_gap(f, g):
    """|G - F| / F on the pilot: raw TOTAL-count agreement.  Lower = better."""
    return np.nan if f.sum() <= 0 else abs(g.sum() - f.sum()) / f.sum()


def rmse(f, g):
    """Root mean squared PER-TILE count error on the pilot (L2, penalizes large errors).
    Lower = better."""
    return float(np.sqrt(np.mean((g - f) ** 2)))


def mae(f, g):
    """Mean absolute PER-TILE count error on the pilot (L1, no extra tail penalty).
    Lower = better."""
    return float(np.mean(np.abs(g - f)))


def downstream_err(budget, reference="dis"):
    """True efficiency per (task, model), mean abs fractional error at `budget`.
    reference='dis'  -> the dis/dis directed estimate (the strategy the proxy is about).
    reference='best' -> the lowest error over all 7 valid configs (best achievable for
                        that model, removing the estimator-choice confound)."""
    df = pd.read_csv(SUMMARY)
    df.columns = [c.lstrip("﻿") for c in df.columns]
    df = df[np.isclose(df.checkpoint, budget)]
    if reference == "dis":
        df = df[(df.sampler == "dis") & (df.estimator == "dis")]
        return {(r.class_key, r.model): r.mean_abs_frac_error for r in df.itertuples()}
    df = df[df[["sampler", "estimator"]].apply(tuple, axis=1).isin(VALID_CONFIGS)]
    best = df.groupby(["class_key", "model"]).mean_abs_frac_error.min()
    return best.to_dict()


def within_task_corr(table, proxy, target="err"):
    """Mean over tasks of Spearman(proxy, target) across models -- the honest 'which
    model, for THIS task' comparison (avoids pooling across task difficulty)."""
    rhos = []
    for _, grp in table.groupby("task"):
        v, e = grp[proxy].to_numpy(), grp[target].to_numpy()
        ok = ~(np.isnan(v) | np.isnan(e))
        if ok.sum() >= 4 and np.ptp(v[ok]) > 0:
            rhos.append(spearmanr(v[ok], e[ok]).statistic)
    return float(np.nanmean(rhos)), len(rhos)


def selection(err_by_model, proxy_by_model, higher_better):
    """One pilot draw, one task: pick the best-proxy model, score it against the rest.
    Returns (fraction of other models the pick beats, whether the pick has the lowest
    error (ties count), pick_err / oracle_err)."""
    cands = [m for m in err_by_model if not np.isnan(proxy_by_model.get(m, np.nan))]
    if len(cands) < 2:
        return np.nan, np.nan, np.nan
    pick = max(cands, key=lambda m: proxy_by_model[m] if higher_better else -proxy_by_model[m])
    others = [m for m in err_by_model if m != pick]
    beats = np.mean([err_by_model[pick] < err_by_model[o] for o in others])
    oracle = min(err_by_model.values())
    return beats, float(err_by_model[pick] <= oracle), (err_by_model[pick] / oracle if oracle > 0 else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=50, help="pilot size K (uniform)")
    ap.add_argument("--draws", type=int, default=300, help="independent pilot draws")
    ap.add_argument("--budget", type=float, default=0.30, help="downstream dis/dis checkpoint")
    ap.add_argument("--pool", choices=["all", "mllm", "open"], default="all",
                    help="all=11 frozen; mllm=drop the 3 open-vocab detectors (8); open=open MLLMs only (5)")
    ap.add_argument("--reference", choices=["dis", "best"], default="dis",
                    help="downstream target: dis/dis (default) or each cell's best valid estimator")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    models = frozen_models()
    if args.pool in ("mllm", "open"):
        models = [m for m in models if m not in OVD]
    if args.pool == "open":
        models = [m for m in models if m not in PROPRIETARY]
    suffix = ("" if args.pool == "all" else f"_{args.pool}") + ("" if args.reference == "dis" else "_best")
    out = OUT.replace(".csv", suffix + ".csv")
    err = downstream_err(args.budget, args.reference)
    pools = {(t, m): load_pool(t, m) for t in TASKS for m in models}
    rng = np.random.default_rng(args.seed)

    # ---- per (task, model): full-pool oracle proxies + per-draw pilot proxies ----
    rows = []
    # store per-draw pilot proxies for the selection test, keyed (task, model)
    draw_spear = {k: [] for k in pools}
    draw_hell = {k: [] for k in pools}
    draw_hell_floor = {k: [] for k in pools}
    draw_chi2 = {k: [] for k in pools}
    draw_gap = {k: [] for k in pools}
    draw_rmse = {k: [] for k in pools}
    draw_mae = {k: [] for k in pools}
    draw_pos = {t: [] for t in TASKS}          # pilot positives, per task (shared idx)

    for t in TASKS:
        N = len(pools[(t, models[0])][0])
        k = min(args.pilot, N)
        for _ in range(args.draws):
            idx = rng.choice(N, size=k, replace=False)   # ONE pilot, scored by all models
            f0 = pools[(t, models[0])][0][idx]
            draw_pos[t].append(int((f0 > 0).sum()))
            for m in models:
                f, g = pools[(t, m)]
                fp, gp = f[idx], g[idx]
                draw_spear[(t, m)].append(spearman(fp, gp))
                draw_hell[(t, m)].append(hellinger(fp, gp))
                draw_hell_floor[(t, m)].append(hellinger(fp, gp, floor=1.0))
                draw_chi2[(t, m)].append(chi_square(fp, gp))
                draw_gap[(t, m)].append(count_gap(fp, gp))
                draw_rmse[(t, m)].append(rmse(fp, gp))
                draw_mae[(t, m)].append(mae(fp, gp))

    for t in TASKS:
        for m in models:
            f, g = pools[(t, m)]
            rows.append(dict(
                task=t, model=m,
                err=err.get((t, m), np.nan),
                pilot_positives=float(np.median(draw_pos[t])),
                # full-pool oracle (needs all of f -- the thing we are trying to avoid)
                oracle_spearman=spearman(f, g),
                oracle_hellinger=hellinger(f, g),
                oracle_hellinger_floor=hellinger(f, g, floor=1.0),
                oracle_chi2=chi_square(f, g),
                oracle_gap=count_gap(f, g),
                oracle_rmse=rmse(f, g),
                oracle_mae=mae(f, g),
                # pilot proxies, averaged over draws (mean of a K-label estimate)
                pilot_spearman=nanmean(draw_spear[(t, m)]),
                pilot_hellinger=nanmean(draw_hell[(t, m)]),
                pilot_hellinger_floor=nanmean(draw_hell_floor[(t, m)]),
                pilot_chi2=nanmean(draw_chi2[(t, m)]),
                pilot_gap=nanmean(draw_gap[(t, m)]),
                pilot_rmse=nanmean(draw_rmse[(t, m)]),
                pilot_mae=nanmean(draw_mae[(t, m)]),
            ))
    table = pd.DataFrame(rows)
    table.to_csv(out, index=False)

    # ---- Claim 1: pilot proxy recovers the oracle's predictive power ----
    def show(tag, proxy):
        r, n = within_task_corr(table, proxy)
        print(f"    {tag:23s} rho(proxy, downstream err) = {r:+.3f}   (avg over {n} tasks)")

    ref_lab = "dis/dis" if args.reference == "dis" else "best-of-7"
    print(f"\n[setup] pool={args.pool} ({len(models)} models) x {len(TASKS)} tasks | "
          f"pilot K={args.pilot}, {args.draws} draws | downstream = {ref_lab} @ {args.budget:.0%}\n")
    def block(prefix):
        show("Hellinger  DIS proposal", f"{prefix}_hellinger_floor")
        show("Hellinger  rank/shape", f"{prefix}_hellinger")
        show("chi-square DIS draw", f"{prefix}_chi2")
        show("count gap  total-cnt", f"{prefix}_gap")
        show("RMSE       per-tile L2", f"{prefix}_rmse")
        show("MAE        per-tile L1", f"{prefix}_mae")
        show("Spearman   rank(ref)", f"{prefix}_spearman")

    print("CLAIM 1  cheap: within-task rank corr of each proxy with true downstream efficiency")
    print("  higher |rho| = stronger predictor. Sign: better ranking/agreement -> lower err.\n")
    print("  ORACLE (full pool f, not deployable):")
    block("oracle")
    print(f"\n  PILOT (K={args.pilot} labels only, deployable):")
    block("pilot")

    # ---- Claim 2: ranking proxy selects a better model than the count proxy ----
    # single-draw selection reliability, averaged over draws (a real pilot is one draw)
    def sel_over_draws(proxy_draws, higher_better):
        beats, wins, ratio = [], [], []
        for t in TASKS:
            e = {m: err[(t, m)] for m in models}
            for i in range(args.draws):
                p = {m: proxy_draws[(t, m)][i] for m in models}
                b, w, r = selection(e, p, higher_better)
                if not np.isnan(b):
                    beats.append(b); wins.append(w); ratio.append(r)
        return np.mean(beats), np.mean(wins), np.median(ratio)

    print("\nCLAIM 2  ranking > count: pick the best-proxy model on one pilot, score vs the rest")
    print("  beats = mean fraction of non-selected models the pick beats (0.5 = chance);")
    print("  win   = fraction of pilots whose pick has the lowest error (ties count);")
    print("  ratio = median of pick_err / oracle_err (1.0 = perfect).\n")
    for tag, draws, hb in [
        ("Hellinger  DIS proposal", draw_hell_floor, False),
        ("Hellinger  rank/shape", draw_hell, False),
        ("chi-square DIS draw", draw_chi2, False),
        ("count gap  total-cnt", draw_gap, False),
        ("RMSE       per-tile L2", draw_rmse, False),
        ("MAE        per-tile L1", draw_mae, False),
        ("Spearman   rank(ref)", draw_spear, True),
    ]:
        b, w, r = sel_over_draws(draws, hb)
        print(f"    {tag:23s} beats {b:.0%} of others | wins {w:.0%} | {r:.2f}x oracle")
    print(f"\n[wrote] {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
