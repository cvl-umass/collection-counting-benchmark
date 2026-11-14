"""Pilot proxy table: how reliably does each proxy, computed on a small labeled pilot,
choose the frozen model that DIS will estimate best with?

    python3 analysis/pilot_proxy_table.py            # -> outputs/analysis/pilot_proxy_table.csv

For every task, draw `--draws` uniform pilots of `--pilot` units, score all frozen models
on each pilot with six proxies, and compare with the models' realized DIS error at 10%
and 30% labeled, on two candidate sets (all frozen models; the MLLMs alone).

Proxies (lower = better; p = f/sum(f) is the normalized true count on the pilot):
    hellinger_floor  H(p, q),  q ~ max(g, 1): the proposal DIS samples from (paper default)
    hellinger_raw    H(p, q_raw),  q_raw ~ g: the same distance without the floor
    rmse, mae        per-unit count error
    gap              |sum(g) - sum(f)| / sum(f)
    chi2             chi-square divergence chi2(p || q) with the floored q

Metrics (ideal value 1 for all four):
    rho    within-task Spearman between the proxy, averaged over pilots, and realized
           error, averaged over tasks
    beats  fraction of the other candidates with higher error than the pick (mean over pilots)
    win    fraction of pilots whose pick has the lowest error (a tie for the lowest counts)
    ratio  pick's error / lowest error (median over pilots)

This is a vectorized equivalent of analysis/pilot_model_selection.py, which stays the
readable reference for one candidate set and budget.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_model_selection import (OVD, ROOT, TASKS, downstream_err, frozen_models,  # noqa: E402
                                   load_pool, nanmean, within_task_corr)

OUT = os.path.join(ROOT, "outputs", "analysis", "pilot_proxy_table.csv")
PROXIES = ["hellinger_floor", "hellinger_raw", "rmse", "mae", "gap", "chi2"]
FLOOR = 1.0   # per-unit sampling floor of the DIS proposal (epsilon in the paper)


def pilot_proxies(f, G, idx):
    """Proxy values for one task.  f: (N,) true counts; G: (M, N) predictions of the M
    models; idx: (D, K) unit indices of D pilots.  Returns {proxy: (D, M) array}, with nan
    where a proxy is undefined (a pilot with no objects, or all-zero predictions for
    hellinger_raw)."""
    F = f[idx]                                   # (D, K)
    Fs = F.sum(1)
    empty = Fs <= 0
    out = {k: np.full((len(idx), len(G)), np.nan) for k in PROXIES}
    with np.errstate(divide="ignore", invalid="ignore"):
        p = F / Fs[:, None]
        for j, g in enumerate(G):
            Gp = g[idx]
            Gs = Gp.sum(1)
            q = np.maximum(Gp, FLOOR)
            q = q / q.sum(1, keepdims=True)
            hf = np.sqrt(np.maximum(0.0, 1.0 - np.sqrt(p * q).sum(1)))
            hr = np.sqrt(np.maximum(0.0, 1.0 - np.sqrt(p * Gp / Gs[:, None]).sum(1)))
            chi2 = (p ** 2 / q).sum(1) - 1.0
            gap = np.abs(Gs - Fs) / Fs
            hf[empty], chi2[empty], gap[empty] = np.nan, np.nan, np.nan
            hr[empty | (Gs <= 0)] = np.nan
            out["hellinger_floor"][:, j], out["hellinger_raw"][:, j] = hf, hr
            out["chi2"][:, j], out["gap"][:, j] = chi2, gap
            out["rmse"][:, j] = np.sqrt(((Gp - F) ** 2).mean(1))
            out["mae"][:, j] = np.abs(Gp - F).mean(1)
    return out


def evaluate(prox, models, cols, err):
    """The four metrics per proxy, for the candidate models at column indices `cols`."""
    names = [models[j] for j in cols]
    res = {}
    for k in PROXIES:
        table = pd.DataFrame([dict(task=t, err=err[(t, m)], v=nanmean(prox[t][k][:, j]))
                              for t in TASKS for j, m in zip(cols, names)])
        beats, win, ratio = [], [], []
        for t in TASKS:
            e = np.array([err[(t, m)] for m in names])
            V = prox[t][k][:, cols]
            V = V[(~np.isnan(V)).sum(1) >= 2]            # need two scored candidates to select
            pick_err = e[np.argmin(np.where(np.isnan(V), np.inf, V), axis=1)]
            beats.append((pick_err[:, None] < e[None, :]).sum(1) / (len(names) - 1))
            win.append(pick_err <= e.min())
            ratio.append(pick_err / e.min())
        beats, win, ratio = map(np.concatenate, (beats, win, ratio))
        res[k] = dict(rho=within_task_corr(table, "v")[0], beats=beats.mean(), win=win.mean(),
                      ratio=np.median(ratio), ratio_mean=ratio.mean(), n_pilots=len(ratio))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=50, help="pilot size K (uniform)")
    ap.add_argument("--draws", type=int, default=10000, help="independent pilots per task")
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.10, 0.30])
    ap.add_argument("--reference", choices=["dis", "best"], default="dis",
                    help="realized error: DIS (default, as in the paper) or each model's best valid estimator")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    models = frozen_models()
    rng = np.random.default_rng(args.seed)
    prox = {}
    for t in TASKS:
        pools = [load_pool(t, m) for m in models]
        f = pools[0][0]
        k = min(args.pilot, len(f))
        idx = np.stack([rng.choice(len(f), size=k, replace=False) for _ in range(args.draws)])
        prox[t] = pilot_proxies(f, [g for _, g in pools], idx)   # each pilot is scored by all models

    rows = []
    for pool in ["all", "mllm"]:
        cols = [j for j, m in enumerate(models) if pool == "all" or m not in OVD]
        for b in args.budgets:
            err = downstream_err(b, args.reference)
            for k, r in evaluate(prox, models, cols, err).items():
                rows.append(dict(pool=pool, n_models=len(cols), budget=b, proxy=k, **r))
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)

    ref = "DIS" if args.reference == "dis" else "best valid estimator"
    print(f"{len(TASKS)} tasks, {args.draws} pilots of {args.pilot} units per task; realized error = {ref}\n")
    for (pool, n, b), grp in d.groupby(["pool", "n_models", "budget"], sort=False):
        print(f"{'MLLM + OVD' if pool == 'all' else 'MLLM'} (n={n}), {b:.0%} labeled")
        print(grp.set_index("proxy")[["rho", "beats", "win", "ratio"]].round(2).to_string(), "\n")
    print(f"[wrote] {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
