"""Collect the 1,000-trajectory ensembles (genuine runs + fixed-model replays).

Writes the per-step F_hat matrices that the estimate and validation stages read.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Six sklearn fits per h refit x N pool workers would oversubscribe BLAS.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.run_estimators import estimate_all  # noqa: E402
from analysis.simulate import (  # noqa: E402
    simulate_trajectory,
    simulate_zeroshot,
    zeroshot_features,
)
from analysis.trajectory import load_trajectory  # noqa: E402

DROP = ["run_dir"]  # long strings; provenance lives in (class, model, sampler, seeds)


def checkpoint_rows(df: pd.DataFrame, checkpoints: list[float]) -> pd.DataFrame:
    """Keep the first step at or past each checkpoint fraction labeled."""
    steps = df[["t", "frac_labeled"]].drop_duplicates().sort_values("t")
    idx = np.searchsorted(steps["frac_labeled"].to_numpy(), checkpoints, side="left")
    idx = np.minimum(idx, len(steps) - 1)
    t_of = dict(zip(checkpoints, steps["t"].to_numpy()[idx]))
    parts = []
    for cp, t_cp in t_of.items():
        rows = df[df["t"] == t_cp].copy()
        rows["checkpoint"] = cp
        parts.append(rows)
    return pd.concat(parts, ignore_index=True).drop(columns=DROP)


# The per-step reducer groups on these; the rest of estimate_all's columns
# (run_dir, seed, source, base_seed) are per-sim provenance we aggregate away.
_PERSTEP_KEYS = ["model", "class_key", "sampler", "frozen", "estimator", "t"]
_PERSTEP_COLS = _PERSTEP_KEYS + ["frac_labeled", "F_hat", "variance", "frac_error", "F_true"]
_PERSTEP_ACC = ["n", "s_fl", "s_F", "s_FF", "s_fe", "s_ffe", "s_afe", "s_var", "n_var"]


def perstep_partial(full: pd.DataFrame) -> pd.DataFrame:
    """Additive per-(sampler, estimator, t) accumulator over one ensemble chunk.

    Reduces a task's per-sim per-step frames to running sums, so partials from
    different bases/chunks of the SAME cell just add (see finalize_perstep).
    Keeping sum and sum-of-squares of F_hat makes the empirical variance of the
    estimator across the simulated runs recoverable exactly at every step.
    """
    d = full.assign(
        _FF=full["F_hat"] ** 2,
        _ffe=full["frac_error"] ** 2,
        _afe=full["frac_error"].abs(),
        _vn=full["variance"].notna().astype("int64"),
    )
    return (
        d.groupby(_PERSTEP_KEYS, dropna=False)
        .agg(n=("F_hat", "size"), s_fl=("frac_labeled", "sum"),
             s_F=("F_hat", "sum"), s_FF=("_FF", "sum"),
             s_fe=("frac_error", "sum"), s_ffe=("_ffe", "sum"), s_afe=("_afe", "sum"),
             s_var=("variance", "sum"), n_var=("_vn", "sum"),  # s_var skips NaN
             F_true=("F_true", "first"))
        .reset_index()
    )


def finalize_perstep(partials: pd.DataFrame) -> pd.DataFrame:
    """Sum a cell's partials, then turn the running sums into per-step stats:
    ensemble mean, EMPIRICAL variance/sd of the estimator across the simulated
    runs (the sampling variance analytical V-hat approximates), the mean
    analytical variance, and frac-error moments."""
    g = partials.groupby(_PERSTEP_KEYS, dropna=False)
    s = g[_PERSTEP_ACC].sum()
    s["F_true"] = g["F_true"].first()
    s = s.reset_index()

    n = s["n"]
    denom = (n - 1).where(n > 1, np.nan)  # sample variance; NaN for n < 2
    out = s[_PERSTEP_KEYS].copy()
    out["n_traj"] = n.astype("int64")
    out["frac_labeled"] = s["s_fl"] / n
    out["F_hat_mean"] = s["s_F"] / n
    out["F_hat_var_emp"] = ((s["s_FF"] - s["s_F"] ** 2 / n) / denom).clip(lower=0)
    out["F_hat_sd_emp"] = np.sqrt(out["F_hat_var_emp"])
    out["variance_analytic_mean"] = (s["s_var"] / s["n_var"]).where(s["n_var"] > 0, np.nan)
    out["mean_frac_error"] = s["s_fe"] / n
    out["sd_frac_error"] = np.sqrt(((s["s_ffe"] - s["s_fe"] ** 2 / n) / denom).clip(lower=0))
    out["mean_abs_frac_error"] = s["s_afe"] / n
    out["F_true"] = s["F_true"]
    return out


# ---- per-step F_hat matrix cache (single source of truth) -------------------
# One (n_traj x T) matrix per (sampler, frozen, estimator): the reweightable
# per-step primitive fhat_perstep (F_hat_i pre-combine for the AIS family; the
# plain per-step curve for Monte Carlo / model-only).  Everything the reduced
# checkpoint/perstep artifacts held is a function of these plus (N, w, F_true)
# and a chosen weighting -- see analysis/derive_estimates.py.

SOURCE_CODE = {"sim": 0, "real": 1}

# Direct estimators (no LURE/COMB combining): their per-step curve IS the final
# F_hat, and their analytical variance is not re-derivable from a reweighting, so
# it is cached alongside (a ``v__`` array).
DIRECT_VAR = {"monte_carlo"}


def keep_matrix(sampler: str, estimator: str) -> bool:
    """The estimator x sampler pairs worth caching (drops redundant / invalid
    combos): is_uniform and model_only_pure never; Monte Carlo only on the random
    sampler (SRSWOR is biased on a directed draw); model_only only on
    random/importance; every AIS-family estimator wherever it is emitted."""
    if estimator in ("is_uniform", "model_only_pure"):
        return False
    if estimator == "monte_carlo":
        return sampler == "random"
    if estimator == "model_only_corrected":
        return sampler in ("random", "importance")
    return True


def matrix_pieces(est: pd.DataFrame, traj) -> tuple:
    """Pivot one trajectory's estimate frame into per-estimator per-step vectors.

    Returns ((sampler, frozen), row_meta, {estimator: (fhat32, var32|None)},
    (N, w, F_true)).  ``var`` is kept only for Monte Carlo, whose analytical
    variance is not re-derivable from fhat under a reweighting."""
    m = traj.meta
    sampler = m["sampler"]
    frozen = bool(m.get("frozen", False))
    src = m.get("source", "sim")
    bs = m.get("base_seed")
    if bs is None:  # a real run heads its own base block (base_seed == its seed);
        bs = int(m["seed"]) if src == "real" else -1   # zero-shot has no base -> -1
    row = (int(m["seed"]), int(bs), SOURCE_CODE.get(src, 0))
    out = {}
    for name, blk in est.groupby("estimator", sort=False):
        if not keep_matrix(sampler, name):
            continue
        blk = blk.sort_values("t")
        fhat = blk["fhat_perstep"].to_numpy(np.float32)
        var = blk["variance"].to_numpy(np.float32) if name in DIRECT_VAR else None
        out[name] = (fhat, var)
    return (sampler, frozen), row, out, (int(traj.N), int(traj.w), float(traj.F_true))


def write_cell_matrices(records: list, dest: Path, class_key: str, model: str) -> dict:
    """Assemble a cell's per-sim pieces into (n_traj x T) matrices and save one
    compressed npz.  Rows within a (sampler, frozen) group are ordered by
    (base_seed, seed) so row r means the same trajectory across every estimator.
    Returns per-group trajectory counts for the log line."""
    groups: dict[tuple, list] = {}
    N = w = F_true = None
    for gkey, row, mats, meta in records:
        groups.setdefault(gkey, []).append((row, mats))
        N, w, F_true = meta
    arrays: dict[str, np.ndarray] = {}
    counts: dict = {}
    for (sampler, frozen), rows in groups.items():
        rows.sort(key=lambda rm: (rm[0][1], rm[0][0]))  # (base_seed, seed)
        row_meta = np.array([r for r, _ in rows], dtype=np.int32)
        tag = f"{sampler}__{int(frozen)}"
        arrays[f"rows__{tag}"] = row_meta
        # Real runs record one extra over-budget step (frac just past max_frac)
        # that the sims stop short of; truncate the group to the common length so
        # the matrix is rectangular.  Sims always cross max_frac at Tmin, so no
        # reported checkpoint (<= max_frac) is ever dropped.
        Tmin = min(next(iter(mats.values()))[0].shape[0] for _, mats in rows)
        for est_name in rows[0][1]:  # estimator set is constant within a group
            arrays[f"f__{tag}__{est_name}"] = np.stack(
                [mats[est_name][0][:Tmin] for _, mats in rows])  # (n, Tmin) float32
            if est_name in DIRECT_VAR:
                arrays[f"v__{tag}__{est_name}"] = np.stack(
                    [mats[est_name][1][:Tmin] for _, mats in rows])
        counts[f"{sampler}/{'frozen' if frozen else 'adaptive'}"] = len(rows)
    arrays["meta__N"] = np.int64(N)
    arrays["meta__w"] = np.int64(w)
    arrays["meta__F_true"] = np.float64(F_true)
    arrays["meta__class_key"] = np.array(class_key)
    arrays["meta__model"] = np.array(model)
    np.savez_compressed(dest, **arrays)
    return counts


def collect_cell(run_dir: str, freeze: bool, n_sims: int, checkpoints: list[float],
                 want_reduced: bool) -> tuple[list, tuple | None]:
    """One base run's contribution: its real trajectory (adaptive cells only,
    frozen cells have no real runs) plus ``n_sims`` simulated ones.  Returns
    (matrix records, reduced) where reduced is (checkpoint rows, per-step partial)
    only when ``want_reduced`` (else None)."""
    recs: list = []
    ckpt, full = ([], []) if want_reduced else (None, None)

    def add(traj):
        est = estimate_all(traj)
        recs.append(matrix_pieces(est, traj))
        if want_reduced:
            ckpt.append(checkpoint_rows(est, checkpoints))
            full.append(est[_PERSTEP_COLS])

    if not freeze:
        add(load_trajectory(run_dir))
    base_seed = json.loads((Path(run_dir) / "config.json").read_text())["seed"]
    seed0 = 1000 * (base_seed + 1) + (500 if freeze else 0)
    for k in range(n_sims):
        add(simulate_trajectory(run_dir, seed=seed0 + k, freeze_g=freeze))
    reduced = ((pd.concat(ckpt, ignore_index=True),
                perstep_partial(pd.concat(full, ignore_index=True))) if want_reduced else None)
    return recs, reduced


def collect_zeroshot(store_dir: str, sampler: str, seed0: int, n_sims: int,
                     checkpoints: list[float], want_reduced: bool) -> tuple[list, tuple | None]:
    """One chunk of exact zero-shot sims from a prediction store.  Returns
    (matrix records, reduced-or-None), like :func:`collect_cell`."""
    feats = zeroshot_features(store_dir) if sampler == "spotcheck" else None
    recs: list = []
    ckpt, full = ([], []) if want_reduced else (None, None)
    for k in range(n_sims):
        traj = simulate_zeroshot(store_dir, seed0 + k, sampler=sampler, feats=feats)
        est = estimate_all(traj)
        recs.append(matrix_pieces(est, traj))
        if want_reduced:
            ckpt.append(checkpoint_rows(est, checkpoints))
            full.append(est[_PERSTEP_COLS])
    reduced = ((pd.concat(ckpt, ignore_index=True),
                perstep_partial(pd.concat(full, ignore_index=True))) if want_reduced else None)
    return recs, reduced


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="outputs/runs")
    ap.add_argument("--fhat-out", default="outputs/analysis/perstep_Fhat",
                    help="per-(class,model) per-step F_hat matrices npz (source of truth)")
    ap.add_argument("--reduced", action="store_true",
                    help="ALSO write the derivable checkpoint/perstep parquet in-line "
                         "(default: matrices only; derive them later with derive_estimates.py)")
    ap.add_argument("--out", default="outputs/analysis/checkpoints",
                    help="per-(class,model) per-SIM checkpoint parquet (only with --reduced)")
    ap.add_argument("--perstep-out", default="outputs/analysis/perstep",
                    help="per-(class,model) per-STEP reduced-summary parquet (only with --reduced)")
    ap.add_argument("--checkpoints", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--per-base", type=int, default=199,
                    help="adaptive sims per base (5 real + 5*per_base = cell total)")
    ap.add_argument("--frozen-per-base", type=int, default=200,
                    help="frozen-g sims per base for the DIS/spotDIS cells")
    ap.add_argument("--zeroshot-per-cell", type=int, default=1000,
                    help="exact sims per zero-shot sampler cell (random, importance, spotcheck)")
    ap.add_argument("--allow-local-h", action="store_true",
                    help="permit zero-shot spotcheck cells off-HPC (smoke tests only; "
                         "the sklearn probe is platform-sensitive)")
    ap.add_argument("--classes", default="", help="substring filter on class_key")
    ap.add_argument("--models", default="", help="substring filter on model")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--merge", action="store_true",
                    help="concatenate per-cell parquet -> estimates_{checkpoints,perstep}.parquet")
    ap.add_argument("--dry-run", action="store_true",
                    help="HPC preflight: report matched cells, then run ONE 1-sim cell "
                         "end-to-end through a parquet round-trip and exit -- catches import, "
                         "simulation, and parquet-engine errors in seconds, before the array")
    args = ap.parse_args()

    # Preflight: the default (matrices) needs only numpy; parquet is required
    # only for the opt-in --reduced artifacts.  Fail in seconds if it is missing.
    if (args.reduced and importlib.util.find_spec("pyarrow") is None
            and importlib.util.find_spec("fastparquet") is None):
        raise SystemExit(
            "parquet engine missing: --reduced writes .parquet cells but neither pyarrow "
            "nor fastparquet is importable. Install once into the env, then resubmit:\n"
            "    conda install -c conda-forge pyarrow      # or: pip install pyarrow")

    fhat_out = Path(args.fhat_out)
    fhat_out.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    perstep_out = Path(args.perstep_out)
    if args.reduced:
        out.mkdir(parents=True, exist_ok=True)
        perstep_out.mkdir(parents=True, exist_ok=True)
    if args.merge:
        for src, name in ((out, "estimates_checkpoints"), (perstep_out, "estimates_perstep")):
            files = sorted(src.glob("*.parquet"))
            if not files:
                print(f"[merge] no parquet in {src}, skipping {name}")
                continue
            merged = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            dest = src.parent / f"{name}.parquet"
            merged.to_parquet(dest, index=False)
            print(f"merged {len(files)} files, {len(merged):,} rows -> {dest}")
        return

    # Group by (class, model); one output file per pair.  A pair is either a set
    # of trajectory runs (FRCNN) or one zero-shot prediction store, never both.
    cells: dict[tuple[str, str], dict] = {}
    for cfg_path in sorted(Path(args.root).rglob("config.json")):
        run = cfg_path.parent
        cfg = json.loads(cfg_path.read_text())
        cls, model = cfg["class_key"], cfg["model"]
        if args.classes not in cls or args.models not in model:
            continue
        cell = cells.setdefault((cls, model), {"runs": [], "store": None})
        if (run / "steps.csv").exists():
            cell["runs"].append((str(run), cfg["sampler"]))
        else:
            cell["store"] = str(run)  # zero-shot g-store, no trajectory

    if args.dry_run:
        n_store = sum(bool(c["store"]) for c in cells.values())
        print(f"[dry-run] {len(cells)} matched cells "
              f"({n_store} zero-shot stores, {len(cells) - n_store} trajectory-run cells); "
              f"checkpoints={args.checkpoints}")
        if not cells:
            raise SystemExit("[dry-run] no cells matched --classes/--models; check the filters.")
        (cls, model), cell = next(iter(cells.items()))
        print(f"[dry-run] smoking 1 sim end-to-end through {cls}/{model} ...")
        if cell["store"]:  # 'random' needs no h-probe, so this stays cheap and cross-platform
            recs, reduced = collect_zeroshot(cell["store"], "random", 0, 1, args.checkpoints, args.reduced)
        else:
            recs, reduced = collect_cell(cell["runs"][0][0], False, 1, args.checkpoints, args.reduced)
        tmp = Path(tempfile.mkdtemp())
        try:
            counts = write_cell_matrices(recs, tmp / "m.npz", cls, model)
            z = np.load(tmp / "m.npz", allow_pickle=False)
            mats = [k for k in z.files if k.startswith("f__")]
            print(f"[dry-run] OK: matrices npz {len(mats)} f-arrays, groups={counts}")
            print(f"[dry-run] sample keys: {mats[:6]}")
            if args.reduced:
                perstep = finalize_perstep(reduced[1])
                reduced[0].to_parquet(tmp / "c.parquet", index=False)
                perstep.to_parquet(tmp / "p.parquet", index=False)
                print(f"[dry-run] reduced OK: checkpoints {reduced[0].shape}, perstep {perstep.shape}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        print("[dry-run] imports, simulation, matrix assembly, and npz I/O all work.")
        return

    for (cls, model), cell in cells.items():
        stem = f"{cls.replace('/', '_')}__{model}"
        fhat_dest = fhat_out / f"{stem}.npz"
        dest = out / f"{stem}.parquet"
        perstep_dest = perstep_out / f"{stem}.parquet"
        done = fhat_dest.exists() and (not args.reduced or (dest.exists() and perstep_dest.exists()))
        if done:
            print(f"[skip] {stem} exists")
            continue
        # Task list: adaptive for every run; frozen additionally for
        # importance/spotcheck; chunked exact sims for zero-shot stores.
        tasks = [(collect_cell, rd, False, args.per_base, args.checkpoints, args.reduced)
                 for rd, _ in cell["runs"]]
        tasks += [(collect_cell, rd, True, args.frozen_per_base, args.checkpoints, args.reduced)
                  for rd, s in cell["runs"] if s in ("importance", "spotcheck")]
        if cell["store"]:
            if sys.platform == "darwin" and not args.allow_local_h:
                raise SystemExit(
                    f"{cls}/{model}: zero-shot spotcheck cells refit the sklearn h probe, "
                    "which diverges on macOS -- run on HPC, or pass --allow-local-h for a "
                    "smoke test (never merge locally-fitted spotcheck cells)."
                )
            chunk = 100
            samplers = (("random", 0), ("importance", 10_000), ("spotcheck", 20_000))
            tasks += [(collect_zeroshot, cell["store"], sampler, seed_off + lo,
                       min(chunk, args.zeroshot_per_cell - lo), args.checkpoints, args.reduced)
                      for sampler, seed_off in samplers
                      for lo in range(0, args.zeroshot_per_cell, chunk)]
        t0 = time.perf_counter()
        if args.jobs > 1:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                futs = [pool.submit(fn, *fn_args) for fn, *fn_args in tasks]
                results = [f.result() for f in as_completed(futs)]
        else:
            results = [fn(*fn_args) for fn, *fn_args in tasks]
        records = [rec for recs, _ in results for rec in recs]
        counts = write_cell_matrices(records, fhat_dest, cls, model)
        msg = f"[done] {stem}: matrices trajs/cell={counts}"
        if args.reduced:
            df = pd.concat([reduced[0] for _, reduced in results], ignore_index=True)
            perstep = finalize_perstep(pd.concat([reduced[1] for _, reduced in results], ignore_index=True))
            df.to_parquet(dest, index=False)
            perstep.to_parquet(perstep_dest, index=False)
            msg += f", {len(df):,} ckpt rows, {len(perstep):,} perstep rows"
        print(f"{msg} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
