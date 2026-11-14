"""Derive the reduced estimate artifacts from the per-step F_hat matrices.

Writes per-checkpoint parquets and the per-step ensemble the figures read.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Direct estimators with no LURE/COMB combining stage: their cached fhat IS the
# final F_hat, and the cached ``v__`` array (where present) IS the variance.
# Everything else, including the incremental difference / ratio / regression,
# is cached as per-step estimates and combined here.
PASSTHROUGH = {"monte_carlo", "model_only_corrected"}
_PERSTEP_KEYS = ["model", "class_key", "sampler", "frozen", "estimator", "t"]


def combine_matrix(per: np.ndarray, N: int, w: int, sqrt_index: str) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised ``ais.combine_per_step`` over an ``(n, T)`` matrix of per-step
    estimates: returns combined ``F_hat`` and ``Var_simple`` at every budget,
    each ``(n, T)``.  Weights ``alpha_i = omega_i^LURE * sqrt_fac(i)`` broadcast
    over rows -- identical math to the scalar path, just batched."""
    per = per.astype(np.float64)
    n, T = per.shape
    i = np.arange(1, T + 1, dtype=np.float64)
    a = float(N) - float(w) - i
    omega = 1.0 / (a * (a + 1.0))
    sf = {"none": np.ones_like(i), "main": np.sqrt(i), "combined": np.sqrt(float(w) + i)}[sqrt_index]
    alpha = omega * sf
    A = np.cumsum(alpha)
    F = np.cumsum(alpha * per, axis=1) / A
    a2 = alpha * alpha
    A2 = np.cumsum(a2)
    A2P = np.cumsum(a2 * per, axis=1)
    A2P2 = np.cumsum(a2 * per * per, axis=1)
    num = A2P2 - 2.0 * F * A2P + (F * F) * A2
    var = np.maximum(num / (A * A), 0.0)
    return F, var


def _sqrt_index(weighting: str, frozen: bool, w: int) -> str:
    if weighting == "lure":
        return "none"
    if weighting == "comb":
        return "main"
    adaptive = (not frozen) and w > 0          # detector g adapts across steps
    return "main" if adaptive else "none"


def derive_cell(npz_path: Path, checkpoints: list[float], weighting: str,
                model_only_genuine_only: bool = False
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One cell npz -> (per-sim checkpoint rows, per-step reduced rows)."""
    z = np.load(npz_path, allow_pickle=True)
    N, w = int(z["meta__N"]), int(z["meta__w"])
    F_true = float(z["meta__F_true"])
    cls, model = str(z["meta__class_key"]), str(z["meta__model"])

    cell_has_real = any((z[k][:, 2] == 1).any()
                        for k in z.files if k.startswith("rows__"))

    ckpt_parts, perstep_parts = [], []
    for key in (k for k in z.files if k.startswith("f__")):
        _, sampler, frz, est = key.split("__", 3)
        frozen = bool(int(frz))
        rows = z[f"rows__{sampler}__{frz}"]           # (n, 3): seed, base_seed, source
        fhat = z[key]                                  # (n, T)
        if est == "model_only_corrected" and model_only_genuine_only and cell_has_real:
            real = rows[:, 2] == 1                     # genuine runs only (see docstring)
            if not real.any():
                continue                               # frozen block: nothing genuine to report
            rows, fhat = rows[real], fhat[real]
        n, T = fhat.shape
        t = np.arange(1, T + 1)
        frac_labeled = (w + t) / N

        if est in PASSTHROUGH:
            F = fhat.astype(np.float64)
            V = (z[f"v__{sampler}__{frz}__{est}"].astype(np.float64)
                 if f"v__{sampler}__{frz}__{est}" in z.files else np.full((n, T), np.nan))
        else:
            F, V = combine_matrix(fhat, N, w, _sqrt_index(weighting, frozen, w))
        FE = (F - F_true) / F_true

        # mean analytical variance per t, skipping all-NaN columns (Monte Carlo's
        # t=1 has no SRSWOR variance; model-only is NaN throughout) without the
        # empty-slice warning nanmean would raise on them.
        vmean = np.full(T, np.nan)
        finite = np.isfinite(V).any(0)
        if finite.any():
            vmean[finite] = np.nanmean(V[:, finite], 0)

        # --- per-step reduction across the n trajectories, at every t ---
        perstep_parts.append(pd.DataFrame({
            "model": model, "class_key": cls, "sampler": sampler, "frozen": frozen,
            "estimator": est, "t": t, "n_traj": n, "frac_labeled": frac_labeled,
            "F_hat_mean": F.mean(0),
            "F_hat_var_emp": F.var(0, ddof=1) if n > 1 else np.full(T, np.nan),
            "F_hat_sd_emp": F.std(0, ddof=1) if n > 1 else np.full(T, np.nan),
            "variance_analytic_mean": vmean,
            "mean_frac_error": FE.mean(0),
            "sd_frac_error": FE.std(0, ddof=1) if n > 1 else np.full(T, np.nan),
            "mean_abs_frac_error": np.abs(FE).mean(0),
            "F_true": F_true,
        }))

        # --- per-sim rows at each checkpoint budget (first t at/past the frac) ---
        idx = np.minimum(np.searchsorted(frac_labeled, checkpoints, side="left"), T - 1)
        for cp, ti in zip(checkpoints, idx):
            ckpt_parts.append(pd.DataFrame({
                "estimator": est, "t": t[ti], "frac_labeled": frac_labeled[ti],
                "F_hat": F[:, ti], "variance": V[:, ti], "frac_error": FE[:, ti],
                "F_true": F_true, "checkpoint": cp, "sampler": sampler,
                "class_key": cls, "model": model,
                "seed": rows[:, 0], "base_seed": rows[:, 1],
                "source": np.where(rows[:, 2] == 1, "real", "sim"), "frozen": frozen,
            }))
    return (pd.concat(ckpt_parts, ignore_index=True),
            pd.concat(perstep_parts, ignore_index=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fhat-in", default="outputs/analysis/perstep_Fhat",
                    help="dir of per-cell F_hat matrix npz")
    ap.add_argument("--out", default="outputs/analysis/checkpoints",
                    help="per-cell per-SIM checkpoint parquet dir")
    ap.add_argument("--perstep-out", default="outputs/analysis/perstep",
                    help="per-cell per-STEP reduced-summary parquet dir")
    ap.add_argument("--weighting", choices=["auto", "lure", "comb"], default="auto")
    ap.add_argument("--checkpoints", nargs="+", type=float,
                    default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    ap.add_argument("--classes", default="", help="substring filter on the npz filename")
    ap.add_argument("--model-only-genuine-only", action="store_true",
                    help="restrict model_only_corrected to genuine (real) runs only, "
                         "wherever the cell has them.  The default emits the full "
                         "1000-trajectory ensemble.")
    args = ap.parse_args()

    fhat_in = Path(args.fhat_in)
    out, perstep_out = Path(args.out), Path(args.perstep_out)
    out.mkdir(parents=True, exist_ok=True)
    perstep_out.mkdir(parents=True, exist_ok=True)
    files = [f for f in sorted(fhat_in.glob("*.npz")) if args.classes in f.stem]
    if not files:
        raise SystemExit(f"no npz in {fhat_in} matching --classes {args.classes!r}")

    print(f"deriving {len(files)} cells | weighting={args.weighting} | checkpoints={args.checkpoints} "
          f"| model_only_genuine_only={args.model_only_genuine_only}")
    for f in files:
        ckpt, perstep = derive_cell(f, args.checkpoints, args.weighting,
                                    args.model_only_genuine_only)
        ckpt.to_parquet(out / f"{f.stem}.parquet", index=False)
        perstep.to_parquet(perstep_out / f"{f.stem}.parquet", index=False)
        print(f"[done] {f.stem}: {len(ckpt):,} ckpt rows, {len(perstep):,} perstep rows")


if __name__ == "__main__":
    main()
