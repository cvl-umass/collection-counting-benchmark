"""Apply every estimator to a ``Trajectory`` -> tidy long-format DataFrame.

One row per (estimator, t): ``F_hat``, ``variance``, ``frac_error``,
``frac_labeled``, plus provenance columns.  The IS estimator is emitted under the
trajectory's ``is_name`` (ais / dis / spotais / spotdis / is_uniform); the
underlying math is the AIS core in every case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from estimators import (
    ais, difference_incremental, frac_error, model_only, monte_carlo,
    ratio_incremental, regression_incremental,
)

from .trajectory import Trajectory


def estimate_all(
    traj: Trajectory, *, sqrt_index: str | None = None, model_assisted: bool | None = None
) -> pd.DataFrame:
    """``model_assisted`` gates the model-assisted estimators (the incremental
    difference, ratio, and regression).  Default (None) emits them on random and
    adaptive trajectories, plus frozen single-snapshot runs (zero-shot models).

    ``sqrt_index`` (None -> auto) picks the combining weights.  ``"main"`` gives
    the COMB weights of Hamilton et al. (2025), LURE times sqrt(i), used when the
    detector ``g`` adapts across steps (multi-snapshot, non-frozen).  Every
    frozen-``g`` proposal -- all DIS/spotDIS settings and all zero-shot stores,
    where ``g`` is a single fixed snapshot -- combines with plain LURE (``"none"``).
    Pass an explicit value to override.
    """
    zero_shot = traj.g.shape[0] == 1
    frozen = bool(traj.meta.get("frozen", False))
    if sqrt_index is None:
        sqrt_index = "main" if (not frozen and not zero_shot) else "none"
    if model_assisted is None:
        model_assisted = traj.meta.get("sampler") in ("random", "importance") and (
            not frozen or zero_shot
        )
    T = traj.T
    t = np.arange(1, T + 1)
    frac_labeled = (traj.w + t) / traj.N
    parts = []

    def add(name, F_hat, variance, per=None):
        # fhat_perstep is the reweightable per-step primitive: the AIS family's
        # pre-combine F_hat_i (``per``), or the plain per-step curve where there is
        # no combining stage (Monte Carlo, model-only).  The collector caches this.
        parts.append(
            pd.DataFrame(
                {
                    "estimator": name,
                    "t": t,
                    "frac_labeled": frac_labeled,
                    "F_hat": F_hat,
                    "variance": variance,
                    "frac_error": frac_error(F_hat, traj.F_true),
                    "fhat_perstep": F_hat if per is None else per,
                }
            )
        )

    mc = monte_carlo(traj.f_steps, traj.warmup_sum, traj.N, traj.w)
    add("monte_carlo", mc.F_hat, mc.variance)

    a = ais(traj.f_steps, traj.q_steps, traj.warmup_sum, traj.N, traj.w, sqrt_index=sqrt_index)
    add(traj.meta["is_name"], a.F_hat, a.variance, per=a.per)

    if model_assisted:
        # The paper's incremental difference / ratio / regression: one family of
        # per-step estimates that differ only in the coefficient c_t, each fit on
        # the labeled prefix D_t and combined with the same LURE/COMB weights.
        common = (traj.f_steps, traj.q_steps, traj.warmup_sum, traj.N, traj.w)
        g_st, g_rem = traj.difference_inputs()
        g_D, gg_D, gf_D = traj.coefficient_inputs()
        diff = difference_incremental(*common, g_st=g_st, g_remainder=g_rem,
                                      sqrt_index=sqrt_index)
        add("difference", diff.F_hat, diff.variance, per=diff.per)
        rat = ratio_incremental(*common, g_st=g_st, g_remainder=g_rem, g_D=g_D,
                                sqrt_index=sqrt_index)
        add("ratio", rat.F_hat, rat.variance, per=rat.per)
        reg = regression_incremental(*common, g_st=g_st, g_remainder=g_rem, g_D=g_D,
                                     gg_D=gg_D, gf_D=gf_D, sqrt_index=sqrt_index)
        add("regression", reg.F_hat, reg.variance, per=reg.per)

    g_total, g_revealed = traj.model_only_inputs()
    mo = model_only(traj.f_steps, traj.warmup_sum, g_total, g_revealed)
    nan = np.full(T, np.nan)
    add("model_only_pure", mo.F_hat_pure, nan)
    add("model_only_corrected", mo.F_hat_corrected, nan)

    out = pd.concat(parts, ignore_index=True)
    out["F_true"] = traj.F_true
    for key in ("run_dir", "sampler", "class_key", "model", "seed", "source",
                "frozen", "base_seed"):
        out[key] = traj.meta.get(key)
    return out
