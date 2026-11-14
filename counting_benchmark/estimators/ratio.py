"""Ratio estimators -- canonical (textbook) and incremental (ratio-assisted).

The incremental form is the plug-in-coefficient template of ``difference``
with ``c_t`` the prefix-fit ratio ``F(D_t) / G_t(D_t)``; it is the one the paper
reports.  The canonical form is kept for comparison.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ._common import EstimatorResult, _as_1d_float
from .difference import plugin_coefficient


def ratio(
    f_steps: np.ndarray,
    warmup_sum: float,
    N: int,
    w: int,
    *,
    g_total: np.ndarray,
    g_warmup: np.ndarray,
    g_main: np.ndarray,
    gg_main: np.ndarray,
    gf_main: np.ndarray,
    ff_main: np.ndarray,
) -> EstimatorResult:
    """Warmup-aware canonical ratio estimate + SRSWOR variance at budget ``t``.

    Args:
        f_steps: ``f(s_t)`` for main steps t=1..T, in draw order, shape ``(T,)``.
        warmup_sum: ``F(W)``, the exact sum over the warmup units.
        N, w: pool size ``|Omega|`` and warmup size ``|W|`` (so ``M = N - w``).
        g_total, g_warmup: active-snapshot detector sums over Omega ``G_t(Omega)``
            and over the warmup set ``G_t(W)``, each ``(T,)`` (``g_warmup = 0`` when
            ``w = 0``).
        g_main: active-snapshot detector sum over the ``t`` main draws
            ``sum_{i<=t} g(s_i)``, ``(T,)``.
        gg_main, ff_main, gf_main: main-draw sums over ``{s_1..s_t}`` of ``g^2``,
            ``f^2`` and ``g*f`` -- the sufficient statistics for the ratio
            variance, ``(T,)``.

    Returns:
        ``EstimatorResult`` (``per`` is ``None``: a direct estimator, no combining
        stage -- like Monte Carlo / model-only).
    """
    f = _as_1d_float("f_steps", f_steps)
    g_om = _as_1d_float("g_total", g_total)       # G_t(Omega)
    g_w = _as_1d_float("g_warmup", g_warmup)      # G_t(W)
    g_m = _as_1d_float("g_main", g_main)          # sum_{i<=t} g(s_i)
    ggm = _as_1d_float("gg_main", gg_main)
    gfm = _as_1d_float("gf_main", gf_main)
    ffm = _as_1d_float("ff_main", ff_main)
    if not (f.size == g_om.size == g_w.size == g_m.size == ggm.size == gfm.size == ffm.size):
        raise ValueError(
            "f_steps, g_total, g_warmup, g_main, gg_main, gf_main, ff_main must share length"
        )

    T = f.size
    t = np.arange(1, T + 1, dtype=np.float64)      # main-sample size |{s_1..s_t}|
    M = float(N) - float(w)                        # |Omega \ W|
    F_main = np.cumsum(f)                          # sum_{i<=t} f(s_i)
    G_rem = g_om - g_w                             # G_t(Omega \ W)

    have_g = g_m > 0.0
    R_hat = np.where(have_g, F_main / np.where(have_g, g_m, 1.0), 0.0)
    expansion = float(warmup_sum) + (M / t) * F_main   # detector-free fallback
    F_hat = np.where(have_g, float(warmup_sum) + R_hat * G_rem, expansion)

    # SRSWOR ratio-estimator variance: s_e^2 = sum (f - R g)^2 / (t - 1).
    resid_ss = ffm - 2.0 * R_hat * gfm + R_hat * R_hat * ggm
    with np.errstate(invalid="ignore", divide="ignore"):
        s2 = resid_ss / (t - 1.0)
        variance = M * (M - t) / t * s2
    variance = np.where((t >= 2.0) & have_g, np.maximum(variance, 0.0), np.nan)

    return EstimatorResult(F_hat=F_hat, variance=variance, per=None)


def ratio_incremental(
    f_steps: np.ndarray,
    q_steps: np.ndarray,
    warmup_sum: float,
    N: int,
    w: int,
    *,
    g_st: np.ndarray,
    g_remainder: np.ndarray,
    g_D: np.ndarray,
    n_min: int = 10,
    g_min: float = 1.0,
    sqrt_index: Literal["main", "combined", "none"] = "main",
) -> EstimatorResult:
    """Incremental ratio-assisted difference estimate + Var_simple at each budget t.

    The plug-in-coefficient member of the difference family with the *prefix-fit*
    scale ``c_t = R_hat_t = F(D_t) / g_t(D_t)`` (``D_t``-measurable, so the template
    stays exactly unbiased -- unlike the textbook :func:`ratio`, which fits on the
    same sample it corrects and pays O(1/n) bias).  Guards fall back to ``c_t = 1``
    (difference) while the prefix carries no scale information.

    Args:
        f_steps, q_steps, warmup_sum, N, w, g_st, g_remainder: as in
            :func:`estimators.difference.plugin_coefficient`.
        g_D: active-snapshot detector sum over the labeled prefix ``g_t(D_t)``,
            ``(T,)`` (``D_t`` excludes ``s_t``).
        n_min: fall back to ``c_t = 1`` while ``|D_t| = w + t - 1 < n_min``.
        g_min: fall back to ``c_t = 1`` while ``g_t(D_t) < g_min``.
    """
    f = _as_1d_float("f_steps", f_steps)
    gD = _as_1d_float("g_D", g_D)
    if f.size != gD.size:
        raise ValueError(f"f_steps ({f.size}) and g_D ({gD.size}) length mismatch")

    F_D = float(warmup_sum) + np.cumsum(f) - f          # F(D_t)
    n_D = float(w) + np.arange(f.size, dtype=np.float64)  # |D_t| = w + t - 1
    fit = (n_D >= n_min) & (gD >= g_min)
    c = np.where(fit, F_D / np.where(fit, gD, 1.0), 1.0)  # R_hat_t, guarded
    return plugin_coefficient(f_steps, q_steps, warmup_sum, N, w,
                              g_st=g_st, g_remainder=g_remainder, c_steps=c,
                              sqrt_index=sqrt_index)
