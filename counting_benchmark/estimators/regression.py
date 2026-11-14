"""Regression estimators -- canonical (textbook) and incremental (regression-assisted).

The incremental form is the plug-in-coefficient template of ``difference``
with ``c_t`` the prefix-fit OLS slope of ``f`` on ``g``; it is the one the paper
reports.  The canonical form is kept for comparison.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ._common import EstimatorResult, _as_1d_float
from .difference import plugin_coefficient


def regression(
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
    """Warmup-aware canonical regression estimate + SRSWOR variance at budget ``t``.

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
            ``f^2`` and ``g*f`` -- the sufficient statistics for the OLS slope and
            the regression variance, ``(T,)``.

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

    # Centered sums over the t main draws: S_xx = sum (g-g_bar)^2,
    # S_xy = sum (g-g_bar)(f-f_bar), S_yy = sum (f-f_bar)^2.
    S_xx = ggm - g_m * g_m / t
    S_xy = gfm - g_m * F_main / t
    S_yy = ffm - F_main * F_main / t
    fit = S_xx > 1e-12 * np.maximum(ggm, 1.0)
    beta = np.where(fit, S_xy / np.where(fit, S_xx, 1.0), 0.0)  # 0 -> expansion fallback
    F_hat = float(warmup_sum) + (M / t) * F_main + beta * (G_rem - (M / t) * g_m)

    # SRSWOR regression-estimator variance: residual SS = S_yy - beta^2 S_xx.
    resid_ss = S_yy - beta * beta * S_xx
    with np.errstate(invalid="ignore", divide="ignore"):
        s2 = resid_ss / (t - 2.0)
        variance = M * (M - t) / t * s2
    variance = np.where(t >= 3.0, np.maximum(variance, 0.0), np.nan)

    return EstimatorResult(F_hat=F_hat, variance=variance, per=None)


def regression_incremental(
    f_steps: np.ndarray,
    q_steps: np.ndarray,
    warmup_sum: float,
    N: int,
    w: int,
    *,
    g_st: np.ndarray,
    g_remainder: np.ndarray,
    g_D: np.ndarray,
    gg_D: np.ndarray,
    gf_D: np.ndarray,
    n_min: int = 10,
    sqrt_index: Literal["main", "combined", "none"] = "main",
) -> EstimatorResult:
    """Incremental regression-assisted difference estimate + Var_simple at budget t.

    The plug-in-coefficient member of the difference family with the *prefix* OLS
    slope (slope-only, unweighted) as ``c_t``

        c_t = sum_{D_t}(g - g_bar)(f - f_bar) / sum_{D_t}(g - g_bar)^2,

    computed from the prefix sufficient statistics as
    ``num_t = gf_D - g_D F(D_t)/|D_t|``, ``den_t = gg_D - g_D^2/|D_t|``.  Being
    ``D_t``-measurable it keeps the template exactly unbiased -- unlike the textbook
    :func:`regression`, whose same-sample fit costs O(1/n) bias.  Guards fall back
    to ``c_t = 1`` (difference) while the fit is degenerate.

    Args:
        f_steps, q_steps, warmup_sum, N, w, g_st, g_remainder: as in
            :func:`estimators.difference.plugin_coefficient`.
        g_D, gg_D, gf_D: active-snapshot prefix sums over ``D_t`` (excluding
            ``s_t``) of ``g``, ``g^2`` and ``g*f``, each ``(T,)``.
        n_min: fall back to ``c_t = 1`` while ``|D_t| = w + t - 1 < n_min``.
    """
    f = _as_1d_float("f_steps", f_steps)
    gD = _as_1d_float("g_D", g_D)
    ggD = _as_1d_float("gg_D", gg_D)
    gfD = _as_1d_float("gf_D", gf_D)
    if not (f.size == gD.size == ggD.size == gfD.size):
        raise ValueError("f_steps, g_D, gg_D, gf_D must share length")

    F_D = float(warmup_sum) + np.cumsum(f) - f            # F(D_t)
    n_D = float(w) + np.arange(f.size, dtype=np.float64)  # |D_t| = w + t - 1
    n_safe = np.maximum(n_D, 1.0)
    num = gfD - gD * F_D / n_safe                          # sum (g - g_bar)(f - f_bar)
    den = ggD - gD * gD / n_safe                           # sum (g - g_bar)^2
    fit = (n_D >= n_min) & (den > 1e-12 * np.maximum(ggD, 1.0))
    c = np.where(fit, num / np.where(fit, den, 1.0), 1.0)  # OLS slope, guarded
    return plugin_coefficient(f_steps, q_steps, warmup_sum, N, w,
                              g_st=g_st, g_remainder=g_remainder, c_steps=c,
                              sqrt_index=sqrt_index)
