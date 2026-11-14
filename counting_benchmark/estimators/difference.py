r"""Difference estimators (model-assisted counting) -- incremental and canonical.

    incremental  F_hat_t = F(D_t) + c_t sum_{Omega \ D_t} g
                           + (f(s_t) - c_t g(s_t)) / q_t(s_t)
    canonical    F_hat_t = F(W) + [G_t(Omega) - G_t(W)]
                           + (M / t) sum_{i<=t} (f(s_i) - g(s_i))

The incremental form is the ``c_t = 1`` member of the plug-in-coefficient
template ``plugin_coefficient``, which ``ratio`` and ``regression`` fit their
own ``c_t`` for and delegate to.  Exactly unbiased for any ``D_t``-measurable
``c_t``.  The canonical form is the warmup-aware SRSWOR textbook estimator,
kept for comparison; the paper reports the incremental form.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ._common import EstimatorResult, _as_1d_float
from .ais import combine_per_step


def plugin_coefficient(
    f_steps: np.ndarray,
    q_steps: np.ndarray,
    warmup_sum: float,
    N: int,
    w: int,
    *,
    g_st: np.ndarray,
    g_remainder: np.ndarray,
    c_steps: np.ndarray,
    sqrt_index: Literal["main", "combined", "none"] = "main",
) -> EstimatorResult:
    """Plug-in-coefficient template estimate + Var_simple at each budget t.

    Args:
        f_steps, q_steps: as in ``ais`` -- ``f(s_t)`` and its proposal prob, ``(T,)``.
        warmup_sum: ``F(W)``, exact warmup sum.
        N, w: pool and warmup sizes.
        g_st: active-snapshot detector prediction at the drawn unit ``g(s_t)``, ``(T,)``.
        g_remainder: active-snapshot detector sum over the *unlabeled* remainder
            ``sum_{k in Omega\\D_t} g(k)``, ``(T,)`` (``D_t`` excludes ``s_t``).
        c_steps: coefficient ``c_t`` per step, ``(T,)``.  Must be ``D_t``-measurable
            (built from the labeled prefix only, never from ``s_t``) -- that is
            what keeps the template exactly unbiased.
        sqrt_index: passed through to the shared combiner.
    """
    f = _as_1d_float("f_steps", f_steps)
    q = _as_1d_float("q_steps", q_steps)
    g_s = _as_1d_float("g_st", g_st)
    g_rem = _as_1d_float("g_remainder", g_remainder)
    c = _as_1d_float("c_steps", c_steps)
    if not (f.size == q.size == g_s.size == g_rem.size == c.size):
        raise ValueError(
            "f_steps, q_steps, g_st, g_remainder, c_steps must share length"
        )
    if np.any(q <= 0.0):
        raise ValueError("q_steps must be strictly positive")

    F_D = float(warmup_sum) + np.cumsum(f) - f  # exact labeled prefix F(D_t)
    per = F_D + c * g_rem + (f - c * g_s) / q  # model-assisted per-step estimate
    return combine_per_step(per, N, w, sqrt_index=sqrt_index)


def difference_incremental(
    f_steps: np.ndarray,
    q_steps: np.ndarray,
    warmup_sum: float,
    N: int,
    w: int,
    *,
    g_st: np.ndarray,
    g_remainder: np.ndarray,
    sqrt_index: Literal["main", "combined", "none"] = "main",
) -> EstimatorResult:
    """Incremental generalized difference estimate: the template at ``c_t = 1``.

    Arguments as :func:`plugin_coefficient` (minus ``c_steps``).
    """
    ones = np.ones(np.asarray(f_steps).size)
    return plugin_coefficient(
        f_steps,
        q_steps,
        warmup_sum,
        N,
        w,
        g_st=g_st,
        g_remainder=g_remainder,
        c_steps=ones,
        sqrt_index=sqrt_index,
    )


def difference(
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
    """Warmup-aware canonical SRSWOR difference estimate + variance at budget ``t``.

    ``F_hat_t = F(W) + [G_t(Omega) - G_t(W)] + (M/t) sum_{i<=t}(f(s_i) - g(s_i))``,
    with ``M = N - w`` the size of ``Omega \\ W``.  The warmup labels are exact, so
    only the ``t`` uniform main draws carry sampling variance; with ``w = 0`` this
    is the plain textbook estimator over the whole sample.  A direct estimator
    (``per = None``, no combining), the regression estimator's fixed-slope
    (``beta = 1``) cousin.

    Args:
        f_steps: ``f(s_t)`` for main steps t=1..T, in draw order, shape ``(T,)``.
        warmup_sum: ``F(W)``, the exact sum over the warmup units.
        N, w: pool size ``|Omega|`` and warmup size ``|W|`` (so ``M = N - w``).
        g_total, g_warmup: active-snapshot detector sums over Omega and over the
            warmup set ``W``, each ``(T,)`` (``g_warmup = 0`` when ``w = 0``).
        g_main: active-snapshot detector sum over the ``t`` main draws
            ``sum_{i<=t} g(s_i)``, ``(T,)``.
        gg_main, ff_main, gf_main: main-draw sums over ``{s_1..s_t}`` of ``g^2``,
            ``f^2``, ``g*f`` -- the sufficient statistics for the residual
            variance, ``(T,)``.

    Variance (SRSWOR total over ``Omega \\ W``): ``v = M(M-t)/t * s_d^2`` with
    ``s_d^2`` the sample variance of the residuals ``d = f - g`` over the ``t`` main
    draws (nan at ``t < 2``).
    """
    f = _as_1d_float("f_steps", f_steps)
    g_om = _as_1d_float("g_total", g_total)
    g_w = _as_1d_float("g_warmup", g_warmup)
    g_m = _as_1d_float("g_main", g_main)
    ggm = _as_1d_float("gg_main", gg_main)
    gfm = _as_1d_float("gf_main", gf_main)
    ffm = _as_1d_float("ff_main", ff_main)
    if not (
        f.size == g_om.size == g_w.size == g_m.size == ggm.size == gfm.size == ffm.size
    ):
        raise ValueError(
            "f_steps, g_total, g_warmup, g_main, gg_main, gf_main, ff_main must share length"
        )

    T = f.size
    t = np.arange(1, T + 1, dtype=np.float64)  # main-sample size |{s_1..s_t}|
    M = float(N) - float(w)  # |Omega \ W|
    F_main = np.cumsum(f)  # sum_{i<=t} f(s_i)
    D_main = F_main - g_m  # sum_{i<=t} (f - g)
    G_rem = g_om - g_w  # G_t(Omega \ W)
    F_hat = float(warmup_sum) + G_rem + (M / t) * D_main

    # s_d^2 = [ sum (f-g)^2 - (sum (f-g))^2 / t ] / (t - 1)
    dd = ffm - 2.0 * gfm + ggm  # sum_{i<=t} (f - g)^2
    with np.errstate(invalid="ignore", divide="ignore"):
        s2 = (dd - D_main * D_main / t) / (t - 1.0)
        variance = M * (M - t) / t * s2
    variance = np.where(t >= 2.0, np.maximum(variance, 0.0), np.nan)

    return EstimatorResult(F_hat=F_hat, variance=variance, per=None)
