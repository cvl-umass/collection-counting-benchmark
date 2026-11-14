"""Simple Monte Carlo estimator -- two-phase survey mean.

Warmup is counted exactly; the remaining ``M = N - w`` units are estimated by
the sample mean of the ``t`` main draws.  This is unbiased only when the main
draws are a uniform without-replacement sample, i.e. the **random** sampler; fed
an importance/spotcheck trajectory it is biased (draws over-weight high-``g``
units), so the pipeline only reports it on the random sampler.
"""

from __future__ import annotations

import numpy as np

from ._common import EstimatorResult, _as_1d_float


def monte_carlo(
    f_steps: np.ndarray,
    warmup_sum: float,
    N: int,
    w: int,
) -> EstimatorResult:
    """Two-phase survey-mean estimate and its SRSWOR variance at each budget t.

    Args:
        f_steps: ``f(s_t)`` for main steps t=1..T, in draw order, shape ``(T,)``.
        warmup_sum: ``F(W)``, the exact sum over the warmup units.
        N: pool size ``|Omega|``.
        w: warmup-set size ``|W|``.

    Returns:
        ``EstimatorResult`` with

            F_hat_t     = F(W) + (N - w) * mean(f_steps[:t])
            variance_t  = (N-w)(N-w-t) * s_t^2 / t          (FPC form)

        ``s_t^2`` is the divisor-(t-1) sample variance, which is unbiased for the
        SRSWOR population variance ``S^2``.  ``variance`` is ``nan`` at t=1 (a
        single draw has no sample variance) and ``0`` at t=M (whole non-warmup
        pool observed -> estimate is exact).
    """
    f = _as_1d_float("f_steps", f_steps)
    T = f.size
    M = int(N) - int(w)
    if M <= 0:
        raise ValueError(f"N - w must be positive (got N={N}, w={w})")
    if T > M:
        raise ValueError(f"more main steps ({T}) than non-warmup units ({M})")

    t = np.arange(1, T + 1, dtype=np.float64)
    cum_f = np.cumsum(f)
    mean = cum_f / t
    F_hat = float(warmup_sum) + M * mean

    # s_t^2 = (sum f^2 - (sum f)^2 / t) / (t - 1); undefined (nan) at t=1.
    cum_f2 = np.cumsum(f * f)
    with np.errstate(invalid="ignore", divide="ignore"):
        s2 = (cum_f2 - cum_f * cum_f / t) / (t - 1.0)
        variance = M * (M - t) * s2 / t
    variance = np.where(t >= 2, np.maximum(variance, 0.0), np.nan)

    return EstimatorResult(F_hat=F_hat, variance=variance)
