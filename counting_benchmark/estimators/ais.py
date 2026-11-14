"""Adaptive importance-sampling estimator (Hamilton et al. 2025).

The core combined without-replacement IS estimator; ``dis``, ``spotais`` and
``spotdis`` delegate here and differ only in which trajectory is fed in.

    F_hat_t   = F(D_t) + f(s_t) / q_t(s_t)
    F_hat_1:t = sum_i alpha_i F_hat_i / sum_i alpha_i
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ._common import EstimatorResult, _as_1d_float


def lure_weights(N: int, w: int, T: int) -> np.ndarray:
    """LURE weights ``omega_i = 1/((N-tau_i)(N-tau_i+1))`` for i=1..T, tau_i=w+i.

    Equivalently ``1/(R_i (R_i - 1))`` with ``R_i = N - tau_i + 1`` the pool size
    remaining before main step i (the ``1/S/(S-1)`` of Hamilton et al.).  Requires
    ``R_i >= 2`` for every step, i.e. ``N - w - T >= 1``; ``MAX_FRAC=0.5`` keeps
    us far from that edge.
    """
    i = np.arange(1, int(T) + 1, dtype=np.float64)
    a = float(N) - float(w) - i  # = N - tau_i ; R_i = a + 1
    if np.any(a < 1.0):
        raise ValueError(
            "LURE weight requires N - w - T >= 1 (remaining pool >= 2 at the last "
            f"step); got N={N}, w={w}, T={T}."
        )
    return 1.0 / (a * (a + 1.0))


def ais(
    f_steps: np.ndarray,
    q_steps: np.ndarray,
    warmup_sum: float,
    N: int,
    w: int,
    *,
    sqrt_index: Literal["main", "combined", "none"] = "main",
) -> EstimatorResult:
    """Combined WOR importance-sampling estimate and Var_simple at each budget t.

    Args:
        f_steps: ``f(s_t)`` for main steps t=1..T, in draw order, shape ``(T,)``.
        q_steps: ``q_t(s_t)`` -- the drawn unit's proposal probability, ``(T,)``.
            Must be strictly positive (unbiasedness of the per-step estimator).
        warmup_sum: ``F(W)``, exact sum over the warmup units.
        N: pool size ``|Omega|``.
        w: warmup-set size ``|W|``.
        sqrt_index: index used inside the sqrt factor of the combined weight.
            ``"main"`` (default) -> ``sqrt(i)``, the count of per-step IS
            estimators being combined; ``"combined"`` -> ``sqrt(w+i)``; ``"none"``
            -> ``1`` (plain LURE, no sqrt factor).  LURE times ``sqrt(i)`` is the
            COMB weighting of Hamilton et al. (2025), meant for adaptive
            proposals, so it is applied only when the detector ``g`` adapts
            across steps; frozen-``g`` samplers (DIS, spotDIS, and every
            zero-shot store) use ``"none"``.

    Returns:
        ``EstimatorResult`` with the combined estimate ``F_hat`` and ``Var_simple``
        at each budget t=1..T.  ``variance[0]`` is 0 (a single per-step estimator
        has no spread).
    """
    f = _as_1d_float("f_steps", f_steps)
    q = _as_1d_float("q_steps", q_steps)
    if f.size != q.size:
        raise ValueError(f"f_steps ({f.size}) and q_steps ({q.size}) length mismatch")
    if np.any(q <= 0.0):
        raise ValueError("q_steps must be strictly positive")

    # Per-step estimator: exact labeled prefix + one-draw IS of the remainder.
    cum_f = np.cumsum(f)
    F_D = float(warmup_sum) + cum_f - f  # F(D_i) = F(W) + sum_{j<i} f_j
    per = F_D + f / q  # F_hat_i
    return combine_per_step(per, N, w, sqrt_index=sqrt_index)


def combine_per_step(
    per: np.ndarray, N: int, w: int, *,
    sqrt_index: Literal["main", "combined", "none"] = "main",
) -> EstimatorResult:
    """Combine unbiased per-step estimates ``per`` (F_hat_i) into the running
    LURE-weighted estimate and Var_simple at each budget t.

    Shared by every estimator in the AIS family: they differ only in how ``per``
    is built (plain IS, model-assisted difference, ...), never in the weighting.
    """
    per = _as_1d_float("per", per)
    T = per.size

    # Combined weights alpha_i = omega_i^LURE * sqrt(index).
    omega = lure_weights(N, w, T)
    i = np.arange(1, T + 1, dtype=np.float64)
    if sqrt_index == "main":
        sqrt_fac = np.sqrt(i)
    elif sqrt_index == "combined":
        sqrt_fac = np.sqrt(float(w) + i)
    elif sqrt_index == "none":
        sqrt_fac = np.ones_like(i)  # plain LURE weights: drop the proposal-adaptation factor
    else:  # pragma: no cover -- guarded by the Literal type
        raise ValueError(
            f"sqrt_index must be 'main', 'combined', or 'none', got {sqrt_index!r}")
    alpha = omega * sqrt_fac

    # Running normalized weighted mean: F_hat_1:t = sum(alpha*per)_{<=t} / A_t.
    A = np.cumsum(alpha)  # A_t = sum_{i<=t} alpha_i  (weight normalizer, not the population variance)
    F_hat = np.cumsum(alpha * per) / A

    # Var_simple_t = sum_{i<=t} alpha_i^2 (per_i - F_hat_t)^2 / A_t^2, expanded so
    # every budget is an O(1) read off these alpha-power prefix sums.
    a2 = alpha * alpha
    A2 = np.cumsum(a2)  # sum_{i<=t} alpha_i^2
    A2P = np.cumsum(a2 * per)  # sum_{i<=t} alpha_i^2 per_i
    A2P2 = np.cumsum(a2 * per * per)  # sum_{i<=t} alpha_i^2 per_i^2
    num = A2P2 - 2.0 * F_hat * A2P + (F_hat * F_hat) * A2
    variance = np.maximum(num / (A * A), 0.0)

    # ``per`` is carried out so the collector can cache the reweightable primitive
    # (F_hat_i pre-combine); F_hat/variance above are the chosen-weighting reduction.
    return EstimatorResult(F_hat=F_hat, variance=variance, per=per)
