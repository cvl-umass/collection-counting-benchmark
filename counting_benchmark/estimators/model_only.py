"""Raw detector-count baseline, on the snapshot active at step t.

    pure       F_hat_t = sum_{s in Omega} g^(t)(s)
    corrected  F_hat_t = F(R_t) + sum_{s not in R_t} g^(t)(s)

Both are deterministic given the snapshot, so neither has per-trajectory
variance; the meaningful spread is across trajectories.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ._common import _as_1d_float


class ModelOnlyResult(NamedTuple):
    """The two detector-count curves over the budget t=1..T (no variance)."""

    F_hat_pure: np.ndarray
    F_hat_corrected: np.ndarray


def model_only(
    f_steps: np.ndarray,
    warmup_sum: float,
    g_total_steps: np.ndarray,
    g_revealed_steps: np.ndarray,
) -> ModelOnlyResult:
    """Pure and label-corrected detector-count estimates at each budget t.

    Args:
        f_steps: ``f(s_t)`` for main steps t=1..T, in draw order, shape ``(T,)``.
        warmup_sum: ``F(W)``, exact sum over the warmup units.
        g_total_steps: ``sum_{Omega} g^(t)``, the active snapshot's full-pool
            prediction sum at each step, shape ``(T,)``.
        g_revealed_steps: ``sum_{R_t} g^(t)``, the active snapshot's prediction
            summed over everything revealed after t main steps, shape ``(T,)``.

    Returns:
        ``ModelOnlyResult`` with ``F_hat_pure`` and ``F_hat_corrected``.
    """
    f = _as_1d_float("f_steps", f_steps)
    g_total = _as_1d_float("g_total_steps", g_total_steps)
    g_revealed = _as_1d_float("g_revealed_steps", g_revealed_steps)
    if not (f.size == g_total.size == g_revealed.size):
        raise ValueError(
            f"length mismatch: f_steps={f.size}, g_total_steps={g_total.size}, "
            f"g_revealed_steps={g_revealed.size}"
        )

    exact_R = float(warmup_sum) + np.cumsum(f)  # F(R_t), R_t inclusive of s_t
    F_hat_pure = g_total.copy()
    F_hat_corrected = exact_R + (g_total - g_revealed)
    return ModelOnlyResult(F_hat_pure=F_hat_pure, F_hat_corrected=F_hat_corrected)
