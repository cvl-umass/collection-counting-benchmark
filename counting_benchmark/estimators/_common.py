"""Shared types and helpers for the estimator package."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class EstimatorResult(NamedTuple):
    """Curves over the main-step budget t = 1..T.

    ``F_hat[k]`` / ``variance[k]`` are the estimate and its variance computed
    from the first ``k+1`` main steps.  ``variance`` is ``nan`` where undefined
    (e.g. the Monte Carlo sample variance at t=1).

    ``per`` holds the *pre-combine* per-step estimates ``F_hat_i`` for the AIS
    family (the reweightable primitive: any LURE/COMB weighting is a function of
    ``per`` alone -- see ``ais.combine_per_step``).  It is ``None`` for estimators
    with no combining stage (Monte Carlo, model-only), whose per-step curve *is*
    ``F_hat``.
    """

    F_hat: np.ndarray
    variance: np.ndarray
    per: np.ndarray | None = None


def frac_error(F_hat: np.ndarray, F_true: float) -> np.ndarray:
    """Signed fractional error ``(F_hat - F_true) / F_true`` at each budget."""
    return (np.asarray(F_hat, dtype=np.float64) - float(F_true)) / float(F_true)


def _as_1d_float(name: str, arr) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    return a
