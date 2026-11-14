"""DISCount fixed-model baseline.

DIS is the *identical* combined-WOR IS estimator as AIS -- same per-step
estimator, same LURE-x-sqrt weights, same Var_simple.  The only difference is
upstream: the trajectory it consumes was drawn with ``q`` built from a frozen
snapshot-0 ``g`` instead of the adaptive snapshot schedule.  That difference
lives in the ``analysis/`` trajectory builder, so here we simply delegate to the
AIS core.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ._common import EstimatorResult
from .ais import ais


def dis(
    f_steps: np.ndarray,
    q_steps: np.ndarray,
    warmup_sum: float,
    N: int,
    w: int,
    *,
    sqrt_index: Literal["main", "combined"] = "main",
) -> EstimatorResult:
    """DISCount estimate: the AIS core applied to a frozen-``g`` trajectory.

    Arguments and return value are exactly :func:`estimators.ais.ais`; ``q_steps``
    is expected to come from a fixed snapshot-0 proposal.
    """
    return ais(f_steps, q_steps, warmup_sum, N, w, sqrt_index=sqrt_index)
