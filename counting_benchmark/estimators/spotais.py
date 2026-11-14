"""Spot-checked adaptive IS estimator.

spotAIS is the AIS core applied to a **spot-checked-proposal** trajectory.  The
mixture ``q = alpha*q_spot + (1-alpha)*q_count`` (over the empty filter ``h``) is
already baked into the recorded/reconstructed ``q_t(s_t)`` by
``experiments/sampling.py:_spotcheck_weights``, so the estimator math is
identical to AIS -- and unbiased no matter how wrong ``h`` is, since the mixture
keeps ``q(s) > 0`` for every unit.  Only the trajectory differs, so we delegate.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ._common import EstimatorResult
from .ais import ais


def spotais(
    f_steps: np.ndarray,
    q_steps: np.ndarray,
    warmup_sum: float,
    N: int,
    w: int,
    *,
    sqrt_index: Literal["main", "combined"] = "main",
) -> EstimatorResult:
    """spotAIS estimate: the AIS core on a spot-checked adaptive trajectory.

    Arguments and return value are exactly :func:`estimators.ais.ais`; ``q_steps``
    is expected to come from the spotcheck sampler (adaptive ``g``, ``h``).
    """
    return ais(f_steps, q_steps, warmup_sum, N, w, sqrt_index=sqrt_index)
