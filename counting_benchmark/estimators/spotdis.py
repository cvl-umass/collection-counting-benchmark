"""Spot-checked DISCount baseline.

spotDIS is the AIS core applied to a spot-checked-proposal trajectory whose
proposal is built from a **frozen** snapshot-0 model.  As with ``dis.py`` the
estimator math is identical to AIS; the fixed-model choice (freeze ``(g[0],
h[0])`` as a package, or freeze ``g[0]`` and refit ``h``) is made upstream in the
``analysis/`` trajectory builder.  We delegate.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ._common import EstimatorResult
from .ais import ais


def spotdis(
    f_steps: np.ndarray,
    q_steps: np.ndarray,
    warmup_sum: float,
    N: int,
    w: int,
    *,
    sqrt_index: Literal["main", "combined"] = "main",
) -> EstimatorResult:
    """spotDIS estimate: the AIS core on a frozen-model spot-checked trajectory.

    Arguments and return value are exactly :func:`estimators.ais.ais`; ``q_steps``
    is expected to come from the spotcheck sampler with a frozen snapshot-0 model.
    """
    return ais(f_steps, q_steps, warmup_sum, N, w, sqrt_index=sqrt_index)
