"""Post-hoc counting estimators: pure functions over trajectory arrays.

Each consumes plain numpy arrays digested from a run's artifacts by the
``analysis/`` layer and returns curves over the main-step budget t.

The paper's estimators, as applied by ``analysis/run_estimators.py``:

    monte_carlo             model-free expansion estimator (uniform sampling)
    model_only              model count, with and without revealed labels
    difference_incremental  incremental difference (c_t = 1)
    ratio_incremental       incremental ratio (prefix-fit coefficient)
    regression_incremental  incremental regression (prefix OLS coefficient)
    ais                     adaptive importance sampling (Hamilton et al. 2025)
    dis                     DISCount fixed-model variant (delegates to ais)
    spotais                 spot-checked adaptive IS (delegates to ais)
    spotdis                 spot-checked DISCount (delegates to ais)

The incremental estimators are exactly unbiased and combined with LURE/COMB
weights; the paper's estimator appendix gives the derivation.

``difference``, ``ratio``, and ``regression`` are the classical textbook survey
estimators, fit on the full sample.  They are provided for comparison only: the
pipeline does not call them and the paper does not report them.
"""
from __future__ import annotations

from ._common import EstimatorResult, frac_error
from .ais import ais, combine_per_step, lure_weights
from .difference import difference, difference_incremental, plugin_coefficient
from .dis import dis
from .model_only import ModelOnlyResult, model_only
from .monte_carlo import monte_carlo
from .ratio import ratio, ratio_incremental
from .regression import regression, regression_incremental
from .spotais import spotais
from .spotdis import spotdis

__all__ = [
    "EstimatorResult",
    "frac_error",
    "ais",
    "combine_per_step",
    "lure_weights",
    "plugin_coefficient",
    "difference",
    "difference_incremental",
    "ratio",
    "ratio_incremental",
    "regression",
    "regression_incremental",
    "dis",
    "monte_carlo",
    "model_only",
    "ModelOnlyResult",
    "spotais",
    "spotdis",
]
