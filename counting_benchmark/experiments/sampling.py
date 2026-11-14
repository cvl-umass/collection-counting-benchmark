"""Pure functions over numpy arrays; no state, no I/O.  The main loop owns the
RNG, the revealed set, and the current model predictions, and calls these
functions one step at a time.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Set, Tuple

import numpy as np


def sample_unit(
    revealed_set: Set[int],
    n_total: int,
    pred_counts: np.ndarray,
    *,
    sampler: Literal["random", "importance", "spotcheck"],
    min_count_per_tile: float,
    rng: np.random.Generator,
    h_empty: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    routing: Literal["trust_h", "trust_g"] = "trust_h",
) -> Tuple[int, float]:
    """Sample one unrevealed unit and return its index and sampling probability.

    Args:
        revealed_set: Tiles already revealed (warmup + previous IS steps).
        n_total: Total pool size (``len(dataset)``).
        pred_counts: Current model predictions ``g(s)``, shape ``(n_total,)``.
        sampler: ``"random"`` (uniform), ``"importance"`` (``q ∝ max(g,
            min_count_per_tile)``), or ``"spotcheck"`` (mixture ``q = α·q_spot +
            (1-α)·q_count`` over an empty filter ``h``; see ``_spotcheck_weights``).
        min_count_per_tile: Lower floor on the per-tile weight used to build the
            proposal, ensuring every unrevealed unit can be sampled.  Used only
            by ``"importance"``.
        rng: numpy random generator (advanced in-place).
        h_empty: Boolean empty-filter prediction ``h(s)`` (``True`` = predicted
            empty), shape ``(n_total,)``.  Required by ``"spotcheck"``.
        alpha: Spot-check budget ``α`` for ``"spotcheck"`` -- the mixture weight on
            the uniform spot proposal over suspected-empty units.
        routing: ``"trust_h"`` or ``"trust_g"``; resolves the ``h``/``g``
            disagreement cell for ``"spotcheck"`` (see ``_spotcheck_weights``).

    Returns:
        ``(s_t, q_t_st)``: the sampled tile index and its normalized
        probability under the current proposal.
    """
    # 1D array of unrevealed units' indices; len(unrevealed) = n_total - len(revealed_set)
    unrevealed = np.fromiter(
        (i for i in range(n_total) if i not in revealed_set),
        dtype=np.int64,
    )
    if unrevealed.size == 0:
        raise RuntimeError("sample_unit called with no unrevealed units left.")

    if sampler == "importance":
        # Avoid zero weight
        weights = np.maximum(
            pred_counts[unrevealed].astype(np.float64),
            float(min_count_per_tile),
        )
    elif sampler == "random":
        # Equal weight over all remaining units
        weights = np.ones(unrevealed.size, dtype=np.float64)
    elif sampler == "spotcheck":
        weights = _spotcheck_weights(
            pred_counts[unrevealed], h_empty[unrevealed], float(alpha), routing
        )
    else:
        raise ValueError(f"Unknown sampler {sampler!r}")

    total = float(weights.sum())
    if total <= 0:
        raise RuntimeError(
            "Sampling weights summed to 0; check min_count_per_tile and pred_counts."
        )
    # Weight normalization: q(s) = g(s)/G(S)
    weights /= total

    local_idx = int(rng.choice(unrevealed.size, p=weights))
    return int(unrevealed[local_idx]), float(weights[local_idx])


def _spotcheck_weights(
    pred_counts: np.ndarray,
    h_empty: np.ndarray,
    alpha: float,
    routing: Literal["trust_h", "trust_g"],
) -> np.ndarray:
    """Normalized spot-checked mixture ``q = α·q_spot + (1-α)·q_count`` over units.

    ``q_count ∝ g`` over units routed to counting (predicted populated, ``g>0``);
    ``q_spot`` is uniform over the rest (``g==0`` or predicted empty).  Every unit
    keeps positive probability for any ``α>0``, so the estimator stays unbiased
    no matter how wrong ``h`` is -- replacing ``min_count_per_tile`` with an
    interpretable spot-check budget.  ``routing`` resolves the one ambiguous cell
    (``h`` empty but ``g>0``): ``trust_h`` spot-checks it; ``trust_g`` counts it,
    leaving ``h`` unused (the no-filter baseline).
    """
    g = pred_counts.astype(np.float64)
    count_set = (g > 0) & ~h_empty if routing == "trust_h" else g > 0
    q_count = np.where(count_set, g, 0.0)
    q_spot = (~count_set).astype(np.float64)

    count_sum, spot_sum = q_count.sum(), q_spot.sum()
    if count_sum == 0.0:  # nothing is trusted-populated: all budget to spot-checking
        alpha = 1.0
    elif spot_sum == 0.0:  # nothing is suspected-empty: all budget to counting
        alpha = 0.0
    if count_sum > 0.0:
        q_count /= count_sum
    if spot_sum > 0.0:
        q_spot /= spot_sum
    return alpha * q_spot + (1.0 - alpha) * q_count


def sample_warmup(
    n_total: int,
    gt_counts: Optional[np.ndarray],
    *,
    strategy: Literal["random", "stratified"],
    n_warmup: int,
    rng: np.random.Generator,
) -> List[int]:
    """Pick warmup tile indices.

    Args:
        n_total: Total pool size.
        gt_counts: Per-tile ground-truth counts, required for ``stratified``;
            ignored for ``random``.
        strategy: ``"random"`` (uniform draw of ``n_warmup`` tiles) or
            ``"stratified"`` (guarantee at least ``n_warmup // 2`` non-empty
            tiles, rest uniform).
        n_warmup: Number of warmup tiles requested; capped at ``n_total``.
        rng: numpy random generator (advanced in-place).

    Returns:
        Warmup tile indices as List[int].
    """
    # No warmup
    if n_warmup <= 0:
        return []

    # Cap warmup size at dataset size
    n = min(int(n_warmup), int(n_total))

    if strategy == "random":
        return [int(i) for i in rng.choice(n_total, size=n, replace=False)]

    if strategy == "stratified":
        if gt_counts is None:
            raise ValueError("stratified warmup requires gt_counts")
        nonempty = np.where(np.asarray(gt_counts) > 0)[0]

        # At least half the warmup set has positive examples
        n_positive = min(n // 2, int(nonempty.size))
        n_remaining = n - n_positive

        positives = (
            rng.choice(nonempty, size=n_positive, replace=False).tolist()
            if n_positive > 0
            else []
        )
        chosen = set(positives)
        remaining_pool = np.fromiter(
            (i for i in range(n_total) if i not in chosen),
            dtype=np.int64,
        )
        rest = (
            rng.choice(
                remaining_pool,
                size=min(n_remaining, remaining_pool.size),
                replace=False,
            ).tolist()
            if n_remaining > 0 and remaining_pool.size > 0
            else []
        )
        return [int(i) for i in positives + rest]

    raise ValueError(f"Unknown warmup strategy {strategy!r}")
