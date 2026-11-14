"""The DINOv3 empty filter ``h`` for the ``spotcheck`` sampler."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from data.feature_cache import cache_dir_for, features_for_dataset


def load_features(cache_root: str, ds, class_key: str) -> np.ndarray:
    """L2-normalised per-unit DINOv3 features aligned to ``ds`` (``out[s_t]`` -> unit ``s_t``).

    Resolves the per-class cache by the same ``(max_whole_image_side, tile_size)``
    layout the extractor wrote, then aligns it to ``ds`` by the pool-invariant
    ``(image_id, x0, y0)`` key, so any image pool maps correctly.
    """
    cdir = cache_dir_for(
        Path(cache_root), class_key, ds.max_whole_image_side, ds.tile_size
    )
    feats, _ = features_for_dataset(cdir, ds)  # strict: every unit must be cached
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return (feats / np.maximum(norms, 1e-10)).astype(np.float32)


def fit_h(
    feats: np.ndarray,
    revealed: Iterable[int],
    gt_counts: np.ndarray,
    *,
    fpr_tol: float,
    min_per_class: int,
    calib_folds: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Refit the probe on ``D_t`` and score the whole pool.

    Returns ``(p_empty, h_empty)`` over all units.  Until ``D_t`` holds
    ``min_per_class`` of each class the probe is untrainable, and we return an
    all-False ``h_empty`` -- which makes the spotcheck mixture fall back to the
    no-filter (``trust_g``) proposal while keeping every unit covered.
    """
    revealed = np.fromiter(revealed, dtype=np.int64)
    empty = gt_counts == 0
    n_pos = int(empty[revealed].sum()) # Number of truly empty units
    n_neg = int(revealed.size - n_pos) # Number of truly populated units
    if min(n_pos, n_neg) < min_per_class:
        zeros = np.zeros(len(gt_counts), dtype=np.float32)
        return zeros, np.zeros(len(gt_counts), dtype=bool)

    feats_std = _standardise(feats, revealed)
    p_empty = _fit_predict(feats_std, revealed, np.arange(len(gt_counts)), empty, seed) 

    # Cut calibrated on out-of-fold D_t scores (Exp-5 rule): the (1 - fpr_tol)
    # quantile of the populated D_t scores.  Out-of-fold so the probe's
    # over-confident in-sample scores do not collapse the threshold.
    p_oof = _oof_scores(feats_std, revealed, empty, calib_folds, seed)
    thr = _operating_threshold(p_oof, empty[revealed], fpr_tol)
    return p_empty, p_empty >= thr


def _standardise(feats: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Z-score every row using the mean/std of the revealed (``D_t``) rows only."""
    mu = feats[train_idx].mean(axis=0, keepdims=True)
    sd = np.maximum(feats[train_idx].std(axis=0, keepdims=True), 1e-6)
    return ((feats - mu) / sd).astype(np.float32)


def _fit_predict(
    feats_std: np.ndarray,
    fit_idx: np.ndarray,
    query_idx: np.ndarray,
    empty: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Train a balanced logistic probe on ``fit_idx``; return p_empty over ``query_idx``."""
    clf = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, random_state=seed
    )
    clf.fit(feats_std[fit_idx], empty[fit_idx].astype(np.int64))
    pos = int(np.where(clf.classes_ == 1)[0][0])  # column for empty (positive)
    return clf.predict_proba(feats_std[query_idx])[:, pos].astype(np.float32)


def _oof_scores(
    feats_std: np.ndarray,
    revealed: np.ndarray,
    empty: np.ndarray,
    calib_folds: int,
    seed: int,
) -> np.ndarray:
    """Out-of-fold p_empty over ``D_t`` (aligned to ``revealed`` order)."""

    y = empty[revealed].astype(np.int64)
    n_folds = max(2, min(calib_folds, int(y.sum()), int(len(y) - y.sum())))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    p = np.empty(revealed.size, dtype=np.float32)
    for tr, va in skf.split(revealed, y):
        p[va] = _fit_predict(feats_std, revealed[tr], revealed[va], empty, seed)
    return p


def _operating_threshold(
    p_train: np.ndarray, train_empty: np.ndarray, fpr_tol: float
) -> float:
    """The (1 - fpr_tol) quantile of the populated ``D_t`` scores (ties excluded)."""
    pop_sorted = np.sort(p_train[~train_empty])  # ascending, populated D_t scores
    n_pop = pop_sorted.size
    k_allow = int(np.floor(fpr_tol * n_pop))  # populated false positives allowed
    if k_allow <= 0:
        return float(np.nextafter(pop_sorted[-1], np.inf))  # above every populated score
    return float(np.nextafter(pop_sorted[n_pop - 1 - k_allow], np.inf))
