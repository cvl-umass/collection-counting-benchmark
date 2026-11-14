"""Reconstruct the empty filter ``h`` from a run's recorded ``D_t``."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data.feature_cache import align_features, cache_dir_for
from experiments.empty_filter import fit_h


def aligned_features(
    cache_root, class_key: str, max_whole_image_side: int, tile_size: int,
    manifest_rows: List[Dict],
) -> np.ndarray:
    """L2-normalised DINOv3 features aligned to ``manifest_rows`` (row i -> unit i)."""
    cdir = cache_dir_for(Path(cache_root), class_key, max_whole_image_side, tile_size)
    feats, _ = align_features(cdir, manifest_rows, strict=True)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return (feats / np.maximum(norms, 1e-10)).astype(np.float32)


def revealed_at(warmup_idx, steps_df: pd.DataFrame, step: int) -> set:
    """The revealed set as of fit-``step``: warmup then draws with ``t <= step``.

    Built incrementally in that order to reproduce the exact Python-set layout
    (hence the ``fit_h`` OOF fold split) the live run had.
    """
    rev: set = set()
    for s in warmup_idx:
        rev.add(int(s))
    for s in steps_df.loc[steps_df["t"] <= step, "s_t"]:
        rev.add(int(s))
    return rev


def rebuild_h(
    feats: np.ndarray, warmup_idx, steps_df: pd.DataFrame, gt: np.ndarray,
    h_steps, *, fpr_tol: float, min_per_class: int, calib_folds: int, seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(p_empty, h_empty)`` of shape ``(n_h, N)`` rebuilt for each fit-step."""
    p_rows, he_rows = [], []
    for step in h_steps:
        p, he = fit_h(
            feats, revealed_at(warmup_idx, steps_df, int(step)), gt,
            fpr_tol=fpr_tol, min_per_class=min_per_class,
            calib_folds=calib_folds, seed=seed,
        )
        p_rows.append(p)
        he_rows.append(he)
    return np.stack(p_rows, axis=0), np.stack(he_rows, axis=0)
