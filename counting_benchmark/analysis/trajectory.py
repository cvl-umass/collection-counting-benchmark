"""Load a run's artifacts into the plain arrays the estimators consume.

One ``Trajectory`` holds everything needed to run every estimator: the per-step
draws (``f_steps``, ``q_steps``), the warmup exact sum, pool sizes, and the
snapshot ``g`` matrix + per-step active snapshot for ``model_only``.  Real runs
load here; simulated ones (``simulate.py``) build the same dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from experiments.experiment_recorder import ExperimentRecorder

# (sampler, frozen-g) -> the IS estimator's name for this trajectory.  The math
# is always the AIS core; the label reflects how the trajectory was produced.
IS_NAME = {
    ("random", False): "is_uniform",
    ("random", True): "is_uniform",   # frozen-g random: proposal is still uniform
    ("importance", False): "ais",
    ("importance", True): "dis",
    ("spotcheck", False): "spotais",
    ("spotcheck", True): "spotdis",
}


@dataclass
class Trajectory:
    f_steps: np.ndarray      # (T,) f(s_t) in draw order
    q_steps: np.ndarray      # (T,) q_t(s_t)
    warmup_sum: float        # F(W)
    warmup_f: np.ndarray     # (w,) f over the warmup units, reveal order
    N: int
    w: int
    F_true: float
    snapshot_id: np.ndarray  # (T,) active snapshot per step (row into g)
    g: np.ndarray            # (n_snap, N) prediction snapshots
    warmup_idx: np.ndarray   # (w,) warmup unit indices
    main_idx: np.ndarray     # (T,) drawn unit indices
    meta: Dict[str, Any]

    @property
    def T(self) -> int:
        return self.f_steps.size

    def model_only_inputs(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(g_total_steps, g_revealed_steps)`` for ``estimators.model_only``.

        ``g_total`` = active snapshot summed over Omega; ``g_revealed`` = the same
        snapshot summed over R_t = warmup + first t draws (inclusive of s_t).
        """
        order = np.concatenate([self.warmup_idx, self.main_idx])   # reveal order
        C = np.cumsum(self.g[:, order], axis=1)                    # (n_snap, w+T)
        g_total = self.g.sum(axis=1)[self.snapshot_id]            # (T,)
        k = self.w + np.arange(1, self.T + 1)                      # |R_t| = w + t
        g_revealed = C[self.snapshot_id, k - 1]                    # (T,)
        return g_total, g_revealed

    def difference_inputs(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(g_st_steps, g_remainder_steps)`` for ``estimators.difference``.

        ``g_st`` = active snapshot at the drawn unit; ``g_remainder`` = the same
        snapshot summed over the unlabeled remainder ``Omega \\ D_t`` (``D_t``
        excludes ``s_t``, so the remainder includes it).
        """
        g_total, g_revealed = self.model_only_inputs()             # R_t inclusive of s_t
        g_st = self.g[self.snapshot_id, self.main_idx]             # g(s_t)
        g_D = g_revealed - g_st                                    # sum over D_t (exclusive)
        return g_st, g_total - g_D

    def coefficient_inputs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(g_D, gg_D, gf_D)`` for the *incremental* ratio / regression plug-in
        coefficients.

        Active-snapshot prefix sums over the labeled prefix ``D_t`` (*exclusive* of
        the fresh draw ``s_t``) of ``g``, ``g^2`` and ``g*f`` -- the sufficient
        statistics for a ``D_t``-measurable ``c_t`` (which is what keeps the
        plug-in template exactly unbiased).  Same reveal-order cumsum machinery as
        ``model_only_inputs``, with the drawn unit's own term removed.
        """
        order = np.concatenate([self.warmup_idx, self.main_idx])   # reveal order
        f_order = np.concatenate([self.warmup_f, self.f_steps])
        G = self.g[:, order]                                       # (n_snap, w+T)
        rows = self.snapshot_id
        cols = self.w + np.arange(self.T)                          # |R_t| - 1 = w + t - 1
        g_st = self.g[rows, self.main_idx]
        g_D = np.cumsum(G, axis=1)[rows, cols] - g_st
        gg_D = np.cumsum(G * G, axis=1)[rows, cols] - g_st * g_st
        gf_D = np.cumsum(G * f_order, axis=1)[rows, cols] - g_st * self.f_steps
        return g_D, gg_D, gf_D

    def canonical_inputs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray, np.ndarray]:
        r"""Sufficient statistics for the warmup-aware canonical estimators.
        
        The warmup set ``W`` is exact, so the SRSWOR expansion runs only over the
        ``t`` main draws, a uniform sample from ``Omega \ W``.  Sums are evaluated
        at each step's active snapshot (constant for a frozen ``g``):
        
          ``g_total``   ``G_t(Omega)``
          ``g_warmup``  ``G_t(W)``               (0 when ``w = 0``)
          ``g_main``    ``sum_{i<=t} g(s_i)``    -- main draws only
          ``gg_main``   ``sum_{i<=t} g(s_i)^2``
          ``gf_main``   ``sum_{i<=t} g(s_i) f(s_i)``
          ``ff_main``   ``sum_{i<=t} f(s_i)^2``
        """
        rows = self.snapshot_id                                    # (T,) active snapshot
        g_total = self.g.sum(axis=1)[rows]                         # G_t(Omega)
        g_warmup = self.g[:, self.warmup_idx].sum(axis=1)[rows]    # G_t(W); 0 when w = 0
        G_main = self.g[:, self.main_idx]                          # (n_snap, T) g over draws
        cols = np.arange(self.T)                                   # each step's main count - 1
        f = self.f_steps
        g_main = np.cumsum(G_main, axis=1)[rows, cols]
        gg_main = np.cumsum(G_main * G_main, axis=1)[rows, cols]
        gf_main = np.cumsum(G_main * f, axis=1)[rows, cols]
        ff_main = np.cumsum(f * f)                                 # snapshot-independent
        return g_total, g_warmup, g_main, gg_main, gf_main, ff_main


def load_trajectory(run_dir, *, frozen: bool = False) -> Trajectory:
    run_dir = Path(run_dir)
    cfg = ExperimentRecorder.load_config(run_dir)
    steps = ExperimentRecorder.load_steps(run_dir).sort_values("t")
    warmup = ExperimentRecorder.load_warmup(run_dir)
    g, _ = ExperimentRecorder.load_snapshots(run_dir)
    sampler = cfg["sampler"]
    meta = {
        "run_dir": str(run_dir),
        "sampler": sampler,
        "class_key": cfg.get("class_key"),
        "model": cfg.get("model"),
        "seed": cfg.get("seed"),
        "frozen": frozen,
        "is_name": IS_NAME[(sampler, frozen)],
        "source": "real",
    }
    return Trajectory(
        f_steps=steps["f_st"].to_numpy(np.float64),
        q_steps=steps["q_t_st"].to_numpy(np.float64),
        warmup_sum=float(warmup["f_st"].sum()),
        warmup_f=warmup["f_st"].to_numpy(np.float64),
        N=int(cfg["N"]),
        w=int(len(warmup)),
        F_true=float(cfg["F_true"]),
        snapshot_id=steps["snapshot_id"].to_numpy(np.int64),
        g=np.asarray(g, np.float64),
        warmup_idx=warmup["s_t"].to_numpy(np.int64),
        main_idx=steps["s_t"].to_numpy(np.int64),
        meta=meta,
    )
