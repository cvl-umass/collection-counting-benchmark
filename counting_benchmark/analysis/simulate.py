"""Fixed-model-trick simulator: replay a cached model path under fresh sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.empty_filter import fit_h
from experiments.experiment_recorder import ExperimentRecorder
from experiments.sampling import sample_unit

from .hfilter import aligned_features
from .trajectory import IS_NAME, Trajectory


def zeroshot_features(store_dir, features_root="data/dinov3") -> np.ndarray:
    """DINOv3 features aligned to a prediction store's manifest (spotcheck sims)."""
    store = Path(store_dir)
    cfg = ExperimentRecorder.load_config(store)
    manifest = ExperimentRecorder.load_manifest(store).sort_values("tile_index")
    return aligned_features(features_root, cfg["class_key"],
                            int(cfg["max_whole_image_side"]), int(cfg["tile_size"]),
                            manifest.to_dict("records"))


def simulate_zeroshot(
    store_dir, seed: int, *, sampler: str, max_frac: float = 0.5, min_count: float = 1.0,
    feats: np.ndarray = None, refit_h_every: int = 25, alpha: float = 0.05,
    routing: str = "trust_h", h_fpr_tol: float = 0.01, h_min_per_class: int = 2,
    h_calib_folds: int = 5,
) -> Trajectory:
    """Exact zero-shot trajectory from a prediction store -- no base run needed.

    A zero-shot model is never retrained and zero-shot runs have no warmup, so
    the process a real run would follow -- draw from ``q`` built on the single
    frozen ``g`` over the remaining pool, reveal the (fully annotated) truth --
    is replayed exactly here: unlike ``simulate_trajectory`` there is **no**
    fixed-model approximation.  ``sampler``:

    - ``random`` / ``importance`` -- plain replay (AIS == DIS for a frozen ``g``;
      the importance cell is labeled ``dis``).
    - ``spotcheck`` -- the empty-filter probe ``h`` is refit LIVE on the
      trajectory's own revealed labels every ``refit_h_every`` steps, via the
      same ``fit_h`` the live loop uses (run.py's zero-shot refit-h decoupling),
      so the replay stays exact-in-process.  Requires ``feats`` (see
      :func:`zeroshot_features`).  The probe is platform-sensitive -- generate
      ensembles on HPC (the sklearn fit diverges on macOS ARM).  Labeled
      ``spotdis`` (frozen ``g``, adaptive ``h`` -- the zero-shot spot cell).

    ``max_frac``/``min_count`` and the ``h`` hyperparameters default to the
    same values the fine-tuned trajectories use.
    """
    if sampler not in ("random", "importance", "spotcheck"):
        raise ValueError(f"unknown sampler {sampler!r}")
    spot = sampler == "spotcheck"
    if spot and feats is None:
        raise ValueError("spotcheck needs feats -- load via zeroshot_features(store_dir)")
    store = Path(store_dir)
    cfg = ExperimentRecorder.load_config(store)
    g, _ = ExperimentRecorder.load_snapshots(store)
    g = np.asarray(g, np.float64)
    manifest = ExperimentRecorder.load_manifest(store).sort_values("tile_index")
    gt = manifest["gt_count"].to_numpy(np.float64)
    N = int(cfg["N"])
    rng = np.random.default_rng(seed)

    def refit(revealed):
        _, h_empty = fit_h(feats, revealed, gt, fpr_tol=h_fpr_tol,
                           min_per_class=h_min_per_class, calib_folds=h_calib_folds,
                           seed=seed)
        return h_empty

    revealed: set = set()
    h_empty = refit(revealed) if spot else None  # snapshot-0 fit on the empty set
    f_steps, q_steps, main_idx = [], [], []
    t = 0
    while True:
        t += 1
        extra = dict(h_empty=h_empty, alpha=alpha, routing=routing) if spot else {}
        s_t, q = sample_unit(revealed, N, g[0], sampler=sampler,
                             min_count_per_tile=min_count, rng=rng, **extra)
        revealed.add(s_t)
        f_steps.append(gt[s_t])
        q_steps.append(q)
        main_idx.append(s_t)
        will_stop = len(revealed) / N >= max_frac or len(revealed) >= N
        if spot and refit_h_every > 0 and t % refit_h_every == 0 and not will_stop:
            h_empty = refit(revealed)  # same order as run.py: draw, reveal, refit
        if will_stop:
            break

    frozen = sampler != "random"  # zero-shot g is frozen; importance -> dis, spot -> spotdis
    T = len(f_steps)
    meta = {
        "run_dir": str(store),
        "sampler": sampler,
        "class_key": cfg.get("class_key"),
        "model": cfg.get("model"),
        "seed": seed,
        "frozen": frozen,
        "is_name": IS_NAME[(sampler, frozen)],
        "source": "sim",
        "base_run": str(store),
        "base_seed": None,
    }
    return Trajectory(
        f_steps=np.asarray(f_steps, np.float64),
        q_steps=np.asarray(q_steps, np.float64),
        warmup_sum=0.0,
        warmup_f=np.zeros(0),
        N=N,
        w=0,
        F_true=float(cfg["F_true"]),
        snapshot_id=np.zeros(T, np.int64),
        g=g,
        warmup_idx=np.zeros(0, np.int64),
        main_idx=np.asarray(main_idx, np.int64),
        meta=meta,
    )


def simulate_trajectory(run_dir, seed: int, *, freeze_g: bool = False,
                        freeze_snap: int | None = None) -> Trajectory:
    """``freeze_g`` pins g to snapshot 0; ``freeze_snap`` pins it to an explicit
    snapshot (negative = from the end, so ``-1`` is the final retrained model) for
    *both* the proposal and the estimators -- needed for directed samplers, whose
    draws depend on which snapshot's g is frozen (unlike uniform-random draws)."""
    run_dir = Path(run_dir)
    cfg = ExperimentRecorder.load_config(run_dir)
    sampler = cfg["sampler"]

    warmup = ExperimentRecorder.load_warmup(run_dir)
    g, snap_steps = ExperimentRecorder.load_snapshots(run_dir)
    manifest = ExperimentRecorder.load_manifest(run_dir).sort_values("tile_index")
    gt = manifest["gt_count"].to_numpy(np.float64)  # indexed by tile_index
    g = np.asarray(g, np.float64)
    snap_steps = np.asarray(snap_steps)

    # Spotcheck reuses the frozen per-snapshot empty filter alongside g.
    h_empty = spot_kw = None
    if sampler == "spotcheck":
        h_empty = np.asarray(ExperimentRecorder.load_h_empty(run_dir), bool)
        spot_kw = dict(alpha=float(cfg["alpha"]), routing=cfg["routing"])

    N = int(cfg["N"])
    warmup_idx = warmup["s_t"].to_numpy(np.int64)
    w = warmup_idx.size
    min_count = float(cfg["min_count_per_tile"])
    max_frac = cfg["max_fraction_labeled"]
    rng = np.random.default_rng(seed)

    frozen = freeze_g or (freeze_snap is not None)
    fixed_snap = (freeze_snap if (freeze_snap is None or freeze_snap >= 0)
                  else g.shape[0] + freeze_snap)   # negative index from the end

    revealed = {int(i) for i in warmup_idx}
    f_steps, q_steps, main_idx, snap_ids = [], [], [], []
    t = 0
    while True:
        t += 1
        if fixed_snap is not None:
            snap = fixed_snap
        elif freeze_g:
            snap = 0
        else:
            snap = int(np.searchsorted(snap_steps, t, side="left") - 1)
        snap = min(max(snap, 0), g.shape[0] - 1)
        extra = dict(h_empty=h_empty[snap], **spot_kw) if sampler == "spotcheck" else {}
        s_t, q = sample_unit(
            revealed, N, g[snap], sampler=sampler,
            min_count_per_tile=min_count, rng=rng, **extra,
        )
        revealed.add(s_t)
        f_steps.append(gt[s_t])
        q_steps.append(q)
        main_idx.append(s_t)
        snap_ids.append(snap)
        if (max_frac is not None and len(revealed) / N >= max_frac) or len(revealed) >= N:
            break

    meta = {
        "run_dir": str(run_dir),
        "sampler": sampler,
        "class_key": cfg.get("class_key"),
        "model": cfg.get("model"),
        "seed": seed,
        "frozen": frozen,
        "is_name": IS_NAME[(sampler, frozen)],
        "source": "sim",
        "base_run": str(run_dir),
        "base_seed": cfg.get("seed"),
    }
    return Trajectory(
        f_steps=np.asarray(f_steps, np.float64),
        q_steps=np.asarray(q_steps, np.float64),
        warmup_sum=float(warmup["f_st"].sum()),
        warmup_f=warmup["f_st"].to_numpy(np.float64),
        N=N,
        w=w,
        F_true=float(cfg["F_true"]),
        snapshot_id=np.asarray(snap_ids, np.int64),
        g=g,
        warmup_idx=warmup_idx,
        main_idx=np.asarray(main_idx, np.int64),
        meta=meta,
    )
