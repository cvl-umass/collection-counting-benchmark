"""Persistent I/O for a single run directory.

The loop records raw signals only; estimators are derived post-hoc.
On-disk formats are documented in ``docs/run_artifacts.md``.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Manifest origin: CountingClassTiledDataset
_MANIFEST_FIELDS = [
    "tile_index",
    "img_idx",
    "image_id",
    "image_path",
    "x0",
    "y0",
    "pw",
    "ph",
    "is_tiled",
    "gt_count",
]
_WARMUP_FIELDS = ["s_t", "f_st", "g_st"]
_STEPS_FIELDS = ["t", "s_t", "f_st", "q_t_st", "snapshot_id", "retrain_step"]

_FLUSH_EVERY = 100  # steps.csv flush interval (rows)
_CHECKPOINT_GLOB = "checkpoint_t*.pt"


@dataclass
class ResumeState:
    """Everything the driver needs to resume a run from disk."""

    model_state: Dict[str, Any]
    rng_numpy: Dict[str, Any]
    rng_torch: torch.Tensor
    rng_torch_cuda: Optional[torch.Tensor]
    last_t: int
    revealed: List[int]
    last_pred_counts: np.ndarray
    active_snap_id: int
    # Empty-filter decision as of the active snapshot (spotcheck runs only).
    h_empty: Optional[np.ndarray] = None
    # Active h snapshot id; decoupled from active_snap_id for frozen-model runs.
    active_h_id: Optional[int] = None


class ExperimentRecorder:
    """Writer/reader for a single run directory."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._steps_path = self.run_dir / "steps.csv"
        self._snapshots_path = self.run_dir / "snapshots.npz"
        self._detections_path = self.run_dir / "detections.npz"
        self._warmup_path = self.run_dir / "warmup.csv"
        self._manifest_path = self.run_dir / "manifest.csv"
        self._config_path = self.run_dir / "config.json"

        # In-memory snapshot accumulator.  Loaded eagerly from disk if present
        # so subsequent appends keep the prior snapshots.
        self._snapshots: List[np.ndarray] = []
        self._snapshot_steps: List[int] = []
        # Empty-filter scores p_empty with their own fit-steps (spotcheck runs).
        # The h stream is decoupled from g in snapshots.npz so frozen-model runs
        # refit h without re-storing g.  Empty for non-spotcheck runs.
        self._h_scores: List[np.ndarray] = []
        self._h_steps: List[int] = []
        # Boolean h_empty parallel to _h_scores (the D_t-calibrated cut).  Empty
        # for runs written before this was recorded (those are backfilled from
        # analysis.hfilter).  _flush only writes it when aligned.
        self._h_empty: List[np.ndarray] = []
        if self._snapshots_path.exists():
            arr = np.load(self._snapshots_path)
            g = np.asarray(arr["g"])
            steps = np.asarray(arr["snapshot_steps"])
            self._snapshots = [g[i].copy() for i in range(g.shape[0])]
            self._snapshot_steps = [int(s) for s in steps]
            if "h" in arr.files:
                h = np.asarray(arr["h"])
                self._h_scores = [h[i].copy() for i in range(h.shape[0])]
                if "h_steps" in arr.files:
                    self._h_steps = [int(s) for s in np.asarray(arr["h_steps"])]
                else:
                    self._h_steps = list(self._snapshot_steps)
                if "h_empty" in arr.files:
                    he = np.asarray(arr["h_empty"])
                    self._h_empty = [he[i].copy() for i in range(he.shape[0])]

        # In-memory per-box detection accumulator, long format: one row per
        # detected box, keyed by (snapshot_id, tile_index).  Loaded eagerly so
        # appends and resume-trimming see the prior rows.  See append_detections.
        self._det_snapshot_id: np.ndarray = np.zeros(0, dtype=np.int32)
        self._det_tile_index: np.ndarray = np.zeros(0, dtype=np.int32)
        self._det_score: np.ndarray = np.zeros(0, dtype=np.float32)
        if self._detections_path.exists():
            arr = np.load(self._detections_path)
            self._det_snapshot_id = np.asarray(arr["snapshot_id"], dtype=np.int32)
            self._det_tile_index = np.asarray(arr["tile_index"], dtype=np.int32)
            self._det_score = np.asarray(arr["score"], dtype=np.float32)

        # steps.csv writer is opened lazily on first append_step so a recorder
        # that never writes any rows leaves no empty CSV behind.
        self._steps_file = None
        self._steps_writer: Optional[csv.DictWriter] = None
        self._steps_rows_since_flush = 0

    # ------------------------------------------------------------------
    # Run-directory state queries
    # ------------------------------------------------------------------

    def has_checkpoint(self) -> bool:
        return self._latest_checkpoint() is not None

    def has_step_data(self) -> bool:
        """True iff any prior run data is on disk (used by the no-clobber guard)."""
        return (
            self._steps_path.exists()
            or self._snapshots_path.exists()
            or self._warmup_path.exists()
        )

    def _latest_checkpoint(self) -> Optional[Path]:
        ckpts = sorted(self.run_dir.glob(_CHECKPOINT_GLOB))
        return ckpts[-1] if ckpts else None

    # ------------------------------------------------------------------
    # One-shot writes
    # ------------------------------------------------------------------

    def write_config(self, cfg: Dict[str, Any]) -> None:
        # Creates new file, or overwrites existing file.
        self._config_path.write_text(
            json.dumps(cfg, indent=2, default=str),
            encoding="utf-8",
        )

    def write_manifest(self, rows: List[Dict[str, Any]]) -> None:
        self._atomic_csv(
            self._manifest_path, _MANIFEST_FIELDS, rows, coerce_bool={"is_tiled"}
        )

    def write_warmup(self, rows: List[Dict[str, Any]]) -> None:
        self._atomic_csv(self._warmup_path, _WARMUP_FIELDS, rows)

    # ------------------------------------------------------------------
    # Streaming writes
    # ------------------------------------------------------------------

    def append_step(
        self,
        *,
        t: int,
        s_t: int,
        f_st: int,
        q_t_st: float,
        snapshot_id: int,
        retrain_step: bool,
    ) -> None:
        if self._steps_writer is None:
            self._open_steps_writer()
        self._steps_writer.writerow(
            {
                "t": int(t),
                "s_t": int(s_t),
                "f_st": int(f_st),
                "q_t_st": float(q_t_st),
                "snapshot_id": int(snapshot_id),
                "retrain_step": 1 if retrain_step else 0,
            }
        )
        self._steps_rows_since_flush += 1
        if self._steps_rows_since_flush >= _FLUSH_EVERY:
            self._steps_file.flush()
            self._steps_rows_since_flush = 0

    def append_snapshot(
        self,
        g: np.ndarray,
        step: int,
        h: Optional[np.ndarray] = None,
        h_empty: Optional[np.ndarray] = None,
    ) -> int:
        """Add a model-prediction snapshot; returns the new snapshot_id.

        ``h`` is the matching per-unit empty-filter score ``p_empty`` and
        ``h_empty`` its boolean D_t-calibrated cut (spotcheck runs); pass them
        consistently or never, so the h streams stay aligned with the snapshots.
        Rewrites ``snapshots.npz`` atomically on every call so a crash mid-run
        always leaves at least the prior snapshots durable on disk.
        """
        snap_id = len(self._snapshots)
        self._snapshots.append(np.asarray(g, dtype=np.float32).copy())
        self._snapshot_steps.append(int(step))
        if h is not None:
            self._h_scores.append(np.asarray(h, dtype=np.float32).copy())
            self._h_steps.append(int(step))
            if h_empty is not None:
                self._h_empty.append(np.asarray(h_empty, dtype=bool).copy())
        self._flush_snapshots()
        return snap_id

    def append_h(
        self, h: np.ndarray, step: int, h_empty: Optional[np.ndarray] = None
    ) -> int:
        """Append an empty-filter snapshot without a new model prediction.

        For frozen-model runs (zero-shot) that refit h on the growing revealed
        set while g stays fixed: stores only the h row (``p_empty``), its boolean
        cut ``h_empty``, and the step, so g is never re-stored.  Returns the new h
        snapshot id.
        """
        h_id = len(self._h_scores)
        self._h_scores.append(np.asarray(h, dtype=np.float32).copy())
        self._h_steps.append(int(step))
        if h_empty is not None:
            self._h_empty.append(np.asarray(h_empty, dtype=bool).copy())
        self._flush_snapshots()
        return h_id

    def append_detections(
        self, snapshot_id: int, per_unit_scores: List[np.ndarray]
    ) -> None:
        """Append a snapshot's per-box confidences (raw signal for soft detection).

        ``per_unit_scores[i]`` is the 1-D array of detection scores for unit
        ``i`` (its ``tile_index``); empty units contribute no rows.  Stored in
        long format -- ``snapshot_id``, ``tile_index``, ``score`` -- so the
        ragged per-unit box counts need no padding.  Pass the ``snapshot_id``
        returned by the matching :meth:`append_snapshot`.  Rewrites
        ``detections.npz`` atomically, like snapshots, so a crash leaves the
        prior rows durable.
        """
        sids: List[np.ndarray] = []
        tiles: List[np.ndarray] = []
        scores: List[np.ndarray] = []
        for tile_idx, sc in enumerate(per_unit_scores):
            sc = np.asarray(sc, dtype=np.float32).ravel()
            if sc.size == 0:
                continue
            sids.append(np.full(sc.size, int(snapshot_id), dtype=np.int32))
            tiles.append(np.full(sc.size, int(tile_idx), dtype=np.int32))
            scores.append(sc)
        if scores:
            self._det_snapshot_id = np.concatenate([self._det_snapshot_id, *sids])
            self._det_tile_index = np.concatenate([self._det_tile_index, *tiles])
            self._det_score = np.concatenate([self._det_score, *scores])
            self._flush_detections()

    def save_checkpoint(
        self,
        *,
        model_state: Dict[str, Any],
        rng_states: Dict[str, Any],
        last_t: int,
        active_snap_id: int,
        h_empty: Optional[np.ndarray] = None,
        active_h_id: Optional[int] = None,
    ) -> None:
        """Write the checkpoint file, then delete any prior checkpoint.

        ``h_empty`` (spotcheck runs) is persisted so resume drives sampling with
        the exact filter decision from the active snapshot, not a recompute.
        ``active_h_id`` defaults to ``active_snap_id`` (h and g refit together);
        frozen-model runs pass the h stream's own id.

        Durability invariant: steps.csv is flushed + fsync'd up to ``last_t``
        *before* the checkpoint is written.  Otherwise a crash between a buffered
        step flush (every ``_FLUSH_EVERY`` rows) and this checkpoint would leave
        ``last_t`` ahead of the durable steps; resume reconstructs ``revealed``
        from steps.csv and restarts at ``last_t + 1``, permanently gapping the
        unflushed rows.
        """
        if self._steps_file is not None:
            self._steps_file.flush()
            os.fsync(self._steps_file.fileno())
            self._steps_rows_since_flush = 0

        name = f"checkpoint_t{int(last_t):07d}.pt"
        path = self.run_dir / name
        tmp = path.with_suffix(".pt.tmp")
        payload = {
            "model_state": model_state,
            "rng_numpy": rng_states["numpy"],
            "rng_torch": rng_states["torch"],
            "rng_torch_cuda": rng_states.get("torch_cuda"),
            "last_t": int(last_t),
            "active_snap_id": int(active_snap_id),
            "active_h_id": int(active_snap_id if active_h_id is None else active_h_id),
            "h_empty": h_empty,
        }
        torch.save(payload, tmp)
        os.replace(tmp, path)
        for prior in self.run_dir.glob(_CHECKPOINT_GLOB):
            if prior != path:
                prior.unlink()

    def finalize(self) -> None:
        """Flush any buffered step rows; snapshots are already on disk."""
        if self._steps_file is not None:
            self._steps_file.flush()
            self._steps_file.close()
            self._steps_file = None
            self._steps_writer = None
            self._steps_rows_since_flush = 0

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def load_resume_state(self) -> ResumeState:
        """Restore everything the driver needs to continue a run.

        Detects and repairs two crash-window inconsistencies:

        - extra orphan snapshot beyond ``active_snap_id`` (crash after
          ``append_snapshot`` but before ``save_checkpoint``): trimmed and
          ``snapshots.npz`` rewritten.
        - extra step rows beyond ``last_t`` (crash after ``append_step`` but
          before the matching ``save_checkpoint``): truncated and
          ``steps.csv`` rewritten.
        """
        ckpt_path = self._latest_checkpoint()
        if ckpt_path is None:
            raise RuntimeError(f"no checkpoint to resume from in {self.run_dir}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        last_t = int(ckpt["last_t"])
        active_snap_id = int(ckpt["active_snap_id"])
        active_h_id = ckpt.get("active_h_id")
        active_h_id = active_snap_id if active_h_id is None else int(active_h_id)

        # Trim orphan snapshots beyond the checkpoint's view.  g and h are trimmed
        # by their own active ids: frozen-model runs grow the h stream (refits)
        # without ever appending a new g.
        trimmed = False
        if active_snap_id + 1 < len(self._snapshots):
            self._snapshots = self._snapshots[: active_snap_id + 1]
            self._snapshot_steps = self._snapshot_steps[: active_snap_id + 1]
            trimmed = True
        elif active_snap_id + 1 > len(self._snapshots):
            raise RuntimeError(
                f"checkpoint references snap_id={active_snap_id} but only "
                f"{len(self._snapshots)} snapshots are on disk"
            )
        if self._h_scores and active_h_id + 1 < len(self._h_scores):
            self._h_scores = self._h_scores[: active_h_id + 1]
            self._h_steps = self._h_steps[: active_h_id + 1]
            self._h_empty = self._h_empty[: active_h_id + 1]
            trimmed = True
        if trimmed:
            self._flush_snapshots()

        # Trim orphan detection rows from the same crash window (an
        # append_detections that outran its checkpoint).
        if self._det_snapshot_id.size and self._det_snapshot_id.max() > active_snap_id:
            keep = self._det_snapshot_id <= active_snap_id
            self._det_snapshot_id = self._det_snapshot_id[keep]
            self._det_tile_index = self._det_tile_index[keep]
            self._det_score = self._det_score[keep]
            self._flush_detections()

        # Truncate orphan step rows beyond last_t.
        self._truncate_steps_to(last_t)

        revealed = self._reconstruct_revealed(last_t)
        last_pred_counts = self._snapshots[active_snap_id].copy()

        return ResumeState(
            model_state=ckpt["model_state"],
            rng_numpy=ckpt["rng_numpy"],
            rng_torch=ckpt["rng_torch"],
            rng_torch_cuda=ckpt.get("rng_torch_cuda"),
            last_t=last_t,
            revealed=revealed,
            last_pred_counts=last_pred_counts,
            active_snap_id=active_snap_id,
            h_empty=ckpt.get("h_empty"),
            active_h_id=active_h_id,
        )

    # ------------------------------------------------------------------
    # Destructive: wipe
    # ------------------------------------------------------------------

    def wipe(self) -> List[str]:
        """Delete every known artifact in the run dir.  Returns deleted names."""
        deleted: List[str] = []
        for name in (
            "config.json",
            "manifest.csv",
            "warmup.csv",
            "steps.csv",
            "snapshots.npz",
            "detections.npz",
        ):
            p = self.run_dir / name
            if p.exists():
                p.unlink()
                deleted.append(name)
        for p in list(self.run_dir.glob(_CHECKPOINT_GLOB)):
            p.unlink()
            deleted.append(p.name)
        for p in list(self.run_dir.glob("*.tmp")):
            p.unlink()
            deleted.append(p.name)

        self._snapshots.clear()
        self._snapshot_steps.clear()
        self._h_scores.clear()
        self._h_steps.clear()
        self._h_empty.clear()
        self._det_snapshot_id = np.zeros(0, dtype=np.int32)
        self._det_tile_index = np.zeros(0, dtype=np.int32)
        self._det_score = np.zeros(0, dtype=np.float32)
        if self._steps_file is not None:
            try:
                self._steps_file.close()
            except Exception:
                pass
            self._steps_file = None
            self._steps_writer = None
            self._steps_rows_since_flush = 0
        return deleted

    # ------------------------------------------------------------------
    # Static readers (analysis-side; never used by the main loop)
    # ------------------------------------------------------------------

    @staticmethod
    def load_steps(run_dir: Path) -> pd.DataFrame:
        df = pd.read_csv(Path(run_dir) / "steps.csv")
        df["retrain_step"] = df["retrain_step"].astype(bool)
        return df

    @staticmethod
    def load_snapshots(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.load(Path(run_dir) / "snapshots.npz")
        return np.asarray(arr["g"]), np.asarray(arr["snapshot_steps"])

    @staticmethod
    def load_h_scores(run_dir: Path) -> Optional[np.ndarray]:
        """Empty-filter scores ``p_empty`` (``(n_h, N)``), or None.

        Present only for spotcheck runs.  ``n_h`` equals ``load_snapshots``'s
        ``n_snap`` for trainable runs (h always refits alongside g); for
        frozen-model (zero-shot) runs using ``--refit-h-every``, ``n_h`` can
        exceed ``n_snap`` since h keeps refitting on its own cadence while g
        stays pinned to the snapshot-0 prediction. Pair with
        :meth:`load_h_steps` to align a row to a given main-loop step ``t``:
        the active row is the one with the largest step ``<= t``.
        """
        arr = np.load(Path(run_dir) / "snapshots.npz")
        return np.asarray(arr["h"]) if "h" in arr.files else None

    @staticmethod
    def load_h_empty(run_dir: Path) -> Optional[np.ndarray]:
        """Boolean D_t-calibrated cut for each :meth:`load_h_scores` row, or None.

        Absent on runs written before h_empty was recorded (and not yet
        backfilled from ``analysis.hfilter``).
        """
        arr = np.load(Path(run_dir) / "snapshots.npz")
        return np.asarray(arr["h_empty"]) if "h_empty" in arr.files else None

    @staticmethod
    def load_h_steps(run_dir: Path) -> Optional[np.ndarray]:
        """Fit-step for each row of :meth:`load_h_scores`, or None.

        Runs that refit h and g together have no ``h_steps`` key in
        ``snapshots.npz``; for them ``snapshot_steps`` gives the same steps.
        """
        arr = np.load(Path(run_dir) / "snapshots.npz")
        if "h_empty" not in arr.files:
            return None
        steps_key = "h_steps" if "h_steps" in arr.files else "snapshot_steps"
        return np.asarray(arr[steps_key])

    @staticmethod
    def load_detections(run_dir: Path) -> pd.DataFrame:
        """Per-box confidences in long format: ``snapshot_id, tile_index, score``.

        Empty (no boxes ever recorded, or a pre-detections run) yields an
        empty frame with those columns.  Recover a soft ``g`` for snapshot
        ``s`` by filtering ``snapshot_id == s`` and aggregating ``score`` over
        ``tile_index`` (units with no rows are zero).
        """
        cols = ["snapshot_id", "tile_index", "score"]
        path = Path(run_dir) / "detections.npz"
        if not path.exists():
            return pd.DataFrame({c: [] for c in cols})
        arr = np.load(path)
        return pd.DataFrame({c: np.asarray(arr[c]) for c in cols})

    @staticmethod
    def manifest_path(run_dir: Path) -> Path:
        """The run's ``manifest.csv``, or the task-level one it shares.

        A live run writes its own copy.  Every run of a task holds the identical
        file (the pool is frozen before any model runs), so the released data
        stores one copy per task directory instead of one per run; fall back to
        it when the per-run copy is absent.
        """
        own = Path(run_dir) / "manifest.csv"
        return own if own.exists() else Path(run_dir).parent / "manifest.csv"

    @staticmethod
    def load_manifest(run_dir: Path) -> pd.DataFrame:
        df = pd.read_csv(ExperimentRecorder.manifest_path(run_dir))
        df["is_tiled"] = df["is_tiled"].astype(bool)
        return df

    @staticmethod
    def load_warmup(run_dir: Path) -> pd.DataFrame:
        return pd.read_csv(Path(run_dir) / "warmup.csv")

    @staticmethod
    def load_config(run_dir: Path) -> Dict[str, Any]:
        return json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _open_steps_writer(self) -> None:
        write_header = not self._steps_path.exists()
        self._steps_file = self._steps_path.open("a", newline="", encoding="utf-8")
        self._steps_writer = csv.DictWriter(self._steps_file, fieldnames=_STEPS_FIELDS)
        if write_header:
            self._steps_writer.writeheader()
            self._steps_file.flush()

    def _flush_snapshots(self) -> None:
        if not self._snapshots:
            return
        g = np.stack(self._snapshots, axis=0)
        steps = np.asarray(self._snapshot_steps, dtype=np.int64)
        arrays = {"g": g, "snapshot_steps": steps}
        if self._h_scores:
            arrays["h"] = np.stack(self._h_scores, axis=0)
            arrays["h_steps"] = np.asarray(self._h_steps, dtype=np.int64)
            # Only when aligned: a run written before h_empty was recorded and
            # then resumed has _h_scores populated but _h_empty empty/short.
            if len(self._h_empty) == len(self._h_scores):
                arrays["h_empty"] = np.stack(self._h_empty, axis=0)
        # np.savez auto-appends ".npz" unless the path already ends in ".npz",
        # so the tmp name must keep that suffix at the end.
        tmp = self._snapshots_path.parent / f"{self._snapshots_path.stem}.tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, self._snapshots_path)

    def _flush_detections(self) -> None:
        # np.savez auto-appends ".npz" unless the path already ends in it, so
        # the tmp name must keep that suffix at the end (see _flush_snapshots).
        tmp = self._detections_path.parent / f"{self._detections_path.stem}.tmp.npz"
        np.savez(
            tmp,
            snapshot_id=self._det_snapshot_id,
            tile_index=self._det_tile_index,
            score=self._det_score,
        )
        os.replace(tmp, self._detections_path)

    def _truncate_steps_to(self, last_t: int) -> None:
        if not self._steps_path.exists():
            return
        df = pd.read_csv(self._steps_path)
        if df.empty or int(df["t"].max()) <= last_t:
            return
        keep = df[df["t"] <= last_t]
        tmp = self._steps_path.with_suffix(".csv.tmp")
        keep.to_csv(tmp, index=False)
        os.replace(tmp, self._steps_path)

    def _reconstruct_revealed(self, last_t: int) -> List[int]:
        revealed: List[int] = []
        if self._warmup_path.exists():
            wdf = pd.read_csv(self._warmup_path)
            if len(wdf) > 0:
                revealed.extend(int(s) for s in wdf["s_t"])
        if self._steps_path.exists():
            sdf = pd.read_csv(self._steps_path)
            sdf = sdf[sdf["t"] <= last_t]
            revealed.extend(int(s) for s in sdf["s_t"])
        return revealed

    @staticmethod
    def _atomic_csv(
        path: Path,
        fieldnames: List[str],
        rows: List[Dict[str, Any]],
        *,
        coerce_bool: Optional[set] = None,
    ) -> None:
        """Write a CSV atomically (tmp + os.replace) to prevent file corruption or partial writes."""
        coerce_bool = coerce_bool or set()
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                out = dict(row)
                for k in coerce_bool:
                    if k in out:
                        out[k] = int(bool(out[k]))
                w.writerow(out)
        os.replace(tmp, path)
