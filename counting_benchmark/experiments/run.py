"""Main experiment driver: the sequential labeling loop for one (class, model, sampler, seed)."""

from __future__ import annotations

import os

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

# Make the repo root importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import CountingClassTiledDataset, collate_detection  # noqa: E402
from experiments.empty_filter import fit_h, load_features  # noqa: E402
from experiments.experiment_recorder import ExperimentRecorder  # noqa: E402
from experiments.sampling import sample_unit, sample_warmup  # noqa: E402
from models import build_model  # noqa: E402

# Root of the counting-datasets checkout.  Override with --index-root /
# --raw-root, or set COUNTING_DATA_ROOT; the fallback assumes the two
# repositories are checked out side by side.
_DATA_ROOT = os.environ.get(
    "COUNTING_DATA_ROOT",
    next((p for p in ("../counting_datasets", "../counting-datasets")
          if os.path.isdir(p)), "../counting_datasets"))


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)

    g = ap.add_argument_group("Dataset / pool")
    g.add_argument("--index-root", default=f"{_DATA_ROOT}/counting/data")
    g.add_argument("--raw-root", default=f"{_DATA_ROOT}/raw")
    g.add_argument(
        "--class-key", required=True, help="e.g. dota/small_vehicle or birds/bird"
    )
    g.add_argument(
        "--tile-sizes",
        default="outputs/suggested_tile_sizes.json",
        help="Suggestion store mapping class_key -> {tile_size, tile_threshold, "
        "...}; sets per-class tiling so the loop matches the DINOv3 cache. "
        'Pass "" to disable and rely on --tile-size/--tile-threshold.',
    )
    g.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="override the store's tile_size for this class (default: from "
        "--tile-sizes).",
    )
    g.add_argument("--max-whole-image-side", type=int, default=1536)
    g.add_argument(
        "--tile-threshold",
        type=int,
        default=None,
        help="override the store's tile_threshold (default: from --tile-sizes, "
        "else falls back to --max-whole-image-side).",
    )
    g.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help="if set, randomly subsample this many source images",
    )
    g.add_argument("--pool-sample-seed", type=int, default=0)
    g.add_argument(
        "--image-pool-list",
        default=None,
        help="path to a text file listing image filenames to include "
        "(mutually exclusive with --pool-size)",
    )

    g = ap.add_argument_group("Sampler & retraining")
    g.add_argument(
        "--sampler", choices=("random", "importance", "spotcheck"), default="random"
    )
    g.add_argument(
        "--retrain-every",
        type=int,
        default=10,
        help="0 disables main-loop retraining (DIS regime with importance)",
    )
    g.add_argument(
        "--min-count-per-tile",
        type=float,
        default=1.0,
        help="floor on per-tile weight in the importance proposal",
    )
    g.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="spot-check budget for the spotcheck proposal: mixture weight on "
        "the uniform spot proposal over suspected-empty units",
    )
    g.add_argument(
        "--routing",
        choices=("trust_h", "trust_g"),
        default="trust_h",
        help="spotcheck routing of the (h empty, g>0) cell: trust_h spot-checks "
        "it (uses the filter), trust_g counts it (filter unused = no-filter "
        "baseline)",
    )

    g = ap.add_argument_group("Empty filter (spotcheck only)")
    g.add_argument(
        "--features-root",
        default="data/dinov3",
        help="root of the cached DINOv3 features (per-class dirs "
        "<dataset>/<class>_us<MWS>_ts<TS>) for the empty-filter probe h",
    )
    g.add_argument(
        "--h-fpr-tol",
        type=float,
        default=0.01,
        help="target populated-tile false-positive rate for calibrating h's cut "
        "on D_t each snapshot",
    )
    g.add_argument(
        "--h-min-per-class",
        type=int,
        default=2,
        help="refit h only once D_t holds this many of BOTH empty/populated; "
        "until then h_empty is all-False (spotcheck falls back to trust_g)",
    )
    g.add_argument(
        "--h-calib-folds",
        type=int,
        default=5,
        help="K for out-of-fold calibration of h's cut on D_t",
    )
    g.add_argument(
        "--refit-h-every",
        type=int,
        default=0,
        help="zero-shot spotcheck only: refit h every K steps on the revealed "
        "set while g stays frozen (0 disables; h stays as fit at the t=0 warmup "
        "seed). Ignored for trainable models, which refit h at each retrain.",
    )

    g = ap.add_argument_group("Warmup")
    g.add_argument("--warmup-n", type=int, default=0)
    g.add_argument(
        "--warmup-strategy", choices=("random", "stratified"), default="random"
    )

    g = ap.add_argument_group("Model")
    g.add_argument(
        "--model",
        required=True,
        choices=(
            "frcnn_resnet50",
            "frcnn_mobilenet",
            "grounding_dino",
            "owlv2",
            "florence2",
        ),
    )
    g.add_argument(
        "--text-prompt",
        default=None,
        help="zero-shot models only; defaults to a prettified class_key",
    )
    g.add_argument(
        "--unfreeze",
        default="stride32",
        choices=("roi_heads", "stride32"),
        help="Faster R-CNN unfreeze level; ignored for zero-shot models",
    )
    g.add_argument("--backbone-lr-factor", type=float, default=0.1)
    g.add_argument("--epochs", type=float, default=5.0)
    g.add_argument("--lr", type=float, default=5e-4)
    g.add_argument("--batch-size", type=int, default=2)
    g.add_argument(
        "--eval-batch-size", type=int, default=None, help="defaults to --batch-size"
    )
    g.add_argument("--num-workers", type=int, default=0)
    g.add_argument("--score-thresh", type=float, default=0.5)
    g.add_argument(
        "--capture-score-floor",
        type=float,
        default=0.05,
        help="record every detection with confidence >= this floor to "
        "detections.npz (raw signal for post-hoc soft detection); does not "
        "affect the counts that drive sampling. 0.05 = the detector's natural "
        "emission floor.",
    )

    g = ap.add_argument_group("Loop control")
    g.add_argument("--max-steps", type=int, default=None)
    g.add_argument("--max-fraction-labeled", type=float, default=None)
    g.add_argument("--seed", type=int, default=0)

    g = ap.add_argument_group("I/O")
    g.add_argument("--out-root", default="outputs")
    g.add_argument(
        "--fresh",
        action="store_true",
        help="wipe the run dir and start over (default is auto-resume)",
    )

    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def derive_run_id(args: argparse.Namespace, *, model_trainable: bool) -> str:
    """Auto-derived run identifier; stable for matching arg sets so resume works."""
    parts: List[str] = [args.model]
    if model_trainable and args.unfreeze:
        parts.append(str(args.unfreeze).replace("/", "_"))

    # A model that can't train never retrains regardless of --retrain-every
    # (see the main loop's retrain_due gate), so the regime label must agree.
    retrains = args.retrain_every > 0 and model_trainable
    if args.sampler == "importance":
        regime = "AIS" if retrains else "DIS"
    elif args.sampler == "spotcheck":
        regime = "spotAIS" if retrains else "spotDIS"
    else:
        regime = "random" if retrains else "random_frozen"
    parts.append(regime)

    if args.sampler == "spotcheck":
        parts.append(f"a{args.alpha:g}")
        parts.append(args.routing)
        if not model_trainable and args.refit_h_every > 0:
            parts.append(f"rh{args.refit_h_every}")

    if args.warmup_n > 0:
        parts.append(f"w{args.warmup_n}")
    if args.pool_size:
        parts.append(f"n{args.pool_size}")
    parts.append(f"s{args.seed}")
    return "__".join(parts)


def compute_run_dir(args: argparse.Namespace, run_id: str) -> Path:
    dataset, class_name = args.class_key.split("/", 1)
    return Path(args.out_root) / dataset / class_name / run_id


def resolve_tiling(args: argparse.Namespace) -> tuple[int, int, Optional[int], str]:
    """Per-class ``(max_whole_image_side, tile_size, tile_threshold, source)``.

    ``tile_size``/``tile_threshold`` come from the suggestion store
    (``--tile-sizes``) so the main loop tiles each class exactly as
    ``scripts/extract_dino_features.py`` did -- the precondition for the DINOv3
    feature cache aligning (see ``data/feature_cache.py``).  An explicit
    ``--tile-size``/``--tile-threshold`` overrides the store entry; the mapping
    mirrors ``tiling_for()`` in the extractor, so keep the two in sync.
    """
    spec: Optional[Dict[str, Any]] = None
    store_path = Path(args.tile_sizes) if args.tile_sizes else None
    if store_path is not None:
        if not store_path.is_file():
            if args.tile_size is None:
                raise SystemExit(
                    f"--tile-sizes {store_path} not found; pass --tile-size "
                    'explicitly, or --tile-sizes "" to disable the store.'
                )
        else:
            store = json.loads(store_path.read_text())
            spec = store.get(args.class_key)
            if spec is None and args.tile_size is None:
                raise SystemExit(
                    f"{args.class_key} is not in {store_path}; pass --tile-size "
                    "explicitly or regenerate the store via build_image_pools.py."
                )

    if args.tile_size is not None:
        ts = args.tile_size
        source = "cli"
    else:
        ts = int(spec["tile_size"])  # spec is guaranteed set here
        source = str(store_path)

    if args.tile_threshold is not None:
        threshold: Optional[int] = args.tile_threshold
    elif spec is not None:
        threshold = int(spec.get("tile_threshold", ts))
    else:
        threshold = None  # dataset falls back to max_whole_image_side

    return args.max_whole_image_side, ts, threshold, source


def build_dataset(args: argparse.Namespace) -> CountingClassTiledDataset:
    from counting_dataset import CountingDatasetIndex

    mws, ts, threshold, source = resolve_tiling(args)
    print(
        f"tiling[{source}]: tile_size={ts}, "
        f"tile_threshold={threshold if threshold is not None else mws}, "
        f"max_whole_image_side={mws}"
    )
    index = CountingDatasetIndex(root=Path(args.index_root))
    base = index.load_class(
        args.class_key,
        split=None,
        load_images=False,
        min_count=1,
        on_missing_split="warn",
    )
    # base: CountingClassDataset
    return CountingClassTiledDataset(
        base,
        raw_root=args.raw_root,
        max_whole_image_side=mws,
        tile_size=ts,
        tile_threshold=threshold,
        pool_size=args.pool_size,
        pool_sample_seed=args.pool_sample_seed,
        image_pool_list=args.image_pool_list,
    )


def make_eval_loader(
    ds: CountingClassTiledDataset, args: argparse.Namespace
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=args.eval_batch_size or args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_detection,
        pin_memory=True,
    )


def train_kwargs_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        backbone_lr_factor=args.backbone_lr_factor,
    )


def refit_h(
    feats: np.ndarray,
    revealed_set: set,
    gt_counts: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    """``(p_empty, h_empty)`` over the pool from the probe refit on ``revealed_set``."""
    return fit_h(
        feats,
        revealed_set,
        gt_counts,
        fpr_tol=args.h_fpr_tol,
        min_per_class=args.h_min_per_class,
        calib_folds=args.h_calib_folds,
        seed=args.seed,
    )


def capture_rng_states(rng: np.random.Generator) -> Dict[str, Any]:
    return {
        "numpy": rng.bit_generator.state,
        "torch": torch.random.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        ),
    }


def restore_rng_states(
    rng: np.random.Generator,
    state_numpy: Dict[str, Any],
    state_torch: torch.Tensor,
    state_torch_cuda: Optional[torch.Tensor],
) -> None:
    rng.bit_generator.state = state_numpy
    torch.random.set_rng_state(state_torch.cpu())
    if state_torch_cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state_torch_cuda.cpu())


def build_config(
    args: argparse.Namespace, run_id: str, ds: CountingClassTiledDataset
) -> Dict[str, Any]:
    cfg = vars(args).copy()
    cfg["run_id"] = run_id
    # Record the realized tiling, not the raw flags: tile_size/tile_threshold
    # may have come from --tile-sizes (in which case the flags are None).
    cfg["tile_size"] = int(ds.tile_size)
    cfg["tile_threshold"] = int(ds.tile_threshold)
    cfg["max_whole_image_side"] = int(ds.max_whole_image_side)
    cfg["F_true"] = int(ds.F_true)
    cfg["N"] = int(len(ds))  # number of units (tiles + (small) whole images)
    cfg["n_images"] = int(ds.num_images)
    return cfg


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    ds = build_dataset(args)
    loader = make_eval_loader(ds, args)
    n_total = len(ds)
    print(
        f"dataset: {n_total} units from {ds.num_images} images; " f"F_true={ds.F_true}"
    )

    # Empty filter h: frozen DINOv3 features (aligned to this pool), refit on the
    # revealed set at each snapshot.  h_empty drives the spotcheck proposal;
    # p_empty is cached per snapshot for offline replays.
    spotcheck = args.sampler == "spotcheck"
    feats = load_features(args.features_root, ds, args.class_key) if spotcheck else None
    gt_counts = np.asarray(ds.gt_counts)
    h_empty: Optional[np.ndarray] = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = build_model(args, device)
    model.configure_unfreeze(args.unfreeze)

    run_id = derive_run_id(args, model_trainable=model.trainable)
    run_dir = compute_run_dir(args, run_id)
    print(f"run_dir: {run_dir}")
    recorder = ExperimentRecorder(run_dir)

    # ----- resume or fresh -----
    resuming = (not args.fresh) and recorder.has_checkpoint()
    if resuming:
        state = recorder.load_resume_state()
        model.load_state_dict(state.model_state)
        restore_rng_states(
            rng,
            state.rng_numpy,
            state.rng_torch,
            state.rng_torch_cuda,
        )
        revealed_set = set(state.revealed)
        pred_counts = state.last_pred_counts
        active_snap_id = state.active_snap_id
        active_h_id = (
            state.active_h_id if state.active_h_id is not None else active_snap_id
        )
        h_empty = state.h_empty  # as of the active snapshot; None unless spotcheck
        start_t = state.last_t + 1
        print(
            f"resumed: t={state.last_t}, revealed={len(revealed_set)}, "
            f"active_snap={active_snap_id}"
        )
    else:
        if recorder.has_step_data() and not args.fresh:
            raise SystemExit(
                f"ERROR: {run_dir} already has run data but no checkpoint; "
                "pass --fresh to overwrite, or move the directory aside."
            )
        if args.fresh:
            deleted = recorder.wipe()
            if deleted:
                print(f"--fresh: deleted {sorted(deleted)}")

        recorder.write_config(build_config(args, run_id, ds))
        recorder.write_manifest(ds.manifest())

        warmup_picks: List[int] = []
        if args.warmup_n > 0:
            warmup_picks = sample_warmup(
                n_total,
                ds.gt_counts,
                strategy=args.warmup_strategy,
                n_warmup=args.warmup_n,
                rng=rng,
            )
            if model.trainable:
                print(
                    f"warmup: fitting on {len(warmup_picks)} tiles "
                    f"(strategy={args.warmup_strategy})"
                )
                model.fit(ds, warmup_picks, device, **train_kwargs_from_args(args))
            else:
                print(
                    f"warmup: revealing {len(warmup_picks)} tiles to seed h "
                    f"(strategy={args.warmup_strategy}; model frozen, not fitted)"
                )

        print("running initial inference...")
        pred_counts, det_scores = model.predict(
            loader,
            device,
            score_thresh=args.score_thresh,
            capture_floor=args.capture_score_floor,
        )
        revealed_set = set(warmup_picks)
        p_empty = None
        if spotcheck:
            p_empty, h_empty = refit_h(feats, revealed_set, gt_counts, args)
        active_snap_id = recorder.append_snapshot(
            pred_counts, step=0, h=p_empty, h_empty=h_empty
        )
        active_h_id = active_snap_id
        recorder.append_detections(active_snap_id, det_scores)
        print(f"snapshot 0: F_model={float(pred_counts.sum()):.1f}")

        recorder.write_warmup(
            [
                {
                    "s_t": s,
                    "f_st": ds.get_gt_count(s),
                    "g_st": float(pred_counts[s]),
                }
                for s in warmup_picks
            ]
        )

        recorder.save_checkpoint(
            model_state=model.state_dict(),
            rng_states=capture_rng_states(rng),
            last_t=0,
            active_snap_id=active_snap_id,
            h_empty=h_empty,
            active_h_id=active_h_id,
        )
        start_t = 1

    # ----- main loop -----
    end_t = (args.max_steps + 1) if args.max_steps else (n_total + 1)
    train_kwargs = train_kwargs_from_args(args)
    t = (
        start_t - 1
    )  # in case the loop body never runs (so the final-ckpt save below has a defined t)

    print(
        f"main loop: t={start_t}..{end_t-1} (sampler={args.sampler}, "
        f"retrain_every={args.retrain_every})"
    )
    t_loop_start = time.perf_counter()
    for t in range(start_t, end_t):
        if len(revealed_set) >= n_total:
            print("all units revealed; stopping")
            break

        s_t, q_t_st = sample_unit(
            revealed_set,
            n_total,
            pred_counts,
            sampler=args.sampler,
            min_count_per_tile=args.min_count_per_tile,
            rng=rng,
            h_empty=h_empty,
            alpha=args.alpha,
            routing=args.routing,
        )
        f_st = int(ds.get_gt_count(s_t))
        revealed_set.add(s_t)

        # Decide stop and retrain together; suppress the retrain if we are about
        # to stop, so we never pay a fine-tune + inference cost for a model we
        # will not consult again.
        will_stop = (
            args.max_fraction_labeled is not None
            and len(revealed_set) / n_total >= args.max_fraction_labeled
        )
        retrain_due = (
            args.retrain_every > 0 and model.trainable and t % args.retrain_every == 0
        )
        retrain_now = retrain_due and not will_stop
        # Frozen-model h refit: spotcheck zero-shot runs adapt h on its own
        # cadence while g stays fixed (trainable models refit h at each retrain).
        refit_h_now = (
            spotcheck
            and not model.trainable
            and args.refit_h_every > 0
            and t % args.refit_h_every == 0
            and not will_stop
        )

        recorder.append_step(
            t=t,
            s_t=s_t,
            f_st=f_st,
            q_t_st=q_t_st,
            snapshot_id=active_snap_id,
            retrain_step=retrain_now,
        )

        if retrain_now:
            model.fit(ds, list(revealed_set), device, **train_kwargs)
            pred_counts, det_scores = model.predict(
                loader,
                device,
                score_thresh=args.score_thresh,
                capture_floor=args.capture_score_floor,
            )
            p_empty = None
            if spotcheck:
                p_empty, h_empty = refit_h(feats, revealed_set, gt_counts, args)
            active_snap_id = recorder.append_snapshot(
                pred_counts, step=t, h=p_empty, h_empty=h_empty
            )
            active_h_id = active_snap_id
            recorder.append_detections(active_snap_id, det_scores)
            recorder.save_checkpoint(
                model_state=model.state_dict(),
                rng_states=capture_rng_states(rng),
                last_t=t,
                active_snap_id=active_snap_id,
                h_empty=h_empty,
                active_h_id=active_h_id,
            )
            print(
                f"[t={t}] retrained on {len(revealed_set)} tiles; "
                f"F_model={float(pred_counts.sum()):.1f}; "
                f"fraction_labeled={len(revealed_set)/n_total:.3f}"
            )

        if refit_h_now:
            p_empty, h_empty = refit_h(feats, revealed_set, gt_counts, args)
            active_h_id = recorder.append_h(p_empty, step=t, h_empty=h_empty)
            recorder.save_checkpoint(
                model_state=model.state_dict(),
                rng_states=capture_rng_states(rng),
                last_t=t,
                active_snap_id=active_snap_id,
                h_empty=h_empty,
                active_h_id=active_h_id,
            )

        if will_stop:
            print(
                f"max_fraction_labeled={args.max_fraction_labeled} reached; "
                "stopping"
            )
            break

    elapsed = time.perf_counter() - t_loop_start
    recorder.finalize()

    # Save a final checkpoint so resume sees the full step history.
    recorder.save_checkpoint(
        model_state=model.state_dict(),
        rng_states=capture_rng_states(rng),
        last_t=t,
        active_snap_id=active_snap_id,
        h_empty=h_empty,
        active_h_id=active_h_id,
    )
    elapsed_hms = str(datetime.timedelta(seconds=int(elapsed)))
    print(
        f"done at t={t} after {elapsed_hms}; "
        f"revealed {len(revealed_set)}/{n_total}"
    )


if __name__ == "__main__":
    main()
