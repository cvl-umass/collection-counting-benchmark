"""Model-path invariance check: is a run's cached model path seed-invariant?

The premise of the fixed-model trick -- if paths agree across seeds, replaying
one path under fresh sampling approximates a genuine run.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.experiment_recorder import ExperimentRecorder  # noqa: E402

_SEED_RE = re.compile(r"^(?P<base>.+)__s(?P<seed>\d+)$")
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_groups(runs_root: Path) -> Dict[Path, List[Tuple[int, Path]]]:
    """Map each seed-stripped group to its ``(seed, run_dir)`` list.

    A run dir qualifies only if it carries a ``__sSEED`` suffix and has a
    ``snapshots.npz`` on disk (incomplete/in-flight runs are skipped).
    """
    groups: Dict[Path, List[Tuple[int, Path]]] = {}
    for cfg_path in sorted(runs_root.glob("*/*/*/config.json")):
        run_dir = cfg_path.parent
        m = _SEED_RE.match(run_dir.name)
        if m is None or not (run_dir / "snapshots.npz").exists():
            continue
        base = run_dir.parent / m.group("base")
        groups.setdefault(base, []).append((int(m.group("seed")), run_dir))
    return groups


# ---------------------------------------------------------------------------
# Cross-seed agreement metrics
# ---------------------------------------------------------------------------


def _pairwise_mean_corr(mat: np.ndarray) -> Tuple[float, int]:
    """Mean off-diagonal correlation over seed pairs, nan-aware.

    ``mat`` is ``(K, N)``; a seed row with zero variance (e.g. all-zero g)
    yields nan correlations that are dropped.  Returns ``(mean, n_valid_pairs)``.
    """
    K = mat.shape[0]
    if K < 2:
        return float("nan"), 0
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(mat)  # nan rows for any zero-variance (e.g. all-0) seed
    iu = np.triu_indices(K, k=1)
    vals = c[iu]
    vals = vals[np.isfinite(vals)]
    return (float(vals.mean()) if vals.size else float("nan")), int(vals.size)


def _pairwise_mean_spearman(mat: np.ndarray) -> Tuple[float, int]:
    ranks = np.vstack([rankdata(row) for row in mat])
    return _pairwise_mean_corr(ranks)


def _pairwise_mean_tv(prob: np.ndarray) -> float:
    """Mean pairwise total-variation distance between rows that each sum to 1."""
    K = prob.shape[0]
    if K < 2:
        return float("nan")
    tot, n = 0.0, 0
    for i in range(K):
        for j in range(i + 1, K):
            tot += 0.5 * float(np.abs(prob[i] - prob[j]).sum())
            n += 1
    return tot / n


def checkpoint_metrics(
    G: np.ndarray, F_true: float, floor: float, importance: bool
) -> Dict[str, float]:
    """All cross-seed metrics for one aligned checkpoint.  ``G`` is ``(K, N)``."""
    Fm = G.sum(axis=1)
    mean_F = float(Fm.mean())
    std_F = float(Fm.std(ddof=1)) if Fm.size > 1 else 0.0

    pearson, n_pairs = _pairwise_mean_corr(G)
    spearman, _ = _pairwise_mean_spearman(G)

    unit_sd = G.std(axis=0, ddof=1)  # per-unit cross-seed SD
    mean_count = float(G.mean())
    rel_unit_sd = float(unit_sd.mean() / (mean_count + _EPS))

    prop_tv = float("nan")
    if importance:
        w = np.maximum(G, float(floor))
        P = w / (w.sum(axis=1, keepdims=True) + _EPS)
        prop_tv = _pairwise_mean_tv(P)

    return {
        "n_seeds": int(G.shape[0]),
        "n_pairs": int(n_pairs),
        "F_model_mean": mean_F,
        "F_model_cv": (std_F / mean_F) if mean_F > 0 else float("nan"),
        "F_model_rel_bias": (mean_F - F_true) / F_true if F_true > 0 else float("nan"),
        "pearson_mean": pearson,
        "spearman_mean": spearman,
        "rel_unit_sd": rel_unit_sd,
        "prop_tv_mean": prop_tv,
    }


# ---------------------------------------------------------------------------
# Per-group processing
# ---------------------------------------------------------------------------


def _load_seed(run_dir: Path) -> Optional[Tuple[Dict[int, int], np.ndarray, dict]]:
    """Return ``(step -> row index, g matrix, config)`` for one run, or None."""
    try:
        g, steps = ExperimentRecorder.load_snapshots(run_dir)
        cfg = ExperimentRecorder.load_config(run_dir)
    except (FileNotFoundError, KeyError, ValueError):
        return None
    step_to_row = {int(s): i for i, s in enumerate(steps)}
    return step_to_row, np.asarray(g, dtype=np.float64), cfg


# ---------------------------------------------------------------------------
# Spotcheck proposal reconstruction (optional; --spot-tv)
# ---------------------------------------------------------------------------
#
# The spotcheck proposal is q = alpha*q_spot + (1-alpha)*q_count, gated by the
# boolean empty-filter decision h_empty -- which is *not* stored per snapshot
# (only the p_empty scores are).  So prop_tv for spotAIS is reconstructed by
# replaying the run's h pipeline exactly: refit the DINOv3 probe on each seed's
# revealed set D_t at the matched step, take the same D_t-calibrated cut, and
# build the mixture over the full pool (the same full-pool convention the
# importance prop_tv uses).  Faithful up to the sklearn build: the h refit is a
# deterministic function of (features, D_t, seed), so any residual score offset
# is common-mode across seeds and cancels in the cross-seed TV.


def _revealed_at(warmup: List[int], stepsdf: "pd.DataFrame", T: int) -> Set[int]:
    """Revealed set as of the snapshot at step ``T``: warmup + IS picks s_1..s_T.

    The snapshot at step ``T`` was fit on ``D_T`` *after* ``s_T`` was revealed
    (see ``experiments/run.py``), so the cut is ``t <= T``.
    """
    r = set(int(s) for s in warmup)
    if T > 0:
        r |= set(int(s) for s in stepsdf.loc[stepsdf["t"] <= T, "s_t"])
    return r


def _load_spot_aux(run_dir: Path) -> Tuple[np.ndarray, List[int], "pd.DataFrame"]:
    """Per-seed pieces for h replay: ``(gt_counts, warmup picks, steps[t,s_t])``."""
    gt = (
        ExperimentRecorder.load_manifest(run_dir)
        .sort_values("tile_index")["gt_count"]
        .to_numpy(np.float64)
    )
    warmup = [int(s) for s in ExperimentRecorder.load_warmup(run_dir)["s_t"]]
    stepsdf = ExperimentRecorder.load_steps(run_dir)[["t", "s_t"]]
    return gt, warmup, stepsdf


def _load_feats(cfg: dict, manifest: "pd.DataFrame", features_root: str) -> np.ndarray:
    """L2-normalised DINOv3 features aligned to ``tile_index`` (matches ``fit_h``)."""
    from data.feature_cache import align_features, cache_dir_for

    cdir = cache_dir_for(
        Path(features_root), str(cfg["class_key"]),
        int(cfg["max_whole_image_side"]), int(cfg["tile_size"]),
    )
    rows = (
        manifest.sort_values("tile_index")[["image_id", "x0", "y0", "tile_index"]]
        .to_dict("records")
    )
    feats, _ = align_features(cdir, rows, strict=True)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return (feats / np.maximum(norms, 1e-10)).astype(np.float32)


def _spot_prop_tv(
    present: List[Tuple[int, np.ndarray]],
    aux: Dict[int, Tuple[np.ndarray, List[int], "pd.DataFrame"]],
    feats: np.ndarray,
    hp: dict,
    step: int,
) -> float:
    """Mean pairwise TV of the reconstructed spotcheck proposals at one step."""
    from experiments.empty_filter import fit_h
    from experiments.sampling import _spotcheck_weights

    Q: List[np.ndarray] = []
    for seed, g_row in present:
        gt, warmup, stepsdf = aux[seed]
        revealed = _revealed_at(warmup, stepsdf, step)
        _, h_empty = fit_h(
            feats, revealed, gt,
            fpr_tol=hp["fpr_tol"], min_per_class=hp["min_per_class"],
            calib_folds=hp["calib_folds"], seed=int(seed),
        )
        q = _spotcheck_weights(
            g_row.astype(np.float64), h_empty, hp["alpha"], hp["routing"]
        )
        s = float(q.sum())
        Q.append(q / s if s > 0 else q)
    return _pairwise_mean_tv(np.vstack(Q))


def process_group(
    base: Path,
    seeds: List[Tuple[int, Path]],
    min_seeds: int,
    spot_tv: bool = False,
    features_root: str = "data/dinov3",
) -> List[Dict[str, object]]:
    """One row per checkpoint shared by >= ``min_seeds`` seeds of this group."""
    loaded = []
    for seed, run_dir in sorted(seeds):
        out = _load_seed(run_dir)
        if out is not None:
            s2r, g, cfg = out
            loaded.append((seed, run_dir, s2r, g, cfg))
    if len(loaded) < min_seeds:
        return []

    # Pool geometry / settings must agree across seeds; take the first and guard N.
    cfg0 = loaded[0][4]
    N = int(cfg0["N"])
    if any(g.shape[1] != N for _, _, _, g, _ in loaded):
        print(f"  SKIP {base}: N mismatch across seeds", file=sys.stderr)
        return []
    F_true = float(cfg0.get("F_true", 0.0))
    warmup_n = int(cfg0.get("warmup_n", 0) or 0)
    floor = float(cfg0.get("min_count_per_tile", 1.0) or 1.0)
    sampler = str(cfg0.get("sampler", "random"))
    # checkpoint_metrics fills prop_tv for importance (q ~ max(g, floor)); random
    # has no g-induced proposal.  spotcheck is filled below via reconstruction.
    importance = sampler == "importance"

    rel = base.relative_to(base.parents[2])  # <dataset>/<class>/<regime>
    dataset, class_name, regime = rel.parts[0], rel.parts[1], rel.parts[2]

    # Optional spotcheck proposal reconstruction (needs the cached features).
    spot = sampler == "spotcheck" and spot_tv
    feats: Optional[np.ndarray] = None
    aux: Dict[int, Tuple] = {}
    hp: Optional[dict] = None
    if spot:
        try:
            man0 = ExperimentRecorder.load_manifest(loaded[0][1])
            feats = _load_feats(cfg0, man0, features_root)
            for seed, run_dir, *_ in loaded:
                aux[seed] = _load_spot_aux(run_dir)
            hp = dict(
                fpr_tol=float(cfg0.get("h_fpr_tol", 0.01)),
                min_per_class=int(cfg0.get("h_min_per_class", 2)),
                calib_folds=int(cfg0.get("h_calib_folds", 5)),
                alpha=float(cfg0.get("alpha", 0.05)),
                routing=str(cfg0.get("routing", "trust_h")),
            )
        except Exception as e:  # missing/mismatched cache -> leave prop_tv NaN
            print(f"  spot-TV disabled for {dataset}/{class_name}/{regime}: {e}",
                  file=sys.stderr)
            spot = False

    # Union of steps; keep those reached by enough seeds.
    all_steps = sorted({s for _, _, s2r, _, _ in loaded for s in s2r})
    rows: List[Dict[str, object]] = []
    for step in all_steps:
        present = [
            (seed, g[s2r[step]]) for seed, _, s2r, g, _ in loaded if step in s2r
        ]
        if len(present) < min_seeds:
            continue
        G = np.vstack([row for _, row in present])
        m = checkpoint_metrics(G, F_true, floor, importance)
        if spot:
            try:
                m["prop_tv_mean"] = _spot_prop_tv(present, aux, feats, hp, int(step))
            except Exception as e:
                print(f"  spot-TV step {step} {dataset}/{class_name}: {e}",
                      file=sys.stderr)
        m.update(
            dataset=dataset,
            cls=class_name,
            regime=regime,
            sampler=sampler,
            step=int(step),
            frac_labeled=(warmup_n + int(step)) / N if N else float("nan"),
            N=N,
            F_true=F_true,
        )
        rows.append(m)
    return rows


# ---------------------------------------------------------------------------
# Summary + verdict
# ---------------------------------------------------------------------------

_COL_ORDER = [
    "dataset", "cls", "regime", "sampler", "step", "frac_labeled", "N",
    "n_seeds", "n_pairs", "F_true", "F_model_mean", "F_model_cv",
    "F_model_rel_bias", "pearson_mean", "spearman_mean", "rel_unit_sd",
    "prop_tv_mean",
]


def verdict(spearman: float, cv: float, pass_spear: float, pass_cv: float) -> str:
    if not np.isfinite(spearman) or not np.isfinite(cv):
        return "NA"
    if spearman >= pass_spear and cv <= pass_cv:
        return "PASS"
    if spearman < 0.90 or cv > 2 * pass_cv:
        return "FAIL"
    return "WARN"


def build_summary(
    per_ckpt: pd.DataFrame,
    ref_fracs: List[float],
    pass_spear: float,
    pass_cv: float,
) -> pd.DataFrame:
    """Headline metrics at each ref budget (nearest checkpoint), per group.

    Headlining at fixed fractions (not each group's deepest checkpoint) keeps
    the cross-group comparison fair: deep checkpoints survive on only the
    longest-running seeds (small K), whereas a shallow fixed budget is reached
    by every seed.  One row per (group, ref_frac).
    """
    keys = ["dataset", "cls", "regime"]
    out = []
    for _, g in per_ckpt.groupby(keys):
        for rf in ref_fracs:
            row = g.iloc[(g["frac_labeled"] - rf).abs().argsort().iloc[0]].copy()
            row["ref_frac"] = rf
            row["verdict"] = verdict(
                row["spearman_mean"], row["F_model_cv"], pass_spear, pass_cv
            )
            out.append(row)
    cols = keys + [
        "ref_frac", "sampler", "step", "frac_labeled", "n_seeds",
        "spearman_mean", "pearson_mean", "F_model_cv", "F_model_rel_bias",
        "rel_unit_sd", "prop_tv_mean", "verdict",
    ]
    return pd.DataFrame(out)[cols].sort_values(keys + ["ref_frac"]).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Plotting (optional)
# ---------------------------------------------------------------------------


def make_plots(per_ckpt: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"  plotting skipped: {e}", file=sys.stderr)
        return

    panels = [
        ("spearman_mean", "mean pairwise Spearman of g"),
        ("F_model_cv", "cross-seed CV of Sum g"),
    ]
    for regime, sub in per_ckpt.groupby("regime"):
        fig, axes = plt.subplots(1, len(panels), figsize=(11, 4.2))
        for ax, (col, title) in zip(axes, panels):
            for (ds, cls), g in sub.groupby(["dataset", "cls"]):
                g = g.sort_values("frac_labeled")
                ax.plot(g["frac_labeled"], g[col], marker=".", ms=3,
                        lw=1, label=f"{ds}/{cls}")
            ax.set_xlabel("fraction labeled")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        axes[0].legend(fontsize=6, ncol=2)
        fig.suptitle(f"Model-path invariance -- {regime}")
        fig.tight_layout()
        path = out_dir / f"{regime}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", default="outputs/runs")
    ap.add_argument("--out-dir", default="outputs/analysis/path_invariance")
    ap.add_argument("--min-seeds", type=int, default=2,
                    help="min seeds that must share a checkpoint to score it")
    ap.add_argument("--summary-fracs", default="0.1,0.2,0.3",
                    help="fixed budgets to headline at (nearest checkpoint each)")
    ap.add_argument("--pass-spearman", type=float, default=0.95)
    ap.add_argument("--pass-cv", type=float, default=0.05)
    ap.add_argument("--spot-tv", action="store_true",
                    help="reconstruct the spotcheck proposal (replays the empty "
                         "filter h on cached DINOv3 features) to fill prop_tv_mean "
                         "for spotAIS groups; slower and needs --features-root")
    ap.add_argument("--features-root", default="data/dinov3",
                    help="root of the cached DINOv3 features (for --spot-tv)")
    ap.add_argument("--plot", action="store_true", help="also write per-regime PNGs")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    if not runs_root.is_dir():
        raise SystemExit(f"--runs-root {runs_root} not found")

    groups = discover_groups(runs_root)
    print(f"discovered {len(groups)} run groups under {runs_root}")
    if args.spot_tv:
        print(f"--spot-tv: reconstructing spotcheck proposals from "
              f"{args.features_root} (slower)")

    all_rows: List[Dict[str, object]] = []
    for base in sorted(groups):
        all_rows.extend(process_group(
            base, groups[base], args.min_seeds,
            spot_tv=args.spot_tv, features_root=args.features_root,
        ))
    if not all_rows:
        raise SystemExit("no scorable groups (need >= --min-seeds complete seeds)")

    ref_fracs = [float(x) for x in args.summary_fracs.split(",") if x.strip()]
    per_ckpt = pd.DataFrame(all_rows)[_COL_ORDER]
    summary = build_summary(per_ckpt, ref_fracs, args.pass_spearman, args.pass_cv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_ckpt.to_csv(out_dir / "per_checkpoint.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    idx = ["dataset", "cls", "regime"]
    with pd.option_context(
        "display.max_rows", None, "display.width", 220,
        "display.float_format", lambda v: f"{v:.3f}",
    ):
        for col, label in [("spearman_mean", "mean pairwise Spearman of g"),
                           ("F_model_cv", "cross-seed CV of Sum g")]:
            piv = summary.pivot_table(index=idx, columns="ref_frac", values=col)
            print(f"\n=== {label} at fixed budgets (fraction labeled) ===")
            print(piv.to_string())
        deepest_rf = max(ref_fracs)
        verd = summary[summary["ref_frac"] == deepest_rf]
        counts = verd["verdict"].value_counts().to_dict()
        print(f"\nverdict @ {deepest_rf:.0%} budget: {counts}  "
              f"(PASS = spearman >= {args.pass_spearman} and "
              f"F_model_cv <= {args.pass_cv}; heuristic)")
    print(f"\nwrote {out_dir/'per_checkpoint.csv'} and {out_dir/'summary.csv'}")

    if args.plot:
        make_plots(per_ckpt, out_dir)


if __name__ == "__main__":
    main()
