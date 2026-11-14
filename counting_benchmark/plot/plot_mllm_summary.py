"""Summary figures and CSVs for the off-the-shelf model comparison.

Writes ``winner_model_grid.csv``, which several of the other plot scripts read,
so this one runs first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The model palette + shared styling live in palette.py (single source of truth for every
# figure -- edit colors there, not here).  This module keeps only the winner-grid logic.
from palette import (  # noqa: E402,F401
    INK, INK2, MUTED, GRID, AXIS, SURFACE, TIERS,
    FAMILY, FAM_ORDER, FAM_COLOR, FAM_BASE, FAM_LABEL, FAM_TEXT, FAM_LEGEND,
    FAM_MEMBERS, FAM_RAMP, ABBR, NAME, MODEL_COLOR, MODEL_TEXT, MC_SENTINEL,
    style, class_meta)

# --- winner-grid candidate logic (estimator semantics, not palette) ----------
# A model's candidate error excludes the biased pure model-only baseline and the
# is_uniform diagnostic; every real estimator (difference/ratio/regression/dis/
# ais/spot...) is fair game.
DROP_ESTIMATORS = {"model_only_pure", "is_uniform"}
# monte_carlo is the *uniform survey mean* -- it never looks at the model, so its
# error is identical across every model that shares a random trajectory.  Crediting
# a model for it would be a false attribution, so we pull it out as a single
# model-agnostic "floor" (mc_floor, colored by the MC_SENTINEL model key) that
# competes against the models but is not tied to any one of them.
MC_ESTIMATOR = "monte_carlo"
# These g-using estimators are reported only on the uniform (random) stream:
# model_only_corrected ignores the sampling design (F(R_t)+sum g, no q), and
# difference/ratio/regression are the correction-under-uniform-sampling family
# (their directed variants were diagnostics, not recommended estimators).  The
# directed importance samplers (dis/ais/spotdis/spotais) keep their own sampler.
RANDOM_ONLY_ESTIMATORS = {"model_only_corrected",
                          "difference", "ratio", "regression"}
# AIS/spotAIS only make sense for the two retraining FRCNN detectors; DIS/spotDIS
# are the frozen-model estimator, valid only for the 11 off-the-shelf models (see
# the method roster in Table 1).  The pipeline also carries a FRCNN
# "frozen at snapshot 0" DIS/spotDIS diagnostic cell (a deliberate side
# comparison) and does directed
# replays of the random-only family -- neither is a candidate here.
ADAPTIVE_ESTIMATORS = {"ais", "spotais"}
FROZEN_ESTIMATORS = {"dis", "spotdis"}
TIE_MARGIN = 0.05  # runner-up within 5% -> no meaningful winner -> split cell

_EST_SHORT = {"monte_carlo": "MC", "model_only_corrected": "model",
              "difference": "diff", "ratio": "ratio", "regression": "reg",
              "dis": "DIS", "ais": "AIS", "spotdis": "spotDIS", "spotais": "spotAIS"}


def _abbr(model: str) -> str:
    return "MC" if model == MC_SENTINEL else ABBR[model]


def method_label(est: str, smp: str) -> str:
    """Compact estimator tag.  Every valid candidate's sampler is fixed by its
    estimator (correction/baseline live on the uniform random stream; dis/ais/
    spotdis/spotais are their own sampler), so the sampler is implied and we
    show only the estimator."""
    return _EST_SHORT[est]


def cell_labels(model: str, est: str, smp: str) -> tuple[str, str]:
    """(top, bottom) text for a grid cell -- the MC floor names no model."""
    if model == MC_SENTINEL:
        return "MC", "uniform"
    return ABBR[model], method_label(est, smp)


_KEEP = ["class_key", "checkpoint", "model", "estimator", "sampler",
         "err", "tier", "frac_empty", "family"]


def model_best(df: pd.DataFrame, exclude_estimators=frozenset()) -> pd.DataFrame:
    """The estimator-sampler each model does best with at every (class, budget).

    One row per (class, checkpoint, model): the argmin over that model's *valid,
    model-using* estimator-sampler pairs (monte_carlo excluded -- it is the
    model-agnostic floor, see ``mc_floor``).  Keeps the winning ``estimator``/
    ``sampler`` and its ``err`` (mean |frac error|).

    ``exclude_estimators`` adds names to drop from the candidate pool on top of
    the fixed ``DROP_ESTIMATORS`` -- e.g. ``{"spotdis", "spotais"}`` to see who
    wins without the spot-check variants in the running."""
    misapplied = (df.estimator.isin(RANDOM_ONLY_ESTIMATORS)
                  & (df.sampler != "random"))
    adaptive_model = df.model.map(FAMILY).eq("detector")
    wrong_regime = ((df.estimator.isin(ADAPTIVE_ESTIMATORS) & ~adaptive_model)
                    | (df.estimator.isin(FROZEN_ESTIMATORS) & adaptive_model))
    drop = DROP_ESTIMATORS | {MC_ESTIMATOR} | set(exclude_estimators)
    cand = df[~df.estimator.isin(drop) & ~misapplied & ~wrong_regime]
    idx = cand.groupby(["class_key", "checkpoint", "model"]).mean_abs_frac_error.idxmin()
    out = cand.loc[idx].rename(columns={"mean_abs_frac_error": "err"}).copy()
    out["family"] = out.model.map(FAMILY)
    return out[_KEEP].reset_index(drop=True)


def mc_floor(df: pd.DataFrame) -> pd.DataFrame:
    """The model-agnostic uniform-Monte-Carlo floor, one row per (class, budget).

    monte_carlo error barely varies across models (only through which random
    trajectory they share), so we collapse it to its best value per cell and
    label it as the pseudo-model ``MC`` in the ``mc_floor`` family."""
    mc = df[(df.estimator == MC_ESTIMATOR) & (df.sampler == "random")]
    idx = mc.groupby(["class_key", "checkpoint"]).mean_abs_frac_error.idxmin()
    out = mc.loc[idx].rename(columns={"mean_abs_frac_error": "err"}).copy()
    out["model"] = MC_SENTINEL
    out["family"] = "mc_floor"
    return out[_KEEP].reset_index(drop=True)


# ---------------------------------------------------------------- winner grid

def _winner_tables(best: pd.DataFrame):
    """(per-cell winner/runner-up table, full per-model ranking table)."""
    best = best.sort_values(["class_key", "checkpoint", "err"]).copy()
    best["rank"] = best.groupby(["class_key", "checkpoint"]).err.rank(
        method="first").astype(int)

    ranking = best[["class_key", "tier", "frac_empty", "checkpoint", "rank",
                    "model", "family", "estimator", "sampler", "err"]].copy()
    ranking["pct_labeled"] = (100 * ranking.checkpoint).round().astype(int)

    win = best[best["rank"] == 1].set_index(["class_key", "checkpoint"])
    run = best[best["rank"] == 2].set_index(["class_key", "checkpoint"])
    cells = win[["tier", "frac_empty", "model", "family",
                 "estimator", "sampler", "err"]].add_prefix("winner_")
    cells = cells.rename(columns={"winner_tier": "tier",
                                  "winner_frac_empty": "frac_empty"})
    cells = cells.join(run[["model", "family", "estimator", "sampler", "err"]]
                       .add_prefix("runner_"))
    cells = cells.reset_index()
    cells["pct_labeled"] = (100 * cells.checkpoint).round().astype(int)
    cells["margin_pct"] = (100 * (cells.runner_err / cells.winner_err - 1)).round(1)
    cells["tie"] = cells.margin_pct < 100 * TIE_MARGIN

    # Degenerate-g flag: on a directed estimator a model whose g is ~uniform drops
    # its model term (coefficient cancellation), so several such models land on the
    # EXACT same error -- an algebraic identity, not a coincidence.  When >=2 models
    # tie exactly at the cell's best, the "winner" is not really that model doing the
    # work (same false-attribution the MC floor avoids); we flag it so it can be
    # reported honestly rather than credited.  (mc_floor rows excluded -- already
    # model-agnostic by construction.)
    deg = _degeneracy(best[best.family != "mc_floor"])
    cells = cells.merge(deg, on=["class_key", "checkpoint"], how="left")
    cells["winner_degenerate"] = ((cells.winner_family != "mc_floor")
                                  & (cells.n_model_tied >= 2))

    cols = ["class_key", "tier", "frac_empty", "checkpoint", "pct_labeled",
            "winner_model", "winner_family", "winner_estimator",
            "winner_sampler", "winner_err", "winner_degenerate", "tied_models",
            "runner_model", "runner_family",
            "runner_estimator", "runner_sampler", "runner_err",
            "margin_pct", "tie"]
    return cells[cols], ranking


def _degeneracy(model_rows: pd.DataFrame) -> pd.DataFrame:
    """Per (class, budget): how many models tie EXACTLY at the best error, and
    which -- the coefficient-cancellation (degenerate-g) signature."""
    out = []
    for (cls, cp), g in model_rows.groupby(["class_key", "checkpoint"]):
        best = g.err.min()
        tied = g[np.isclose(g.err, best, atol=1e-9)]
        out.append((cls, cp, len(tied), ",".join(sorted(tied.model))))
    return pd.DataFrame(out, columns=["class_key", "checkpoint",
                                      "n_model_tied", "tied_models"])


def _present_families(cells: pd.DataFrame):
    """(families present as a winner/tie-runner, member-list per family)."""
    drawn = set(cells.winner_model) | set(cells.loc[cells.tie, "runner_model"])
    members_of = {f: [m for m in ([MC_SENTINEL] if f == "mc_floor"
                                  else FAM_MEMBERS[f]) if m in drawn]
                  for f in FAM_ORDER + ["mc_floor"]}
    fams = [f for f in FAM_ORDER + ["mc_floor"] if members_of[f]]
    return fams, members_of


def _grid_legend(lg, fams, members_of) -> None:
    """Grouped legend: one column per present family, a shaded swatch + name per
    model that appears in the grid.  (Estimator abbreviations are defined in the
    caption, not on the figure.)"""
    lg.set_xlim(0, 1)
    lg.set_ylim(0, 1)
    lg.axis("off")
    n = len(fams)
    for i, f in enumerate(fams):
        x0 = i / n + 0.006
        lg.text(x0, 0.94, FAM_LEGEND[f], fontsize=8.5, fontweight="bold",
                color=INK, va="top")
        for j, m in enumerate(members_of[f]):
            y = 0.72 - j * 0.155
            # No swatch outline: the grid cells have none either, and an outline
            # darkens the pale members enough to break the tint ordering.
            lg.scatter([x0 + 0.010], [y], marker="s", s=95,
                       color=MODEL_COLOR[m], linewidths=0)
            lg.text(x0 + 0.030, y, f"{_abbr(m)}  {NAME[m]}", fontsize=7.2,
                    va="center", color=INK2)


def fig_winner_grid(best: pd.DataFrame, meta: pd.DataFrame, cells: pd.DataFrame,
                    out: Path, budgets=None, cell_h: float = 0.62,
                    fig_width: float | None = None) -> None:
    cps = sorted(budgets) if budgets is not None else sorted(best.checkpoint.unique())
    cps = [cp for cp in cps if cp in set(best.checkpoint)]
    nrow, ncol = len(meta), len(cps)
    cls_row = {c: nrow - 1 - i for i, c in enumerate(meta.class_key)}  # full at top
    cp_col = {cp: j for j, cp in enumerate(cps)}
    cells = cells[cells.checkpoint.isin(cps)]
    cell_by = cells.set_index(["class_key", "checkpoint"])
    fams, members_of = _present_families(cells)

    gh = cell_h * nrow        # taller cells: room for model + estimator, no overlap
    leg_h = 0.30 * max((len(members_of[f]) for f in fams), default=1) + 0.7
    # width is the larger of what the grid needs and what the legend needs, so
    # the family columns never overlap when the grid is narrow (few budgets).
    # Passing fig_width pins it to the paper's text width instead, so the point
    # sizes below are the sizes that reach the page (no LaTeX rescaling).
    fig_w = fig_width or max(0.54 * ncol + 2.3, 1.72 * len(fams) + 0.4)
    fig = plt.figure(figsize=(fig_w, gh + leg_h), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[gh, leg_h], hspace=0.14)
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(SURFACE)

    for (cls, cp), row in cell_by.iterrows():
        r, c = cls_row[cls], cp_col[cp]
        wm, wmeth = cell_labels(row.winner_model, row.winner_estimator,
                                row.winner_sampler)
        wcol, wtxt = MODEL_COLOR[row.winner_model], MODEL_TEXT[row.winner_model]
        if row.tie:
            rm, rmeth = cell_labels(row.runner_model, row.runner_estimator,
                                    row.runner_sampler)
            rcol, rtxt = MODEL_COLOR[row.runner_model], MODEL_TEXT[row.runner_model]
            # diagonal cut top-right -> bottom-left; winner = upper-left half.
            tl, tr, br, bl = (c, r + 1), (c + 1, r + 1), (c + 1, r), (c, r)
            ax.add_patch(Polygon([tl, tr, bl], closed=True, facecolor=wcol,
                                 edgecolor=SURFACE, linewidth=2))
            ax.add_patch(Polygon([tr, br, bl], closed=True, facecolor=rcol,
                                 edgecolor=SURFACE, linewidth=2))
            # The cut is the line y=x in cell coordinates, so a label at height y
            # has only x<y of room: push each half's text toward its own corner,
            # which buys the long names (spotDIS) width they cannot get mid-cell.
            ax.text(c + 0.32, r + 0.80, wm, ha="center", va="center",
                    fontsize=6.8, fontweight="bold", color=wtxt)
            ax.text(c + 0.32, r + 0.60, wmeth, ha="center", va="center",
                    fontsize=5.7, color=wtxt, alpha=0.82)
            ax.text(c + 0.68, r + 0.40, rm, ha="center", va="center",
                    fontsize=6.8, fontweight="bold", color=rtxt)
            ax.text(c + 0.68, r + 0.20, rmeth, ha="center", va="center",
                    fontsize=5.7, color=rtxt, alpha=0.82)
        else:
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=wcol,
                                   edgecolor=SURFACE, linewidth=2))
            ax.text(c + 0.5, r + 0.63, wm, ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=wtxt)
            ax.text(c + 0.5, r + 0.35, wmeth, ha="center", va="center",
                    fontsize=7.2, color=wtxt, alpha=0.82)

    ax.set_xlim(0, ncol)
    ax.set_ylim(0, nrow)
    ax.set_xticks(np.arange(ncol) + 0.5, [f"{100 * cp:g}" for cp in cps], fontsize=8.5)
    # Class name on the axis, emptiness in its own right-hand column: one ragged
    # parenthetical per row read as noise, two aligned columns read as a table.
    ax.set_yticks([cls_row[c] + 0.5 for c in meta.class_key],
                  list(meta.short), fontsize=8.5, color=INK2)
    ax.text(ncol + 0.14, nrow + 0.22, "% empty", fontsize=7.5, color=MUTED,
            ha="left", va="bottom")
    for _, m in meta.iterrows():
        ax.text(ncol + 0.14, cls_row[m.class_key] + 0.5, f"{m.frac_empty:.0%}",
                fontsize=8, color=MUTED, ha="left", va="center")
    # Hairline between emptiness tiers -- shows the row order is a grouping, not
    # an arbitrary sort, without spending words on it.
    keys = list(meta.class_key)
    for i in range(len(meta) - 1):
        if meta.tier.iloc[i] != meta.tier.iloc[i + 1]:
            y = cls_row[keys[i]]
            ax.plot([-0.02, ncol + 0.02], [y, y], color=AXIS, linewidth=0.9,
                    solid_capstyle="butt", zorder=5)
    ax.set_xlabel("% of collection labeled", fontsize=9.5, color=INK2)
    ax.tick_params(colors=MUTED, length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    _grid_legend(fig.add_subplot(gs[1]), fams, members_of)
    # PDF as well as PNG: this one goes in the main text, and a vector copy keeps
    # the type crisp at print resolution (Type 42, set in the rcParams above).
    for ext in ("png", "pdf"):
        fig.savefig(out / f"winner_model_grid.{ext}", dpi=200,
                    bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------ detector vs mllm

def _family_envelope(best: pd.DataFrame) -> pd.DataFrame:
    """Per (class, checkpoint, family) the single best member: which model and
    estimator reached the family's lowest error at that budget."""
    idx = best.groupby(["class_key", "checkpoint", "family"]).err.idxmin()
    fam = best.loc[idx, ["class_key", "checkpoint", "family", "model",
                         "estimator", "sampler", "err", "tier", "frac_empty"]].copy()
    fam["pct_labeled"] = (100 * fam.checkpoint).round().astype(int)
    fam["family_rank"] = fam.groupby(["class_key", "checkpoint"]).err.rank(
        method="first").astype(int)
    return fam.reset_index(drop=True)


def fig_detector_vs_mllm(fam: pd.DataFrame, meta: pd.DataFrame, out: Path) -> None:
    """Best-per-family error over budget, one panel per class -- the crossover.

    Each *point* is whichever family member (model + estimator) is lowest at that
    budget, so a curve is an envelope over the family and the underlying model can
    change from budget to budget -- it is not one fixed model.  The gray dotted
    line is the model-agnostic uniform-Monte-Carlo floor: a family curve above it
    means no model in that family beats plain uniform labeling."""
    style_kw = {"detector": ("-", 2.4, 1.0), "open_mllm": ("-", 2.2, 1.0),
                "prop_mllm": ("-", 2.2, 1.0), "vlm": ("--", 1.4, 0.75)}
    fig, axes = plt.subplots(4, 3, figsize=(14.5, 13), facecolor=SURFACE)
    for ax, (_, m) in zip(axes.flat, meta.iterrows()):
        style(ax)
        sub = fam[fam.class_key == m.class_key]
        mc = sub[sub.family == "mc_floor"].sort_values("checkpoint")
        if len(mc):
            ax.plot(100 * mc.checkpoint, mc.err, ":", color=FAM_COLOR["mc_floor"],
                    linewidth=1.8, label=FAM_LABEL["mc_floor"], zorder=1)
        for f in FAM_ORDER:
            s = sub[sub.family == f].sort_values("checkpoint")
            if not len(s):
                continue
            ls, lw, alpha = style_kw[f]
            ax.plot(100 * s.checkpoint, s.err, ls, color=FAM_COLOR[f],
                    linewidth=lw, alpha=alpha, marker="o", markersize=3.5,
                    label=FAM_LABEL[f], zorder=3 if f != "vlm" else 2)
        ax.set_yscale("log")
        ax.set_title(f"{m.class_key}  ({m.tier}, {m.frac_empty:.0%} empty)",
                     fontsize=9, color=INK)
    for ax in axes[-1]:
        ax.set_xlabel("% labeled", fontsize=9, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("best mean |frac error| in family", fontsize=9, color=INK2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("When does the fine-tuned detector overtake the MLLMs?\n"
                 "(each point = the best model+estimator in that family at that "
                 "budget; the winning model can change across budgets)",
                 y=1.05, fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(out / "detector_vs_mllm.png", dpi=150, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------ directed vs random training

def fig_training_ais_vs_random(df: pd.DataFrame, meta: pd.DataFrame, out: Path,
                               model: str = "frcnn_resnet50") -> None:
    """Does directed (AIS) training sharpen the detector, or just reveal labels?

    Only the FRCNN detectors retrain, and AIS reveals positive tiles far earlier
    than uniform random.  We separate the two effects with the two model-only
    curves: ``pure`` (``sum g``) reflects the trained detector's prediction alone,
    while ``corrected`` (``F(R_t)+sum g``) also folds in the exact revealed labels.
    If seeing positives early sharpened the model, the pure AIS curve would sit
    below the pure random curve; if the gain is only the revealed labels, the pure
    curves overlap and only the corrected curves separate."""
    d = df[df.model == model]
    R, A = "#2a78d6", "#008300"  # random = blue, AIS (adaptive) = green
    series = [("pure, random", "model_only_pure", "random", R, "-"),
              ("pure, AIS", "model_only_pure", "ais", A, "-"),
              ("corrected, random", "model_only_corrected", "random", R, "--"),
              ("corrected, AIS", "model_only_corrected", "ais", A, "--")]
    fig, axes = plt.subplots(4, 3, figsize=(14.5, 13), facecolor=SURFACE)
    for ax, (_, m) in zip(axes.flat, meta.iterrows()):
        style(ax)
        sub = d[d.class_key == m.class_key]
        for label, est, smp, color, ls in series:
            rows = sub[(sub.estimator == est) & (sub.sampler == smp)].sort_values(
                "checkpoint")
            if len(rows):
                ax.plot(100 * rows.checkpoint, rows.mean_abs_frac_error, ls,
                        color=color, linewidth=2, marker="o", markersize=3,
                        label=label, alpha=0.9)
        ax.set_yscale("log")
        ax.set_title(f"{m.class_key}  ({m.tier}, {m.frac_empty:.0%} empty)",
                     fontsize=9, color=INK)
    for ax in axes[-1]:
        ax.set_xlabel("% labeled", fontsize=9, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("mean |frac error|", fontsize=9, color=INK2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Does directed (AIS) training sharpen the detector?  ({model})\n"
                 "(pure = the trained model's sum g alone; corrected = plus the "
                 "revealed labels.  Pure curves overlapping = training no better; "
                 "only corrected separating = the gain is revealed positives)",
                 y=1.05, fontsize=12.5, color=INK)
    fig.tight_layout()
    fig.savefig(out / f"training_ais_vs_random_{model}.png", dpi=150,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


# -------------------------------------------------------------------- timing

def fig_timing(timing_csv: Path, out: Path) -> None:
    if not timing_csv.exists():
        print(f"skip timing figure: {timing_csv} not found "
              "(reconstructed from cluster job logs; not part of this release)")
        return
    d = pd.read_csv(timing_csv).dropna(subset=["n_units"])
    d["ms_per_tile"] = 1000.0 * d.sec_per_cycle / d.n_units
    mcolor = {"frcnn_resnet50": "#0072b2", "frcnn_mobilenet": "#e69f00"}
    order = ["frcnn_resnet50", "frcnn_mobilenet"]

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.5, 5.4), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1.45, 1]})

    # left: raw per-cycle cost grows with pool size (median line per pool size)
    style(axl)
    for m in order:
        s = d[d.model == m]
        axl.scatter(s.n_units, s.sec_per_cycle, s=22, color=mcolor[m],
                    alpha=0.45, edgecolors="none", label=m, zorder=2)
        med = s.groupby("n_units").sec_per_cycle.median().sort_index()
        axl.plot(med.index, med.values, color=mcolor[m], linewidth=2,
                 marker="o", markersize=5, zorder=3)
    axl.set_xscale("log")
    axl.set_yscale("log")
    axl.set_xlabel("pool size (tiles inferred each retrain)", fontsize=9, color=INK2)
    axl.set_ylabel("seconds per retrain+inference cycle", fontsize=9, color=INK2)
    axl.set_title("Raw cost grows with pool size\n"
                  "(dots = segments, line = median per pool size)",
                  fontsize=10, color=INK)
    axl.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")

    # right: pool-size-normalized cost -> apples-to-apples between detectors
    style(axr)
    data = [d[d.model == m].ms_per_tile.to_numpy() for m in order]
    medians = {m: np.median(v) for m, v in zip(order, data)}
    ratio = medians[order[0]] / medians[order[1]]
    xhi = float(np.percentile(np.concatenate(data), 96)) * 1.2
    y = [2, 1]
    bp = axr.boxplot(data, positions=y, vert=False, widths=0.55,
                     patch_artist=True, showfliers=False, whis=(5, 95),
                     medianprops=dict(color=INK, linewidth=2))
    for patch, m in zip(bp["boxes"], order):
        patch.set(facecolor=mcolor[m], alpha=0.55, edgecolor=mcolor[m])
    for whisk in bp["whiskers"] + bp["caps"]:
        whisk.set(color=MUTED)
    for yi, m in zip(y, order):
        s = d[d.model == m]
        med = medians[m]
        hours = s.elapsed_sec.sum() / 3600.0
        axr.text(med, yi + 0.34, f"median {med:.0f} ms/tile", ha="center",
                 fontsize=8, color=INK2)
        axr.text(0.60 * xhi, yi, f"{len(s)} segments\n{hours:.0f} GPU-h total",
                 va="center", fontsize=8, color=INK2)
    axr.set_yticks(y, order, fontsize=9, color=INK2)
    axr.set_xlabel("milliseconds per pool tile per cycle", fontsize=9, color=INK2)
    axr.set_xlim(0, xhi)
    axr.set_ylim(0.4, 2.9)
    axr.set_title(f"Normalized by pool size, resnet50 costs ~{ratio:.1f}x mobilenet\n"
                  "(box = spread across segments, line = median)",
                  fontsize=10, color=INK)

    fig.suptitle("What the real detector trajectories cost\n"
                 "(one cycle = 25 label steps + 1 retrain + full-pool inference; "
                 "GPU-h = total main-loop wall-clock summed over all segments)",
                 y=1.07, fontsize=12.5, color=INK)
    fig.tight_layout()
    fig.savefig(out / "trajectory_timing.png", dpi=150, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="outputs/analysis/estimates_summary.csv")
    ap.add_argument("--timing", default="outputs/analysis/trajectory_timing.csv")
    ap.add_argument("--out-dir", default="outputs/analysis/figures/diagnostic")
    ap.add_argument("--csv-dir", default="outputs/analysis")
    ap.add_argument("--grid-budgets", nargs="+", type=float, default=None,
                    help="budget columns for winner_model_grid (fractions); "
                         "default = all checkpoints. Fewer -> a more compact "
                         "main-text figure, e.g. 0.05 0.1 0.2 0.3 0.4 0.5")
    ap.add_argument("--grid-cell-h", type=float, default=0.62,
                    help="winner_model_grid row height in inches (raise it if "
                         "model and estimator crowd inside near-tie cells)")
    ap.add_argument("--grid-width", type=float, default=None,
                    help="pin winner_model_grid width in inches to the paper's "
                         "text width (e.g. 3.3 for one column, 7.0 for two) so the font "
                         "sizes are what land on the page; default = auto")
    ap.add_argument("--exclude-estimators", nargs="+", default=[],
                    metavar="ESTIMATOR",
                    help="drop these estimators from every model's candidate "
                         "pool before picking winners, e.g. --exclude-estimators "
                         "spotdis spotais to compare against the spot-check "
                         "variants. Applies to winner_model_grid AND "
                         "detector_vs_mllm (both are built from the same "
                         "candidate pool). Valid names: dis ais spotdis spotais "
                         "difference ratio regression model_only_corrected")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_dir = Path(args.csv_dir)
    df = pd.read_csv(args.input)

    meta = class_meta(df)
    # per-model g-using bests, plus the model-agnostic uniform-MC floor as one
    # extra candidate per cell (it can win, but is never credited to a model).
    combined = pd.concat(
        [model_best(df, exclude_estimators=set(args.exclude_estimators)),
         mc_floor(df)], ignore_index=True)

    cells, ranking = _winner_tables(combined)
    readable = {MC_SENTINEL: "(uniform_monte_carlo)"}
    cells.drop(columns="checkpoint").replace(readable).to_csv(
        csv_dir / "winner_model_grid.csv", index=False)
    deg = cells[cells.winner_degenerate]
    if len(deg):
        classes = ", ".join(sorted(deg.class_key.unique()))
        print(f"NOTE: {len(deg)}/{len(cells)} winner cells are degenerate-g "
              f"(>=2 models tie exactly -> no model credited); classes: {classes}. "
              "See winner_degenerate/tied_models in winner_model_grid.csv.")
    ranking.replace(readable).to_csv(
        csv_dir / "winner_model_ranking.csv", index=False)
    fig_winner_grid(combined, meta, cells, out, budgets=args.grid_budgets,
                    cell_h=args.grid_cell_h, fig_width=args.grid_width)

    fam = _family_envelope(combined)
    fam[["class_key", "tier", "frac_empty", "pct_labeled", "family",
         "model", "estimator", "sampler", "err", "family_rank"]].replace(
        readable).to_csv(csv_dir / "detector_vs_mllm.csv", index=False)
    fig_detector_vs_mllm(fam, meta, out)

    for frcnn in ["frcnn_resnet50", "frcnn_mobilenet"]:
        fig_training_ais_vs_random(df, meta, out, model=frcnn)

    fig_timing(Path(args.timing), out)
    print(f"wrote MLLM summary figures to {out}/ and CSVs to {csv_dir}/")


if __name__ == "__main__":
    main()
