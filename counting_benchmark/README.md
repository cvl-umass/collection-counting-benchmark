# Collection-level counting with human–AI collaboration — code and data

Tiling, the sequential labeling loop, the estimators, replay simulation, and analysis. Its companion `../counting_datasets` loads the raw imagery and builds the dataset index; the two are used together.

## Setup

```bash
conda env create -f ../environment.yml   # at the repository root
conda activate odcount
```

The two directories are expected side by side, as laid out in this repository:

```
<parent>/counting_benchmark/     # this one
<parent>/counting_datasets/      # the counting_dataset package + raw imagery
```

Scripts that read imagery resolve `../counting_datasets` by default. Override with `COUNTING_DATA_ROOT`, or per invocation with `--index-root` and `--raw-root`.

## Layout

```
estimators/    model-only, Monte Carlo, incremental difference/ratio/regression,
               AIS/DIS, spotAIS/spotDIS, and the LURE/COMB combining weights
experiments/   run.py (the sequential labeling loop), the recorder, the sampler,
               the DINOv3 emptiness filter h, and zero-shot g computation
models/        Faster R-CNN (ResNet-50, MobileNetV3) and the frozen OVDs
data/          tiling, image-pool selection, feature cache, frozen pool lists
analysis/      trajectory loading, replay simulation, estimator application,
               estimate derivation, win counts, pilot selection, CI coverage
scripts/       pool construction, class selection, feature extraction, MLLM
               g-store packaging, simulation collection, replay validation
plot/          the figures
docs/          on-disk formats for the run artifacts and the tiled dataset
outputs/       configuration, run artifacts, and analysis results
```

## Naming

A few names in the code differ from the paper.

| Paper                          | Code                                               |
| ------------------------------ | -------------------------------------------------- |
| unit                           | tile (`tile_index`)                                |
| uniform sampling               | `--sampler random`                                 |
| AIS / DIS                      | `--sampler importance` (fine-tuned / frozen model) |
| spotAIS / spotDIS              | `--sampler spotcheck`                              |
| spot-check fraction λ          | `--alpha`                                          |
| threshold tolerance φ          | `--h-fpr-tol`                                      |
| proposal floor ε               | `--min-count-per-tile`                             |
| model-only baseline            | `model_only_corrected`                             |
| frozen predictions g₀          | `*__g_fixed` stores                                |

## What is included

**Run artifacts** (`outputs/runs/<dataset>/<class>/`) are the low-level record of every run: 492 in total, being 360 fine-tuned detector trajectories (2 backbones × 12 tasks × 3 samplers × 5 seeds) and 132 frozen zero-shot stores (11 models × 12 tasks: 3 OVDs and 8 MLLMs). Each run holds `config.json` and `snapshots.npz`, and detector runs add `steps.csv` and `warmup.csv`. `manifest.csv` is stored once per task, since every run of a task shares it. Formats are documented in `docs/run_artifacts.md`.

Two repackaging choices apply, both lossless and verified to leave every estimator's output unchanged: predicted counts `g` are stored as `uint16` because they are integer detection counts, and the diagnostic `p_empty` score array is dropped in favor of the boolean `h_empty` that the sampler actually reads.

**Analysis results** (`outputs/analysis/`) hold the benchmark results the paper reports: `estimates_summary.csv` (mean/sd/quantiles of fractional error per model × task × sampler × estimator × budget), `estimates_perstep.parquet` (the same ensemble at every step, projected to the columns the log-log figure uses), `ci_coverage.csv`, `ci_spike_regimes.csv`, and the replay-validation tables under `replay_validation/` (`shift.csv` holds the paper's Shift column). A few reported numbers come from sources outside this repository: GPU-hours and inference costs were measured from cluster job logs and API usage records, and the pronghorn case study uses data that is not released.

**Configuration** covers `outputs/selected_classes.json` (the 12 benchmark tasks), `outputs/suggested_tile_sizes.json` (per-class tile geometry), `outputs/regimes_v2.csv` (per-class occupancy statistics), and `data/image_pools/*.txt` (the frozen admitted-image lists).

## What is not included, and why

**Intermediate simulation caches.** `outputs/analysis/perstep_Fhat/` holds the 1,000-trajectory per-step matrices at roughly 5 GB. It and its reductions are regenerable from the run artifacts by stage 5 below, and simulation seeds are deterministic, so a rebuild reproduces the shipped results exactly.

**The DINOv3 feature cache** (~800 MB for the benchmark classes) is needed only to refit the emptiness filter `h` live, which affects the *zero-shot* spot-checked cells; the detector spot-check replays read the cached `h_empty` and need nothing. `scripts/extract_dino_features.py` rebuilds it. The logistic probe is platform-sensitive, so those particular cells are not expected to reproduce bit-for-bit across machines.

**Raw MLLM prediction CSVs and detector weights.** The MLLM per-tile predictions ship in derived form as the `*__g_fixed` stores, and `scripts/build_mllm_gstores.py` documents how they were packaged from the raw inference output. Detector checkpoints and per-box detection dumps are omitted for size; nothing in the reported results depends on them.

**Cluster job scripts.** Runs were executed as scheduler array jobs, whose submission wrappers are cluster-specific. The underlying invocations are given below instead.

## Reproducing

Stages 1–4 need the raw imagery and a GPU. Stages 5–7 run from what ships here, with one caveat: stage 5 rebuilds the large per-step caches, and three later commands read those caches rather than the shipped CSVs, so they are marked *(needs stage 5)*. Everything else in stages 6–7 runs directly on the included artifacts, and the results those three would rebuild ship as `ci_coverage.csv`, `ci_spike_regimes.csv`, and the tables under `replay_validation/`, so no rerun is needed just to read the reported numbers.

**1. Tile geometry and image pools**

```bash
python3 scripts/build_image_pools.py --tile-budget 5000
python3 scripts/analyze_counting_regimes.py      # -> outputs/regimes_v2.csv
python3 scripts/freeze_selected_classes.py       # -> outputs/selected_classes.json
```

**2. DINOv3 features** — one call per class key, for the spot-check filter.

```bash
python3 scripts/extract_dino_features.py --class-key birds/bird \
    --dino-input-max-side 1024 --num-workers 8
```

**3. Frozen zero-shot predictions** — one call per model × class, with `--model` in `{grounding_dino, owlv2, florence2}`.

```bash
python3 experiments/compute_zeroshot_g.py --class-key birds/bird --model owlv2 \
    --image-pool-list data/image_pools/birds__bird__budget5000_seed0.txt \
    --score-thresh 0.5 --batch-size 6 --num-workers 8 --out-root outputs/runs
```

MLLM predictions were produced separately and packaged with `python3 scripts/build_mllm_gstores.py`.

**4. Detector trajectories** — 360 runs, with `--model` in `{frcnn_mobilenet, frcnn_resnet50}`, `--sampler` in `{random, importance, spotcheck}`, and `--seed` in `0..4`, over the 12 class keys.

```bash
python3 experiments/run.py --class-key birds/bird --model frcnn_mobilenet \
    --sampler importance --seed 0 \
    --image-pool-list data/image_pools/birds__bird__budget5000_seed0.txt \
    --score-thresh 0.5 --retrain-every 25 --max-fraction-labeled 0.5 \
    --warmup-n 50 --warmup-strategy stratified \
    --batch-size 4 --num-workers 8 --eval-batch-size 6 \
    --out-root outputs/runs
```

Add `--alpha 0.05 --routing trust_h --features-root data/dinov3` for `--sampler spotcheck`. Runs resume from their last checkpoint if re-invoked; `--fresh` restarts one.

**5. Replay to 1,000 trajectories per setting, then derive estimates**

```bash
python3 scripts/collect_sims.py --jobs 8              # -> outputs/analysis/perstep_Fhat/
python3 analysis/derive_estimates.py --weighting auto # -> checkpoints/ + perstep/
python3 scripts/collect_sims.py --merge               # -> estimates_*.parquet
python3 scripts/summarize_checkpoints.py              # -> estimates_summary.csv
```

`--weighting auto` applies COMB where the detector adapts and LURE elsewhere, which is what the paper reports.

**6. Analysis**

```bash
python3 analysis/estimator_wins.py                    # win counts by budget and tier
python3 analysis/estimator_wins.py --scope offtheshelf
python3 analysis/pilot_model_selection.py             # pilot proxy comparison
python3 analysis/pilot_proxy_table.py                 # pilot proxy table (10,000 pilots)
python3 scripts/path_agreement_stats.py               # cross-seed model-path divergence
python3 scripts/model_path_invariance.py              # optional per-snapshot path-agreement tables
python3 analysis/ci_coverage.py                       # (needs stage 5) interval coverage by tier
python3 analysis/ci_spike_regimes.py                  # (needs stage 5) spot-check under-coverage
python3 scripts/replay_validation.py                  # (needs stage 5) replay validation tables
```

**7. Figures**

Run `plot/plot_mllm_summary.py` first, since it writes `winner_model_grid.csv` that three of the others read.

```bash
python3 plot/plot_mllm_summary.py
python3 plot/plot_fixed_and_crossover.py              # estimator curves + family crossover
python3 plot/plot_results_combined.py                 # results overview
python3 plot/plot_model_wins.py                       # per-model win counts
python3 plot/plot_mllm_disagreement.py                # zero-shot total disagreement
python3 scripts/make_replay_validation_figs.py        # replay landing positions
```
