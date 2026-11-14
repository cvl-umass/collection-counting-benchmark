# `run.py` artifacts

Files written to each run directory (`outputs/runs/<dataset>/<class>/<run_id>/`) by `ExperimentRecorder`. The loop records raw signals only; every estimator is derived post-hoc from these.

| File | Content |
|------|---------|
| `config.json` | Full args + `run_id`, realized tiling (`tile_size`, `tile_threshold`, `max_whole_image_side`), `F_true`, `N`, `n_images`. |
| `manifest.csv` | One row per unit: `tile_index`, `img_idx`, `image_id`, `image_path`, `x0`, `y0`, `pw`, `ph`, `is_tiled`, `gt_count`. `gt_count` is `f(s)` for **all** units. |
| `warmup.csv` | Warmup picks: `s_t`, `f_st`, `g_st` (post-warmup model count). |
| `steps.csv` | One row per step `t`: `s_t`, `f_st`, `q_t_st`, `snapshot_id`, `retrain_step`. The as-run trajectory. |
| `snapshots.npz` | `g` `(S, N)` hard counts at `score_thresh` for all units per snapshot; `snapshot_steps` `(S,)` the `t` of each snapshot. Spot-check runs add the boolean `h_empty` and its `h_steps`. |

Notes:
- `manifest.csv` is stored once per task directory rather than once per run, since the unit pool is frozen before any model runs and every run of a task shares it.
- `tile_index` is the canonical unit index; it equals `s_t` everywhere.
- `image_id` is a stable content hash (survives pool caps / reseeds); `img_idx` is just a positional index into the base pool.
- `snapshot_id` indexes `snapshots.npz`; `snapshot_steps[id]` maps it to the step `t` it was taken at (`{0} ∪ {retrain steps}`).
- `g` is stored as `uint16`, a lossless recast of the integer detection counts.

Two files the loop also writes are omitted from this release for size, and nothing in the reported results reads them: `detections.npz` (long-format per-box confidences above `capture_score_floor`, which allow any soft or re-thresholded `g'` to be reconstructed post-hoc) and `checkpoint_t*.pt` (model weights plus RNG state for exact resume, most recent only).
