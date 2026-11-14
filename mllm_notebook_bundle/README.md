# MLLM counting notebooks

The multimodal-LLM counting pipeline of the paper as small, runnable notebooks for a single GPU node: inference, count extraction, frozen-store packaging, and timing and cost.

## Start here

```bash
jupyter lab
```

The notebooks were run in their own environment, separate from `../environment.yml`. Local inference needs vLLM or Transformers, which that file does not pin; the analysis notebooks need only pandas and matplotlib.

Open the notebooks in numeric order. Every notebook repeats one clearly marked parameter cell at the top. The default live-inference switches are safe previews:

- Set `COUNTING_DATA_ROOT` to the directory containing `raw/` and `counting/data/` (by default `../counting_datasets`) before running local MLLM inference.
- Set `RUN_INFERENCE=True` in `01_open_source_vllm_inference.ipynb` after choosing one model. It assumes one visible GPU and records direct wall-clock timing.
- Set `RUN_REQUESTS=True` in `02_proprietary_api_inference.ipynb` only when paid calls are intended. It reads the OpenRouter key from the `OPENROUTER_API_KEY` environment variable; no key is printed or saved.
- The analysis, store-building, reference-summary, and export notebooks run from the repository's existing outputs. When opened outside the repository, the included sample and reference files keep the visual cells runnable.

The split is by pipeline stage: setup, local inference, API inference, answer cleanup, frozen-store packaging, timing and cost, exports, and reference data.

## Reference count data

The full per-tile prediction exports (about 111 MB) are not included. `07_reference_count_summary.ipynb` reads them from

```text
outputs/runs/open_source_all_tiles_current.csv
outputs/runs/proprietary_all_tiles_current.csv
```

and derives `data/reference_summary.csv`. Each row is a model × task summary containing row count, ground-truth mean/median/spread, predicted mean/median/spread, MAE, exact-match rate, answer-valid rate, and parse-failure rate. The notebook also writes a deterministic four-rows-per-model/task sample (384 rows total) preserving the raw answer and parse fields.

These are reference values from completed runs, not ground truth for the model itself. They are a practical comparison target for a new run; differences should be investigated by row key, prompt/model version, and parse status.

The per-tile counts themselves are available in packaged form: as `*__g_fixed` stores under `../counting_benchmark/outputs/runs/`, and as one table in `../clc12/clc12_zeroshot_predictions.csv`.

## File-by-file map

| File | Purpose |
| --- | --- |
| `00_setup_and_config.ipynb` | Shows the configurable roots, task list, model defaults, and the notebook-stage flow. |
| `01_open_source_vllm_inference.ipynb` | Two-pass local inference (free-form count, then integer choice), resumable JSONL labels, optional direct timing, and prediction/truth plots. Supports the five open-source model configurations. |
| `02_proprietary_api_inference.ipynb` | OpenRouter request construction, staged count/extraction calls, bounded local concurrency, incremental JSONL responses, and API error plots. The included examples are the safe default; point the request builder at the full manifest for a full pool. |
| `03_extract_and_verify_answers.ipynb` | Final-answer extraction, range fallback, literal verification, strict `{"value": integer}` recovery validation, and extraction-failure visualizations. |
| `04_build_g_fixed_stores.ipynb` | Alignment-checked packaging of per-tile `final_value` predictions into `g_fixed` stores using the canonical manifest/config and `snapshots.npz` format. |
| `05_timing_and_cost.ipynb` | Uses direct notebook wall-clock timing for local models and direct API timing for proprietary models. It reports missing timing rather than guessing. |
| `06_exports.ipynb` | Compact CSV exports from open-source JSONL caches and proprietary interpreted records, with a preview fallback and error-distribution plot. |
| `07_reference_count_summary.ipynb` | Derives the compact reference summary and raw samples from the full source exports, then visualizes model/task count ranges and agreement. |
| `data/reference_summary.csv` | Compact 96-row model × task reference table derived from the full source exports. |
| `data/reference_raw_sample.csv` | 384-row combined raw sample with the nine source fields plus `reference_source`. |
| `data/open_source_reference_sample.csv` | 240-row source-separated view of the raw sample. |
| `data/proprietary_reference_sample.csv` | 144-row source-separated view of the raw sample. |
| `data/example_tiles.csv` | Six-tile manifest for the included visual examples. |
| `assets/*.png` | Six small tile images used by the API and visualization examples. Each remains under the license of its source dataset (see `../counting_datasets/datasets.md`). |

## Portability and safety notes

No notebook submits cluster jobs or reads scheduler telemetry. The only GPU assumption is that the local inference notebook runs on a node with one visible CUDA device. API keys and bearer headers are runtime-only. Notebook outputs are cleared; figures are produced when cells run.

The included samples are enough for the example plots, but not enough to rebuild a full `g_fixed` store or run the full local or API stages. Those need the raw counting datasets, the tile manifests, model weights or network access, and (for OpenRouter) an API key; the notebooks fail with a clear missing-input error rather than fabricating results.
