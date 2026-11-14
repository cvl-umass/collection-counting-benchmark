# Collection-Level Counting Benchmark (CLC-12)

[![arXiv](https://img.shields.io/badge/arXiv-2611.14380-b31b1b.svg)](https://arxiv.org/abs/2611.14380)

Official code for **[What Factors Matter for Collection-Level Counting with Human–AI Collaboration?](https://arxiv.org/abs/2611.14380)**

Collection-level counting estimates the total number of target objects across a large image collection by combining a model's predictions with a limited number of human-verified tiles. **CLC-12** is a benchmark of 12 such tasks, and this repository holds the pipeline used to study which choices matter: the model, the estimator, and how tiles are chosen for labeling.

## Repository layout

| Folder                  | Contents                                                                                                                                           |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `clc12/`                | CLC-12 as two small CSVs: every counting unit with its ground-truth count, and the zero-shot count each off-the-shelf model predicts for it.       |
| `counting_datasets/`    | Download script for the four source datasets and the `counting_dataset` package that indexes them. Only the two Birds photographs are stored here. |
| `counting_benchmark/`   | Tiling, the sequential labeling loop, samplers, estimators, model wrappers, replay simulation, analysis, and plotting.                             |
| `mllm_notebook_bundle/` | Notebooks for the multimodal-LLM counting pipeline: inference, count extraction, and timing and cost.                                              |

`counting_benchmark/outputs/` also holds the run records and result tables behind the paper (predicted counts, sampled tiles, and estimates), so the reported numbers can be inspected without rerunning anything. No model checkpoints are distributed: detectors are fine-tuned from public COCO weights during a run, and zero-shot models load from their public releases. Each folder has its own README with details.

## The CLC-12 tasks

Twelve tasks from four datasets, three per tier of tile emptiness, each with a fixed pool of at most about 5,000 tiles.

| Tier    | % empty tiles | Tasks                                                                |
| ------- | ------------- | -------------------------------------------------------------------- |
| Full    | < 5%          | `malaria/trophozoite`, `malaria/difficult`, `malaria/red_blood_cell` |
| Partial | 5–60%         | `birds/bird`, `dota/tennis_court`, `dota/harbor`                     |
| Sparse  | 60–90%        | `dota/ship`, `dota/plane`, `dota/small_vehicle`                      |
| Rare    | ≥ 90%         | `kenyan_wildlife/giraffe`, `dota/helicopter`, `dota/storage_tank`    |

[`clc12/clc12_tiles.csv`](clc12/) lists all 33,605 counting units with their source image, tile rectangle, and ground-truth count, so the benchmark can be used without this codebase. With the zero-shot predictions next to it, estimators and sampling strategies can be tested on CPU with no image download. The task definitions are in `counting_benchmark/outputs/selected_classes.json`. Dataset statistics, licenses, and citations are in [`counting_datasets/datasets.md`](counting_datasets/datasets.md).

## Setup

```bash
git clone https://github.com/cvl-umass/collection-counting-benchmark.git
cd collection-counting-benchmark

conda env create -f environment.yml
conda activate odcount
pip install -e counting_datasets
```

Then fetch the raw images (roughly 17 GB to download, no account needed) and build the index. The Birds photographs are already in the repository.

```bash
cd counting_datasets
bash download_data.sh              # DOTA, Kenyan Wildlife, Malaria
bash download_data.sh --check      # verify the layout before indexing
python build_index.py              # -> counting/data/index.sqlite
```

A GPU is needed for model inference and detector training. Analysis and plotting run on CPU.

## Quick start

Run one labeling trajectory with a fine-tuned detector and importance sampling:

```bash
cd counting_benchmark
python3 experiments/run.py --class-key birds/bird --model frcnn_mobilenet \
    --sampler importance --seed 0 \
    --image-pool-list data/image_pools/birds__bird__budget5000_seed0.txt \
    --score-thresh 0.5 --retrain-every 25 --max-fraction-labeled 0.5 \
    --warmup-n 50 --warmup-strategy stratified \
    --batch-size 4 --num-workers 8 --eval-batch-size 6 \
    --out-root outputs/runs
```

Or analyze the shipped results without the raw data, for example estimator win counts by budget and tier:

```bash
cd counting_benchmark
python3 analysis/estimator_wins.py
```

## Using your own model

As in [Active Measurement](https://github.com/cvl-umass/active-measurement), the framework this benchmark builds on, you plug in a detector rather than download one:

1. Subclass `CountingModel` in `counting_benchmark/models/base.py` and implement `predict` (per-tile counts and per-box scores). For a trainable model, also override `fit`, `state_dict`, and `load_state_dict`.
2. Register it in `build_model` in `counting_benchmark/models/__init__.py` and add its name to the `--model` choices in `counting_benchmark/experiments/run.py`.
3. Run `experiments/run.py --model <your_model>` as in the quick start.

## Reproducing the paper

[`counting_benchmark/README.md`](counting_benchmark/README.md#reproducing) gives the full pipeline command by command, and lists the large intermediate caches that are omitted here and how to rebuild them. The benchmark results the paper reports can be read from `counting_benchmark/outputs/analysis/`.

Some reported numbers have sources outside this repository. GPU-hours and inference costs were measured from cluster job logs and API usage records. The aerial pronghorn case study is not included, because the data-sharing terms of that collaboration are not yet cleared.

## Citation

```bibtex
@misc{jeong2026factorsmattercollectionlevelcounting,
      title={What Factors Matter for Collection-Level Counting with Human--{AI} Collaboration?},
      author={Seoyun Jeong and Logan Lawrence and Owen L. Martin and Benjamin Koger and Daniel Sheldon and Subhransu Maji},
      year={2026},
      eprint={2611.14380},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2611.14380},
}
```

Please also cite the original dataset papers listed in [`counting_datasets/datasets.md`](counting_datasets/datasets.md). Each dataset is distributed under its original license, so check those terms before using the data.
