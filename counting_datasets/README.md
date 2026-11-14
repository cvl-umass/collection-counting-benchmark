# Counting datasets — raw data loading for the benchmark

This directory turns four heterogeneous counting datasets into one SQLite index with a common schema, so the benchmark can query any class by a single `dataset/class` key. It is a data-loading layer only: no models, no training, no evaluation. Those live in `../counting_benchmark`.

The benchmark uses exactly three entry points from here — `CountingDatasetIndex`, `load_class()`, and `image_metadata()` — all shown under [Querying the index](#querying-the-index).

## Datasets

The 12 benchmark tasks are drawn from four datasets: **Birds** (Tree Swallows flock), **DOTA v1.5**, **Kenyan Wildlife Aerial Survey**, and **Malaria** infected blood smears. Per-dataset statistics, licenses, and citations are in [`datasets.md`](datasets.md); each is distributed under its original license. The two Birds photographs are stored in this repository; the other three datasets are downloaded from their original sources.

## Setup

```bash
conda env create -f ../environment.yml   # at the repository root
conda activate odcount
pip install -e .                         # from this directory
```

Python ≥ 3.10. `pip install -e .` alone suffices for the data layer; the conda environment is needed only for the benchmark's GPU stack.

## Building the index

Raw images for DOTA, Kenyan Wildlife, and Malaria are not included (roughly 17 GB to download). Fetch them, then build:

```bash
bash download_data.sh              # everything; or any of: dota kenyan malaria birds
bash download_data.sh --check      # verify the layout, download nothing
python build_index.py              # -> counting/data/index.sqlite
```

`download_data.sh` is idempotent and needs no account or API token, so it is safe to re-run after an interrupted transfer. **Run `--check` before building**: `build_index.py` produces partial counts rather than failing loudly when a path is missing. Edit the `DATASETS` dict at the top of `build_index.py` to build a subset.

Birds needs no download: its two source `.tif` photographs, point annotations, and the pre-computed Otsu pseudo-bbox cache (`raw/birds/pseudo_bboxes_cache.json`) are part of this repository, and the script only verifies the photographs' checksums. To rebuild the cache from scratch instead, run `python raw/birds/generate_bird_bbox_cache.py` before building.

## Querying the index

The benchmark works class-by-class, so it only ever uses the class-centric view: iterate the images containing one semantic class.

```python
from counting_dataset import CountingDatasetIndex

index = CountingDatasetIndex("counting/data")   # directory holding index.sqlite
ds = index.load_class(class_key="birds/bird")

img, target = ds[0]
print(target["count"], target["instances"])

meta = ds.image_metadata()   # per-image id/path/size/count, no annotations loaded
```

`image_metadata()` is what the tiling stage reads: it returns lightweight per-image records so image pools can be built without decoding pixels.

**Counts come only from `role="instance"` annotations.** Alternative geometries for the same objects are kept alongside under `target["aux"]`, keyed by role — `obb` for DOTA's oriented boxes, `point` for the original Birds annotator points — and never contribute to a count. Each adapter picks the canonical geometry most usable downstream, which is axis-aligned boxes for every dataset here.

`usage_example.py` is a longer runnable tour of the API, including the image-centric `load_dataset()` view that the benchmark does not use.

## Layout

```
counting/src/         library code (adapters, index builder, query API)
counting/data/        built SQLite index (generated, not included)
raw/                  raw datasets (downloaded, except Birds)
raw/birds/            source photographs, point labels, bbox cache, pre-processing scripts
build_index.py        build script with per-dataset toggles
download_data.sh      fetch and arrange raw datasets
datasets.md           per-dataset statistics, licenses, citations
usage_example.py      end-to-end API demo
```
