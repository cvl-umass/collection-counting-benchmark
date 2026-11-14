# CLC-12 as CSV

The benchmark's counting units and their ground-truth counts, in a form that needs none of the code in this repository. There are 33,605 units across the 12 tasks. No imagery is included.

## `clc12_tiles.csv`

One row per counting unit. A unit is a tile cropped from a source image, or the whole image when it is small enough to be left untiled.

| Column                           | Meaning                                                                                      |
| -------------------------------- | -------------------------------------------------------------------------------------------- |
| `task`                           | Task key, `<dataset>/<class>`, e.g. `dota/ship`.                                             |
| `dataset`, `class`               | The two halves of `task`.                                                                    |
| `tier`                           | Occupancy tier by fraction of empty units: `full`, `partial`, `sparse`, or `rare`.           |
| `tile_index`                     | Unit index within the task, `0..N-1`. `(task, tile_index)` is the unique key.                |
| `image_id`                       | Stable content hash of the source image.                                                     |
| `image_path`                     | Source image, relative to `counting_datasets/` once `download_data.sh` has been run.         |
| `x0`, `y0`, `width`, `height`    | The unit's rectangle in source-image pixels, with `(x0, y0)` the top-left corner.            |
| `is_tiled`                       | `1` if the unit is a crop of a larger image, `0` if it is the whole image.                   |
| `gt_count`                       | Ground-truth number of objects of the task's class in the unit.                              |

The true collection total of a task is the sum of its `gt_count` column.

## `clc12_zeroshot_predictions.csv`

The count each off-the-shelf model predicts for each unit, with no fine-tuning. It joins to `clc12_tiles.csv` on `(task, tile_index)`. There is one column per model: three open-vocabulary detectors (`grounding_dino`, `owlv2`, `florence2`, counted at score threshold 0.5) and eight multimodal LLMs.

Together the two files are enough to evaluate a sampling strategy or an estimator on CPU, without downloading any images:

```python
import pandas as pd

tiles = pd.read_csv("clc12_tiles.csv")
preds = pd.read_csv("clc12_zeroshot_predictions.csv")
df = tiles.merge(preds, on=["task", "tile_index"])

ship = df[df.task == "dota/ship"]
print(ship.gt_count.sum(), ship.owlv2.sum())   # true total vs. model-only estimate
```

To get the pixels for a unit, download the datasets (see `../counting_datasets/`) and crop:

```python
from PIL import Image

r = tiles.iloc[0]
unit = Image.open(f"../counting_datasets/{r.image_path}").crop(
    (r.x0, r.y0, r.x0 + r.width, r.y0 + r.height)
)
```

Predictions from the fine-tuned detectors change as labeling proceeds, so they are not in these files. They are in the run records under `../counting_benchmark/outputs/runs/`.

Both files are regenerated from those run records by `python3 scripts/export_clc12_tiles.py`, run from `../counting_benchmark/`.
