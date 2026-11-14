# Integrated Datasets

The four datasets behind the 12 benchmark tasks, with statistics, licensing, and references to the original sources. Each is read by a dataset-specific adapter and normalized into the shared schema, and the statistics below reflect the indexed version of each dataset.

## Summary Table

| Dataset                       | Domain          | # Images | Annotation Type (canonical → auxiliary) | # Classes | Images per Class (mean) | Total Objects | Objects per Image (mean) | Multi-class Images | License         |
| ----------------------------- | --------------- | -------- | --------------------------------------- | --------- | ----------------------- | ------------- | ------------------------ | ------------------ | --------------- |
| Birds (Tree Swallows Flock)   | Wildlife        | 2        | HBB (pseudo-bbox) → points              | 1         | 2                       | 18,696        | 9,348                    | No                 | MIT             |
| DOTA v1.5                     | Aerial urban    | 1,869    | HBB → OBB                               | 16        | 343.38                  | 280,196       | 149.92                   | Yes                | Academic only   |
| Kenyan Wildlife Aerial Survey | Aerial wildlife | 561      | HBB                                     | 3         | 197.67                  | 4,304         | 7.67                     | Yes                | CC0             |
| Malaria Infected Blood Smears | Microscopy      | 1,328    | HBB                                     | 7         | 434.00                  | 86,035        | 64.79                    | Yes                | CC BY-NC-SA 3.0 |

### Notes on interpretation

- **# Images** counts images in the index. For Birds this is 2 (the original .tif files); tiling for model training is handled downstream.
- **Annotation Type** lists the canonical (`role="instance"`) format first, then auxiliary formats after `→`. Only canonical annotations contribute to object counts.
- **Images per Class (mean)** is not meaningful for single-class datasets and hence omitted.
- **Total Objects** and **Objects / Image** are computed using instance-only annotations (`role="instance"`). See the note on annotation roles in [`README.md`](README.md#querying-the-index).

## Dataset Descriptions

### 🐦 Birds (Tree Swallows Flock)

This dataset consists of two large high-resolution ground-level photographs of tree swallow flocks, captured in two distinct scenes: an open sky background ("sky", `_DSC5214.tif`) and a dense reed-bed background ("reeds", `_DSC5295.tif`). Labels were created at tile level using the VGG Image Annotator (VIA) tool and record each bird as a point click.

**The index stores the original .tif files as the unit of analysis**, not the tiles. Tile-local point annotations are translated to global image coordinates during index build. Per-tile Otsu pseudo-bounding boxes (computed by `generate_bird_bbox_cache.py`) are stored as `role="instance"` HBB annotations — the canonical counted format — while the original point annotations are stored as `role="point"` auxiliary annotations.

Tiling for model training is handled downstream by the benchmark, consistent with the other large-image datasets here.

The two scenes can be loaded independently via `meta_filter={"scene": "sky"}` or `meta_filter={"scene": "reeds"}`, or together by omitting the filter.

The pseudo-bbox cache (`raw/birds/pseudo_bboxes_cache.json`) ships with this release, so no pre-processing is needed. Regenerating it from the source `.tif` files instead requires one run of `python raw/birds/generate_bird_bbox_cache.py` before building the index.

- **Domain:** Wildlife monitoring
- **Source images:** 2 high-resolution TIF photographs
- **Tile sizes used during annotation:** 200×200 px (sky scene), 160×160 px (reeds scene)
- **Scenes:** 2 — `sky` and `reeds`
- **Annotations (canonical):** HBB pseudo-bboxes derived from Otsu thresholding (`role="instance"`, `source=GENERATED`)
- **Annotations (auxiliary):** Point annotations from VIA annotators (`role="point"`, `source=ORIGINAL`)
- **Classes:** 1 (bird)
- **Images in index:** 2
- **License:** MIT

**Reference:** Hamilton, Max, Jinlin Lai, Wenlong Zhao, Subhransu Maji, and Daniel Sheldon. "Active Measurement: Efficient Estimation at Scale." _Advances in Neural Information Processing Systems_ 38 (2025): 145769–145807.

The raw `.tif` photographs are included in this repository (`raw/birds/`) under the terms of their MIT license. They are lossless deflate-compressed TIFFs, pixel-identical to the originals, and `download_data.sh` verifies their SHA-256 checksums.

### ✈️ DOTA v1.5

DOTA is a dense, large-scale detection dataset featuring extreme object counts, heavy class imbalance, and frequent multi-class co-occurrence. **HBBs are the canonical counting format** (`role="instance"`), with OBBs stored as auxiliary alternative geometry (`role="obb"`). This reflects the fact that most downstream counting models operate on axis-aligned boxes. Only the train and validation sets of DOTA v1.5 are indexed, since the test-set annotations are not public.

- **Domain:** Aerial urban scenes
- **Images:** 1,869
- **Annotations (canonical):** Axis-aligned bounding boxes, HBB (`role="instance"`)
- **Annotations (auxiliary):** Oriented bounding boxes, OBB (`role="obb"`)
- **Classes:** 16 object categories (harbor, small vehicle, baseball diamond, etc.)
- **Image resolution:** ~420 × 350 to ~12,000 × 5,000 pixels
- **Objects per image:** Mean ~150, Max >10,000
- **License:** No named license. The dataset page states: "All images and their associated annotations in DOTA-v1.5 can be used for academic purposes only, but any commercial use is prohibited."

**Reference:** J. Ding, N. Xue, Y. Long, G. -S. Xia and Q. Lu, "Learning RoI Transformer for Oriented Object Detection in Aerial Images," 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Long Beach, CA, USA, 2019, pp. 2844-2853, doi: 10.1109/CVPR.2019.00296. [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8953881) [[download dataset]](https://captain-whu.github.io/DOTA/dataset.html)

### 🦒 Kenyan Wildlife Aerial Survey

This dataset is provided in COCO format and contains multi-class aerial scenes of wildlife. It is representative of real-world aerial survey conditions.

- **Domain:** Aerial wildlife monitoring
- **Images:** 561
- **Annotations:** Axis-aligned bounding boxes (HBB)
- **Classes:** 3 (elephants, giraffes, zebras)
- **Image resolution:** ~5,000 × 3,000 pixels
- **Objects per image:** Mean ~7.7, Max 108
- **License:** CC0 (Public Domain)

**Reference:** Eikelboom JAJ, Wind J, van de Ven E, et al. Improving the precision and accuracy of animal population estimates with aerial image object detection. Methods Ecol Evol. 2019; 10: 1875–1887. https://doi.org/10.1111/2041-210X.13277 [[pdf]](https://besjournals.onlinelibrary.wiley.com/doi/epdf/10.1111/2041-210X.13277) [[download dataset]](https://www.kaggle.com/datasets/davidrpugh/kenyan-wildlife-aerial-survey)

### 🦠 Malaria Infected Human Blood Smears

This dataset features dense object distributions and significant class imbalance. It is well suited for evaluating counting methods in high-density microscopy settings.

- **Domain:** Microscopy / biomedical imaging
- **Images:** 1,328
- **Annotations:** Axis-aligned bounding boxes (HBB)
- **Classes:** 7 cell types
- **Image resolution:** ~1,600 × 1,200 to ~1,900 × 1,400 pixels
- **Objects per image:** Mean ~65, Max 223
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 3.0

**Reference:** Image set BBBC041v1 from the Broad Bioimage Benchmark Collection [[article]](https://www.nature.com/articles/nmeth.2083) [[download dataset]](https://bbbc.broadinstitute.org/BBBC041)
