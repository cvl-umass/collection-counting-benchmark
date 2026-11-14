#!/usr/bin/env bash
#
# Download and arrange the raw datasets required to build the counting index.
#
# Usage:
#   bash download_data.sh              # all datasets
#   bash download_data.sh dota         # one or more of: dota kenyan malaria birds
#   bash download_data.sh --check      # verify layout only, download nothing
#
# Everything lands under raw/ in the layout the adapters expect.  Each step is
# idempotent: a dataset that is already in place is skipped, so the script can be
# re-run safely after a failed or interrupted download.
#
# Total download size is roughly 17 GB. Expect this to take a while.
#
# Requirements
#   unzip, curl            — always
#   gdown                  — DOTA only   (pip install gdown; already in environment.yml)
#
# No account or API token is needed for any dataset.  Birds needs no download:
# its two photographs and annotations are part of this repository, under
# raw/birds/; see datasets.md.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults to raw/ beside this script.  Override to download into a different
# location -- a scratch filesystem, say, or an existing dataset directory you
# want to verify:
#     COUNTING_RAW_ROOT=/path/to/raw bash download_data.sh --check
RAW_ROOT="${COUNTING_RAW_ROOT:-${HERE}/raw}"
STAGE="${RAW_ROOT}/.staging"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

log()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '\033[33m    WARNING: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "'$1' not found. $2"
}

# Move every file matching a pattern out of a staging tree into a flat target
# directory, regardless of how the archive nested them.  This keeps the script
# working even if the upstream zips change their internal folder names.
#
# AppleDouble sidecars (._name.png, shipped inside archives zipped on macOS) match
# the same globs as real images but are resource forks, not data.  They are
# skipped here so they neither clutter the dataset nor inflate the file counts.
harvest() {
  local src_dir="$1" pattern="$2" dest_dir="$3"
  mkdir -p "$dest_dir"
  find "$src_dir" -type f -name "$pattern" ! -name '._*' -exec mv -n {} "$dest_dir/" \;
}

count_files() {
  local dir="$1" pattern="$2"
  [[ -d "$dir" ]] || { echo 0; return; }
  find "$dir" -type f -name "$pattern" ! -name '._*' | wc -l | tr -d ' '
}

# Multi-GB fetch over links that may be slow, flaky, or firewalled.
#
#   -C -                     resume a partially downloaded file instead of
#                            restarting from zero
#   --speed-limit/--speed-time
#                            give up if throughput stays under 1 KB/s for two
#                            minutes, rather than hanging forever at 0 B/s
#   CURL_EXTRA_OPTS          escape hatch for site-specific needs, e.g.
#                            CURL_EXTRA_OPTS="-4" to force IPv4, or "-x <proxy>"
fetch_resumable() {
  local outfile="$1" url="$2"
  # shellcheck disable=SC2086
  curl -fL -C - \
       --retry 5 --retry-delay 10 --retry-connrefused \
       --connect-timeout 30 \
       --speed-limit 1024 --speed-time 120 \
       --progress-bar \
       ${CURL_EXTRA_OPTS:-} \
       -o "$outfile" "$url"
}

# ---------------------------------------------------------------------------
# DOTA v1.5  ->  raw/dota_v1-5/{train,val}/{images,annotation/{hbb,obb}}
# ---------------------------------------------------------------------------
#
# DOTA is distributed as Google Drive zips, which is why this needs gdown rather
# than a plain curl.  Images arrive split across part1..part3 for train and a
# single part1 for val; HBB and OBB labels arrive as separate archives whose .txt
# files share the same basenames, so each is staged separately before harvesting.

DOTA_DIR="${RAW_ROOT}/dota_v1-5"

dota_fetch() {
  local outfile="$1" url="$2"
  if [[ -s "$outfile" ]]; then
    info "exists, skipping: $(basename "$outfile")"
    return 0
  fi
  info "downloading $(basename "$outfile")"
  gdown --fuzzy --output "$outfile" "$url"
}

download_dota() {
  if [[ -d "${DOTA_DIR}/train/images" && -d "${DOTA_DIR}/val/images" ]]; then
    log "DOTA v1.5 — already present, skipping"
    return 0
  fi

  log "DOTA v1.5"
  need_cmd gdown "Install with: pip install gdown"
  need_cmd unzip "Install unzip via your package manager."

  local zips="${STAGE}/dota_zips"
  mkdir -p "$zips"

  dota_fetch "${zips}/train_part1.zip" "https://drive.google.com/file/d/1BlaGYNNEKGmT6OjZjsJ8HoUYrTTmFcO2/view"
  dota_fetch "${zips}/train_part2.zip" "https://drive.google.com/file/d/1JBWCHdyZOd9ULX0ng5C9haAt3FMPXa3v/view"
  dota_fetch "${zips}/train_part3.zip" "https://drive.google.com/file/d/1pEmwJtugIWhiwgBqOtplNUtTG2T454zn/view"
  dota_fetch "${zips}/train_hbb.zip"   "https://drive.google.com/file/d/1-vLCMhIW9CV2cmCPPBbDR9_hdecf5bLb/view"
  dota_fetch "${zips}/train_obb.zip"   "https://drive.google.com/file/d/12uPWoADKggo9HGaqGh2qOmcXXn-zKjeX/view"
  dota_fetch "${zips}/val_part1.zip"   "https://drive.google.com/file/d/1uCCCFhFQOJLfjBpcL5MC0DHJ9lgOaXWP/view"
  dota_fetch "${zips}/val_hbb.zip"     "https://drive.google.com/file/d/1XDWNx3FkH9layL8jVUkEHJ_-CY8K4zse/view"
  dota_fetch "${zips}/val_obb.zip"     "https://drive.google.com/file/d/1FkCSOCy4ieNg1UZj1-Irfw6-Jgqa37cC/view"

  info "extracting and arranging (this takes several minutes)"
  local split kind
  for split in train val; do
    # images
    rm -rf "${STAGE}/dota_${split}_img"
    mkdir -p "${STAGE}/dota_${split}_img"
    for z in "${zips}/${split}_part"*.zip; do
      [[ -e "$z" ]] || continue
      unzip -q -o "$z" -d "${STAGE}/dota_${split}_img"
    done
    harvest "${STAGE}/dota_${split}_img" '*.png' "${DOTA_DIR}/${split}/images"

    # annotations — hbb and obb staged apart, their .txt basenames collide
    for kind in hbb obb; do
      rm -rf "${STAGE}/dota_${split}_${kind}"
      mkdir -p "${STAGE}/dota_${split}_${kind}"
      unzip -q -o "${zips}/${split}_${kind}.zip" -d "${STAGE}/dota_${split}_${kind}"
      harvest "${STAGE}/dota_${split}_${kind}" '*.txt' "${DOTA_DIR}/${split}/annotation/${kind}"
    done
  done

  info "train: $(count_files "${DOTA_DIR}/train/images" '*.png') images, $(count_files "${DOTA_DIR}/train/annotation/hbb" '*.txt') hbb, $(count_files "${DOTA_DIR}/train/annotation/obb" '*.txt') obb"
  info "val:   $(count_files "${DOTA_DIR}/val/images" '*.png') images, $(count_files "${DOTA_DIR}/val/annotation/hbb" '*.txt') hbb, $(count_files "${DOTA_DIR}/val/annotation/obb" '*.txt') obb"
}

# ---------------------------------------------------------------------------
# Kenyan Wildlife Aerial Survey  ->  raw/kenyan-wildlife-aerial-survey/coco/
# ---------------------------------------------------------------------------
#
# Kaggle's public dataset endpoint serves this archive without an API token: the
# request 302s to a signed Google Cloud Storage URL.  No credentials, no CLI.
#
# The archive is ~4 GB and holds three annotation variants side by side at its
# root -- coco/, retinanet-keras/ and yolov11/.  Only coco/ is read by the
# adapter, so only that subtree is extracted; pass KENYA_FULL=1 to unpack all
# three.

KENYA_DIR="${RAW_ROOT}/kenyan-wildlife-aerial-survey"
KENYA_SLUG="davidrpugh/kenyan-wildlife-aerial-survey"
KENYA_URL="https://www.kaggle.com/api/v1/datasets/download/${KENYA_SLUG}"

download_kenyan() {
  if [[ -f "${KENYA_DIR}/coco/train/_annotations.coco.json" ]]; then
    log "Kenyan Wildlife — already present, skipping"
    return 0
  fi

  log "Kenyan Wildlife Aerial Survey"
  need_cmd curl  "Install curl via your package manager."
  need_cmd unzip "Install unzip via your package manager."

  local zip="${STAGE}/kenyan-wildlife-aerial-survey.zip"

  # A previous run may have left a truncated or HTML-error file behind, so the
  # archive is validated rather than trusted on the strength of its name alone.
  if [[ -s "$zip" ]] && unzip -tqq "$zip" >/dev/null 2>&1; then
    info "archive already downloaded and valid, skipping fetch"
  else
    info "downloading (~3.9 GB, no Kaggle account required)"
    fetch_resumable "$zip" "$KENYA_URL"
    unzip -tqq "$zip" >/dev/null 2>&1 || die \
      "downloaded file is not a valid ZIP -- delete ${zip} and retry. If it keeps failing, Kaggle may have changed this endpoint; download by hand from https://www.kaggle.com/datasets/${KENYA_SLUG} and unzip into ${KENYA_DIR}/"
  fi

  info "extracting"
  mkdir -p "$KENYA_DIR"
  if [[ -n "${KENYA_FULL:-}" ]]; then
    unzip -q -o "$zip" -d "$KENYA_DIR"
  else
    unzip -q -o "$zip" 'coco/*' -d "$KENYA_DIR" || {
      warn "coco/ pattern matched nothing — extracting the full archive instead"
      unzip -q -o "$zip" -d "$KENYA_DIR"
    }
  fi

  local split
  for split in train valid test; do
    info "coco/${split}: $(count_files "${KENYA_DIR}/coco/${split}" '*.jpg') images"
  done
}

# ---------------------------------------------------------------------------
# Malaria (BBBC041v1)  ->  raw/malaria/{images,training.json,test.json}
# ---------------------------------------------------------------------------

MALARIA_DIR="${RAW_ROOT}/malaria"
MALARIA_URL="https://data.broadinstitute.org/bbbc/BBBC041/malaria.zip"

download_malaria() {
  if [[ -f "${MALARIA_DIR}/training.json" && -f "${MALARIA_DIR}/test.json" ]]; then
    log "Malaria — already present, skipping"
    return 0
  fi

  log "Malaria infected blood smears (BBBC041v1)"
  need_cmd curl  "Install curl via your package manager."
  need_cmd unzip "Install unzip via your package manager."

  local zip="${STAGE}/malaria.zip"
  if [[ -s "$zip" ]] && unzip -tqq "$zip" >/dev/null 2>&1; then
    info "archive already downloaded and valid, skipping fetch"
  else
    info "downloading malaria.zip (~2.3 GB)"
    info "if this stalls at 0 B/s, see the BBBC041 notes in README.md"
    fetch_resumable "$zip" "$MALARIA_URL"
    unzip -tqq "$zip" >/dev/null 2>&1 || die \
      "downloaded file is not a valid ZIP -- delete ${zip} and retry, or fetch malaria.zip by hand from https://bbbc.broadinstitute.org/BBBC041 and place it there"
  fi

  info "extracting"
  rm -rf "${STAGE}/malaria_x"
  mkdir -p "${STAGE}/malaria_x" "$MALARIA_DIR"
  unzip -q -o "$zip" -d "${STAGE}/malaria_x"

  # The archive nests its contents under a top-level folder; locate the JSONs
  # and the image directory rather than assuming a fixed depth.
  local f
  for f in training.json test.json; do
    local found
    found="$(find "${STAGE}/malaria_x" -type f -name "$f" | head -1)"
    [[ -n "$found" ]] || die "could not find $f inside malaria.zip"
    mv -f "$found" "${MALARIA_DIR}/${f}"
  done
  harvest "${STAGE}/malaria_x" '*.png' "${MALARIA_DIR}/images"
  harvest "${STAGE}/malaria_x" '*.jpg' "${MALARIA_DIR}/images"

  info "images: $(count_files "${MALARIA_DIR}/images" '*') files"
}

# ---------------------------------------------------------------------------
# Birds (Tree Swallows Flock)  ->  raw/birds/
# ---------------------------------------------------------------------------
#
# Nothing to download: the two source photographs, the point annotations, and
# the pre-computed pseudo-bbox cache are all part of this repository.  The
# photographs are lossless deflate-compressed TIFFs, pixel-identical to the
# originals -- see datasets.md.  This step only confirms they are intact.

BIRDS_DIR="${RAW_ROOT}/birds"

# SHA-256 of the repository copies.  A mismatch usually means an incomplete
# clone or a hand-placed uncompressed original, which decodes to the same pixels
# but hashes differently.
birds_expected_sha() {
  case "$1" in
    _DSC5214.tif) echo "5a6a681e536ee222e05118c614ae4b77c8400b6d70f111adb17e632a9bc1e580" ;;
    _DSC5295.tif) echo "b14198ed66acae0883186119c6d4b088155f5ba9f7ace813e148eaa9922cd4d9" ;;
  esac
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

download_birds() {
  log "Birds (Tree Swallows Flock) — included in this repository"

  local name file actual missing=0
  for name in _DSC5214.tif _DSC5295.tif; do
    file="${BIRDS_DIR}/${name}"
    if [[ ! -f "$file" ]]; then
      warn "missing ${file}"
      missing=1
      continue
    fi
    actual="$(sha256_of "$file")"
    if [[ -z "$actual" ]]; then
      warn "no sha256 tool available — skipping checksum for ${name}"
    elif [[ "$actual" == "$(birds_expected_sha "$name")" ]]; then
      info "checksum ok: ${name}"
    else
      warn "${name} differs from the repository copy (fine if it is an uncompressed original)"
    fi
  done

  if [[ "$missing" -eq 1 ]]; then
    info "The Birds photographs are tracked in this repository at ${HERE}/raw/birds/"
    info "Restore them with:  git checkout -- counting_datasets/raw/birds"
    info "or copy _DSC5214.tif and _DSC5295.tif into ${BIRDS_DIR}/ by hand."
    return 1
  fi
}

# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

check_layout() {
  log "Layout check"
  local ok=1
  _need_path() {
    if [[ -e "$1" ]]; then
      printf '    \033[32m[ok]\033[0m   %s\n' "${1#"$RAW_ROOT"/}"
    else
      printf '    \033[31m[MISS]\033[0m %s\n' "${1#"$RAW_ROOT"/}"
      ok=0
    fi
  }

  _need_path "${DOTA_DIR}/train/images"
  _need_path "${DOTA_DIR}/train/annotation/hbb"
  _need_path "${DOTA_DIR}/train/annotation/obb"
  _need_path "${DOTA_DIR}/val/images"
  _need_path "${DOTA_DIR}/val/annotation/hbb"
  _need_path "${DOTA_DIR}/val/annotation/obb"
  _need_path "${KENYA_DIR}/coco/train/_annotations.coco.json"
  _need_path "${KENYA_DIR}/coco/valid/_annotations.coco.json"
  _need_path "${KENYA_DIR}/coco/test/_annotations.coco.json"
  _need_path "${MALARIA_DIR}/images"
  _need_path "${MALARIA_DIR}/training.json"
  _need_path "${MALARIA_DIR}/test.json"
  _need_path "${BIRDS_DIR}/_DSC5214.tif"
  _need_path "${BIRDS_DIR}/_DSC5295.tif"
  _need_path "${BIRDS_DIR}/pseudo_bboxes_cache.json"
  _need_path "${BIRDS_DIR}/tiles/tiles_DSC5214/labels"
  _need_path "${BIRDS_DIR}/tiles/tiles_DSC5295/labels"

  echo
  if [[ "$ok" -eq 1 ]]; then
    info "All expected paths present.  Next:  python build_index.py"
  else
    warn "Some paths are missing — build_index.py will silently produce empty or"
    warn "partial counts for the affected datasets rather than failing loudly."
  fi
  return 0
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

# The Birds photographs, point annotations, and pseudo-bbox cache live in this
# repository under raw/birds/.  When COUNTING_RAW_ROOT redirects the raw tree
# somewhere else, copy them across so the dataset is not silently left
# incomplete at the new location.
seed_shipped_birds() {
  local shipped="${HERE}/raw/birds"
  [[ "$RAW_ROOT" != "${HERE}/raw" ]] || return 0
  [[ -d "${shipped}/tiles" ]] || return 0

  local target="${RAW_ROOT}/birds"
  [[ -f "${target}/pseudo_bboxes_cache.json" && -d "${target}/tiles" \
     && -f "${target}/_DSC5214.tif" && -f "${target}/_DSC5295.tif" ]] && return 0

  info "copying the Birds dataset into ${target}"
  mkdir -p "$target"
  cp -Rn "${shipped}/tiles" "${target}/" 2>/dev/null || true
  cp -n  "${shipped}/pseudo_bboxes_cache.json" "${target}/" 2>/dev/null || true
  cp -n  "${shipped}"/_DSC52*.tif "${target}/" 2>/dev/null || true
}

main() {
  mkdir -p "$RAW_ROOT" "$STAGE"
  seed_shipped_birds

  local targets=("$@")
  if [[ ${#targets[@]} -eq 0 ]]; then
    targets=(dota kenyan malaria birds)
  fi

  if [[ "${targets[0]}" == "--check" ]]; then
    check_layout
    return 0
  fi

  local failed=()
  local t
  for t in "${targets[@]}"; do
    case "$t" in
      dota)    download_dota    || failed+=("dota")    ;;
      kenyan)  download_kenyan  || failed+=("kenyan")  ;;
      malaria) download_malaria || failed+=("malaria") ;;
      birds)   download_birds   || failed+=("birds")   ;;
      *)       die "unknown dataset '$t' (expected: dota kenyan malaria birds)" ;;
    esac
  done

  check_layout

  if [[ ${#failed[@]} -gt 0 ]]; then
    echo
    warn "needs manual action: ${failed[*]}  (see the notes above)"
  fi

  echo
  info "Staging files kept in ${STAGE}"
  info "Delete that directory once the layout check passes to reclaim disk space."
}

main "$@"
