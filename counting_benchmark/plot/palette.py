"""Shared palettes and plot styling for every figure."""

from __future__ import annotations

import colorsys

import matplotlib

matplotlib.use("Agg")
# Print typography.
matplotlib.rcParams.update(
    {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}
)
import pandas as pd  # noqa: E402

# --- shared ink / surface ----------------------------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
TIERS = ["full", "partial", "sparse", "rare"]

# --- tiers -------------------------------------------------------------------
TIER_COLOR = {
    "full": "#86b6ef",
    "partial": "#3987e5",
    "sparse": "#1c5cab",
    "rare": "#0d366b",
}

# =============================================================================
# MODEL palette
# =============================================================================
# Model -> family.  Four families, fixed order
FAMILY = {
    "frcnn_resnet50": "detector",
    "frcnn_mobilenet": "detector",
    "grounding_dino": "vlm",
    "owlv2": "vlm",
    "florence2": "vlm",
    "aya-vision-8b": "open_mllm",
    "internvl3-5-8b-hf": "open_mllm",
    "qwen3-vl-8b-instruct": "open_mllm",
    "molmo-7b-d-0924": "open_mllm",
    "molmo2-8b": "open_mllm",
    "gemini-3.1-pro-preview": "prop_mllm",
    "gpt-5.2": "prop_mllm",
    "qwen3.7-plus": "prop_mllm",
}
FAM_ORDER = ["detector", "vlm", "prop_mllm", "open_mllm"]

FAM_HUE = {
    "detector": "#49006a",
    "vlm": "#f768a1",
    "open_mllm": "#ae017e",
    "prop_mllm": "#f9ada7",
}

FAM_COLOR = {**FAM_HUE, "mc_floor": MUTED}
FAM_BASE = dict(FAM_HUE)

FAM_LABEL = {
    "detector": "fine-tuned detector (FRCNN)",
    "vlm": "open-vocab detector, OVD (DINO/OWL/Florence)",
    "open_mllm": "open MLLM",
    "prop_mllm": "proprietary MLLM",
    "mc_floor": "no model helps (uniform Monte Carlo)",
}
# White text on the dark fills, ink on the light ones
FAM_TEXT = {
    "detector": "#ffffff",
    "vlm": INK,
    "open_mllm": "#ffffff",
    "prop_mllm": INK,
    "mc_floor": "#ffffff",
}
FAM_LEGEND = {
    "detector": "Fine-tuned det.",
    "vlm": "OVD",
    "open_mllm": "Open MLLM",
    "prop_mllm": "Prop. MLLM",
    "mc_floor": "No model",
}
ABBR = {
    "frcnn_resnet50": "R50",
    "frcnn_mobilenet": "MobN",
    "grounding_dino": "gDINO",
    "owlv2": "OWL",
    "florence2": "Flor",
    "aya-vision-8b": "Aya",
    "internvl3-5-8b-hf": "IntV",
    "qwen3-vl-8b-instruct": "QwV",
    "molmo-7b-d-0924": "Mlm",
    "molmo2-8b": "Mlm2",
    "gemini-3.1-pro-preview": "Gem",
    "gpt-5.2": "GPT",
    "qwen3.7-plus": "Qw+",
}
# Full model names for the legend (cells show the short ABBR; the legend maps each shade + abbrev to the real model).
NAME = {
    "frcnn_resnet50": "FRCNN-R50",
    "frcnn_mobilenet": "FRCNN-MobN",
    "grounding_dino": "GroundingDINO",
    "owlv2": "OWLv2",
    "florence2": "Florence-2",
    "aya-vision-8b": "Aya-Vision-8B",
    "internvl3-5-8b-hf": "InternVL3.5-8B",
    "qwen3-vl-8b-instruct": "Qwen3-VL-8B",
    "molmo-7b-d-0924": "Molmo-7B-D",
    "molmo2-8b": "Molmo2-8B",
    "gemini-3.1-pro-preview": "Gemini-3.1-Pro",
    "gpt-5.2": "GPT-5.2",
    "qwen3.7-plus": "Qwen3.7-Plus",
}
MC_SENTINEL = "__mc__"  # stands in for the model column on the Monte-Carlo-floor rows

# --- per-model shading -------------------------------------------------------
# Each model gets its family's hue but a distinct lightness, so members of a family read as tints of one color.
FAM_RAMP = {
    "detector": (0.34, 0.0),
    "vlm": (0.44, 0.0),
    "open_mllm": (0.58, 0.0),
    "prop_mllm": (0.20, 0.0),
}
FAM_MEMBERS: dict[str, list[str]] = {}
for _m, _f in FAMILY.items():
    FAM_MEMBERS.setdefault(_f, []).append(_m)


def _mix(a, b, w):
    """Blend RGB ``a`` toward ``b`` by fraction ``w`` (0 -> a, 1 -> b)."""
    return tuple(a[i] * (1 - w) + b[i] * w for i in range(3))


def _rot_hue(rgb, turns: float):
    """Rotate hue by ``turns`` at fixed lightness/saturation.  Rotation is safe in HLS
    (it is *lightness ramps* there that go neon on saturated greens, so those stay in RGB
    below)."""
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb((h + turns) % 1.0, l, s)


def _shades(base_hex: str, n: int, light: float = 0.5, hue: float = 0.0):
    """``n`` shades of ``base_hex``, lightest first, spanning ``light`` of the way to white
    and ``hue`` turns of hue rotation (centered on the base, so the family still reads as its
    base hue).  Nothing is ever blended toward black -- that had washed the palette out -- so
    every member keeps the base's saturation."""
    base = matplotlib.colors.to_rgb(base_hex)
    if n == 1:
        return [base]
    out = []
    for i in range(n):
        t = i / (n - 1)  # 0 = lightest, 1 = vivid end
        rgb = _rot_hue(base, hue * (t - 0.5))
        out.append(_mix(rgb, (1.0, 1.0, 1.0), light * (1 - t)))
    return out


def _rel_lum(rgb) -> float:
    def lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ink_on(rgb) -> str:
    """Black or white text -- whichever gives more WCAG contrast on ``rgb``."""
    L = _rel_lum(rgb)
    return "#ffffff" if (1.05 / (L + 0.05)) >= ((L + 0.05) / 0.05) else INK


MODEL_COLOR: dict[str, str] = {}  # model -> cell fill (family hue, own tint)
MODEL_TEXT: dict[str, str] = {}  # model -> readable text color on that fill
for _f, _members in FAM_MEMBERS.items():
    _light, _hue = FAM_RAMP[_f]
    for _model, _rgb in zip(
        _members, _shades(FAM_BASE[_f], len(_members), _light, _hue)
    ):
        MODEL_COLOR[_model] = matplotlib.colors.to_hex(_rgb)
        MODEL_TEXT[_model] = _ink_on(_rgb)
MODEL_COLOR[MC_SENTINEL] = "#dedcd5"  # neutral warm-gray floor
MODEL_TEXT[MC_SENTINEL] = INK
NAME[MC_SENTINEL] = "uniform Monte Carlo"

# =============================================================================
# ESTIMATOR palette
# =============================================================================
# config -> strategy (how the estimator uses the model's prediction g).
EST_STRATEGY = {
    "model_only_corrected": "trust",  # believe g, add the revealed labels
    "monte_carlo": "ignore",  # never read g (the model-blind floor)
    "difference": "correct",
    "ratio": "correct",
    "regression": "correct",
    "dis": "direct",
    "ais": "direct",  # sample proportional to g (frozen / retrained)
    "spotdis": "spot",
    "spotais": "spot",  # direct + emptiness filter
}
# Per-config fill.
EST_COLOR = {
    "model_only_corrected": "#9c9c9c",  # trust  -> gray
    "monte_carlo": "#70747a",  # ignore -> charcoal
    "difference": "#c7e9b4",  # correct-> pale green
    "ratio": "#7fcdbb",  # correct-> teal-green
    "regression": "#41b6c4",  # correct-> cyan
    "dis": "#1d91c0",  # direct -> blue (the default choice)
    "spotdis": "#225ea8",  # spot   -> deeper blue
    "ais": "#253494",  # direct -> indigo (retrained)
    "spotais": "#081d58",  # spot   -> navy
}
EST_TEXT = {k: _ink_on(matplotlib.colors.to_rgb(v)) for k, v in EST_COLOR.items()}
# Ramp order = stack order, so adjacent segments are always one lightness step apart.

EST_ORDER = [
    "spotais",
    "ais",
    "spotdis",
    "dis",
    "regression",
    "ratio",
    "difference",
    "monte_carlo",
    "model_only_corrected",
]

EST_LABEL = {
    "model_only_corrected": "model-only",
    "monte_carlo": "Monte Carlo",
    "difference": "difference",
    "ratio": "ratio",
    "regression": "regression",
    "dis": "DIS",
    "ais": "AIS",
    "spotdis": "spotDIS",
    "spotais": "spotAIS",
}


# --- shared axis styling -----------------------------------------------------
def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def class_meta(df: pd.DataFrame) -> pd.DataFrame:
    """One row per class, ordered full -> rare then by emptiness."""
    meta = df[["class_key", "tier", "frac_empty"]].drop_duplicates()
    meta["rank"] = meta.tier.map(TIERS.index)
    # Display name, not the key: "dota/small_vehicle" -> "Small vehicle".  Raw snake_case
    # keys in an axis are the giveaway that a figure came straight off a script; the key
    # still lives in class_key for every join.
    meta["short"] = (
        meta.class_key.str.split("/")
        .str[1]
        .str.replace("_", " ", regex=False)
        .str.capitalize()
    )
    return meta.sort_values(["rank", "frac_empty", "class_key"], ignore_index=True)
