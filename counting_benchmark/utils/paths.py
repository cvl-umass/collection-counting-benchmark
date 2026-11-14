"""Path resolution for image files referenced by the counting dataset index."""

from __future__ import annotations

from pathlib import Path


def resolve_image_path(p: str, raw_root: Path) -> Path:
    """Resolve an image path string from the SQLite index into an on-disk Path.

    Supported cases:
      1. Absolute paths are returned as-is.
      2. Paths starting with ``raw/...`` are interpreted relative to
         ``raw_root.parent`` (adapters often stored ``raw/<dataset>/...``).
      3. Everything else is interpreted relative to the current directory.
    """
    pp = Path(p)
    if pp.is_absolute():
        return pp

    s = str(pp).replace("\\", "/")
    if s.startswith("raw/"):
        return raw_root.parent / pp

    return Path.cwd() / pp
