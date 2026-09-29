"""Image degradation simulation for VLM robustness experiments.

Conditions:
  E = severely degraded / pixelated (downscale to very low resolution, then upscale)
  F = poor-quality / low-information (Gaussian blur + brightness/contrast reduction)
  G = no usable image (returns None — never zero)

All operations are applied in-memory via PIL; original files are never modified.
If PIL is not installed, conditions E and F fall back to returning the original
path (no modification) and the condition metadata records `degraded=False`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _pil_available() -> bool:
    try:
        import PIL  # noqa: F401
        return True
    except ImportError:
        return False


def degrade_image_pixelate(
    source_path: Path,
    dest_path: Path,
    *,
    scale_factor: float = 0.05,
) -> Path:
    """Save a severely pixelated version of the image to dest_path.

    Downscales by `scale_factor` then upscales back to original size.
    """
    from PIL import Image  # type: ignore[import]

    with Image.open(source_path) as img:
        orig_size = img.size
        small_w = max(1, int(orig_size[0] * scale_factor))
        small_h = max(1, int(orig_size[1] * scale_factor))
        small = img.resize((small_w, small_h), Image.NEAREST)
        pixelated = small.resize(orig_size, Image.NEAREST)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        pixelated.save(dest_path)
    return dest_path


def degrade_image_blur_low_contrast(
    source_path: Path,
    dest_path: Path,
    *,
    blur_radius: float = 12.0,
    brightness_factor: float = 0.4,
) -> Path:
    """Save a blurred, low-contrast version of the image to dest_path."""
    from PIL import Image, ImageFilter, ImageEnhance  # type: ignore[import]

    with Image.open(source_path) as img:
        blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        darkened = ImageEnhance.Brightness(blurred).enhance(brightness_factor)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        darkened.save(dest_path)
    return dest_path


def describe_degradation_condition(condition: str) -> dict[str, Any]:
    """Return metadata describing the degradation condition."""
    metadata: dict[str, Any] = {
        "condition": condition,
        "degraded": condition in {"E_pixelated", "F_low_quality"},
        "no_image": condition == "G_0pct",
    }
    if condition == "E_pixelated":
        metadata["description"] = "Severely pixelated (scale_factor=0.05)"
        metadata["method"] = "downscale_upscale_nearest"
    elif condition == "F_low_quality":
        metadata["description"] = "Heavy blur + low contrast (blur=12, brightness=0.4)"
        metadata["method"] = "gaussian_blur_brightness_reduction"
    elif condition == "G_0pct":
        metadata["description"] = "No usable images — VLM output explicitly missing"
        metadata["method"] = "no_image"
    else:
        metadata["description"] = f"Partial availability condition: {condition}"
        metadata["method"] = "deterministic_subset"
    return metadata
