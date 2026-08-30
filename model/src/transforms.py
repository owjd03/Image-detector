"""Deterministic image-space robustness transformations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


TransformFunction = Callable[[Image.Image, int], Image.Image]


@dataclass(frozen=True)
class Condition:
    condition_id: str
    family: str
    parameters: dict[str, float | int | str]
    transform: TransformFunction
    held_out: bool = False


def seed_for_source(global_seed: int, source_id: str, view_index: int = 0) -> int:
    payload = f"{global_seed}\0{source_id}\0{view_index}".encode("utf-8")
    # Keep the deterministic value inside Parquet/PyTorch signed int64 range.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _jpeg(quality: int) -> TransformFunction:
    def apply(image: Image.Image, seed: int) -> Image.Image:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()
    return apply


def _webp(quality: int) -> TransformFunction:
    def apply(image: Image.Image, seed: int) -> Image.Image:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="WEBP", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()
    return apply


def _blur(sigma: float) -> TransformFunction:
    return lambda image, seed: image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=sigma))


def _resize(scale: float) -> TransformFunction:
    def apply(image: Image.Image, seed: int) -> Image.Image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        reduced = rgb.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BICUBIC)
        return reduced.resize((width, height), Image.Resampling.BICUBIC)
    return apply


def _noise(sigma: float) -> TransformFunction:
    def apply(image: Image.Image, seed: int) -> Image.Image:
        values = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        noise = np.random.default_rng(seed).normal(0.0, sigma, size=values.shape)
        result = np.clip(values + noise, 0.0, 1.0)
        return Image.fromarray(np.rint(result * 255.0).astype(np.uint8))
    return apply


def _enhance(kind: str, factor: float) -> TransformFunction:
    classes = {"brightness": ImageEnhance.Brightness, "contrast": ImageEnhance.Contrast, "saturation": ImageEnhance.Color}
    return lambda image, seed: classes[kind](image.convert("RGB")).enhance(factor)


def _center_crop(image: Image.Image, seed: int) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    crop_width, crop_height = max(1, round(width * 0.8)), max(1, round(height * 0.8))
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    return rgb.crop((left, top, left + crop_width, top + crop_height)).resize((width, height), Image.Resampling.BICUBIC)


def _compose(*steps: TransformFunction) -> TransformFunction:
    def apply(image: Image.Image, seed: int) -> Image.Image:
        result = image.convert("RGB").copy()
        for step in steps:
            result = step(result, seed)
        return result
    return apply


GRADED_CONDITIONS: dict[str, Condition] = {}
for quality in (90, 70, 50, 30):
    GRADED_CONDITIONS[f"jpeg_q{quality}"] = Condition(f"jpeg_q{quality}", "jpeg", {"quality": quality}, _jpeg(quality))
for sigma in (0.5, 1.0, 2.0):
    key = f"blur_s{sigma}"
    GRADED_CONDITIONS[key] = Condition(key, "blur", {"sigma": sigma}, _blur(sigma))
for scale in (0.5, 0.25):
    key = f"resize_{scale}"
    GRADED_CONDITIONS[key] = Condition(key, "resize", {"scale": scale}, _resize(scale))
for sigma in (0.02, 0.05, 0.10):
    key = f"noise_s{sigma:.2f}"
    GRADED_CONDITIONS[key] = Condition(key, "noise", {"sigma": sigma}, _noise(sigma))
for kind in ("brightness", "contrast", "saturation"):
    for factor in (0.8, 1.2):
        key = f"{kind}_x{factor}"
        GRADED_CONDITIONS[key] = Condition(key, kind, {"factor": factor}, _enhance(kind, factor))
GRADED_CONDITIONS["center_crop_0.8"] = Condition("center_crop_0.8", "crop", {"retain": 0.8}, _center_crop)

HELD_OUT_CONDITIONS = {
    "heldout_webp_q50": Condition("heldout_webp_q50", "webp", {"quality": 50}, _webp(50), True),
    "heldout_jpeg_q50_resize_0.5": Condition("heldout_jpeg_q50_resize_0.5", "composed", {"steps": "jpeg_q50,resize_0.5"}, _compose(_jpeg(50), _resize(0.5)), True),
    "heldout_jpeg_q20": Condition("heldout_jpeg_q20", "jpeg", {"quality": 20}, _jpeg(20), True),
}


def apply_condition(image: Image.Image, condition_id: str, *, seed: int) -> Image.Image:
    if condition_id == "clean":
        return image.convert("RGB").copy()
    condition = GRADED_CONDITIONS.get(condition_id) or HELD_OUT_CONDITIONS.get(condition_id)
    if condition is None:
        raise KeyError(f"Unknown condition ID: {condition_id}")
    return condition.transform(image.convert("RGB").copy(), seed)
