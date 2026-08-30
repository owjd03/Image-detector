from io import BytesIO

import numpy as np
from PIL import Image

from model.src.transforms import GRADED_CONDITIONS, HELD_OUT_CONDITIONS, apply_condition, seed_for_source


def fixture_image() -> Image.Image:
    values = np.arange(32 * 24 * 3, dtype=np.uint8).reshape(24, 32, 3)
    return Image.fromarray(values)


def test_every_condition_preserves_dimensions_and_input() -> None:
    source = fixture_image()
    before = source.tobytes()
    for condition_id in [*GRADED_CONDITIONS, *HELD_OUT_CONDITIONS]:
        result = apply_condition(source, condition_id, seed=42)
        assert result.size == source.size
        assert result.mode == "RGB"
    assert source.tobytes() == before


def test_noise_is_reproducible_and_source_separated() -> None:
    source = fixture_image()
    first_seed = seed_for_source(42, "one")
    assert apply_condition(source, "noise_s0.05", seed=first_seed).tobytes() == apply_condition(source, "noise_s0.05", seed=first_seed).tobytes()
    assert apply_condition(source, "noise_s0.05", seed=first_seed).tobytes() != apply_condition(source, "noise_s0.05", seed=seed_for_source(42, "two")).tobytes()
    assert 0 <= first_seed <= (1 << 63) - 1


def test_jpeg_is_real_round_trip() -> None:
    source = fixture_image()
    transformed = apply_condition(source, "jpeg_q30", seed=0)
    assert transformed.tobytes() != source.tobytes()
    buffer = BytesIO()
    transformed.save(buffer, format="JPEG")
    assert buffer.getvalue().startswith(b"\xff\xd8")


def test_composed_condition_uses_documented_order() -> None:
    source = fixture_image()
    composed = apply_condition(source, "heldout_jpeg_q50_resize_0.5", seed=0)
    reverse = apply_condition(apply_condition(source, "resize_0.5", seed=0), "jpeg_q50", seed=0)
    assert composed.tobytes() != reverse.tobytes()
