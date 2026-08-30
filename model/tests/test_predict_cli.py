from argparse import Namespace
import json
from pathlib import Path

from PIL import Image
import pytest

from model.scripts.predict import discover, run


class MockEngine:
    loads = 0
    def __init__(self, **kwargs):
        MockEngine.loads += 1
    def predict_images(self, images):
        return [image.getpixel((0, 0))[0] / 255 for image in images]
    def metadata(self):
        return {"mock": True}


def arguments(input_dir: Path, output: Path) -> Namespace:
    return Namespace(input_dir=input_dir, output=output, checkpoint=None, config=Path("unused"), device="cpu", batch_size=2, verbose=False)


def test_discovery_order_and_output_contract(tmp_path) -> None:
    (tmp_path / "nested").mkdir()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(tmp_path / "nested" / "B.PNG")
    Image.new("L", (4, 4), 0).save(tmp_path / "a.jpg")
    (tmp_path / "ignore.txt").write_text("x")
    assert [path.relative_to(tmp_path).as_posix() for path in discover(tmp_path)] == ["a.jpg", "nested/B.PNG"]
    output = tmp_path / "predictions.json"; MockEngine.loads = 0
    summary = run(arguments(tmp_path, output), MockEngine)
    payload = json.loads(output.read_text())
    assert MockEngine.loads == 1 and summary["readable"] == 2
    assert list(payload[0]) == ["image_path", "pred"]
    assert [row["image_path"] for row in payload] == ["a.jpg", "nested/B.PNG"]


def test_corrupt_image_writes_separate_errors(tmp_path) -> None:
    Image.new("RGBA", (4, 4), (128, 0, 0, 255)).save(tmp_path / "good.webp")
    (tmp_path / "bad.JPG").write_bytes(b"not an image")
    output = tmp_path / "results.json"
    summary = run(arguments(tmp_path, output), MockEngine)
    assert summary["readable"] == 1 and summary["errors"] == 1
    errors = json.loads((tmp_path / "results.errors.json").read_text())
    assert errors[0]["image_path"] == "bad.JPG"
    assert set(errors[0]) == {"image_path", "reason"}


def test_exif_orientation_is_applied_and_repeat_is_identical(tmp_path) -> None:
    image = Image.new("RGB", (6, 4), (64, 0, 0))
    exif = Image.Exif()
    exif[274] = 6
    image.save(tmp_path / "rotated.JPEG", exif=exif)
    output = tmp_path / "predictions.json"

    first = run(arguments(tmp_path, output), MockEngine)
    first_payload = output.read_text(encoding="utf-8")
    second = run(arguments(tmp_path, output), MockEngine)

    assert first["readable"] == second["readable"] == 1
    assert output.read_text(encoding="utf-8") == first_payload


def test_output_overwrite_removes_stale_error_file(tmp_path) -> None:
    Image.new("RGB", (4, 4), (10, 0, 0)).save(tmp_path / "good.png")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"broken")
    output = tmp_path / "predictions.json"
    run(arguments(tmp_path, output), MockEngine)
    assert (tmp_path / "predictions.errors.json").exists()

    bad.unlink()
    Image.new("RGB", (4, 4), (20, 0, 0)).save(tmp_path / "second.PNG")
    run(arguments(tmp_path, output), MockEngine)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert [row["image_path"] for row in payload] == ["good.png", "second.PNG"]
    assert not (tmp_path / "predictions.errors.json").exists()


def test_empty_and_unsupported_only_directories_are_fatal(tmp_path) -> None:
    output = tmp_path / "predictions.json"
    with pytest.raises(ValueError, match="no supported images"):
        run(arguments(tmp_path, output), MockEngine)
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    with pytest.raises(ValueError, match="no supported images"):
        run(arguments(tmp_path, output), MockEngine)
