"""Shared image-sample adapters for SID Parquet and CIFAKE folders."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterator

from PIL import Image, ImageOps
import pyarrow.parquet as pq


@dataclass(frozen=True)
class ImageSample:
    source_id: str
    dataset: str
    native_locator: str
    official_split: str
    original_label: int
    binary_label: int | None
    role: str
    source_group_id: str
    image: Image.Image


def canonical_rgb(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation and return a detached RGB image."""

    return ImageOps.exif_transpose(image).convert("RGB").copy()


def iter_sid(root: Path) -> Iterator[ImageSample]:
    """Yield selected SID records without expanding them to loose files."""

    for role_dir in sorted(
        path for path in root.iterdir() if path.is_dir() and not path.name.startswith("_")
    ):
        for parquet_path in sorted(role_dir.glob("*.parquet")):
            relative = parquet_path.relative_to(root.parents[3]).as_posix()
            table = pq.read_table(parquet_path)
            for row_index, row in enumerate(table.to_pylist()):
                label = int(row["label"])
                binary = label if label in {0, 1} else None
                with Image.open(BytesIO(row["image"]["bytes"])) as decoded:
                    image = canonical_rgb(decoded)
                source_id = f"sid:{row['official_split']}:{row['img_id']}"
                yield ImageSample(
                    source_id=source_id,
                    dataset="sid",
                    native_locator=f"{relative}#row={row_index}",
                    official_split=str(row["official_split"]),
                    original_label=label,
                    binary_label=binary,
                    role=str(row["role"]),
                    source_group_id=source_id,
                    image=image,
                )


def iter_cifake(root: Path, *, include_train: bool = True) -> Iterator[ImageSample]:
    """Yield CIFAKE JPGs while preserving its supplied train/test split."""

    splits = ("train", "test") if include_train else ("test",)
    for split in splits:
        for class_name, label in (("REAL", 0), ("FAKE", 1)):
            directory = root / split / class_name
            for path in sorted(directory.glob("*"), key=lambda item: item.as_posix().lower()):
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                with Image.open(path) as decoded:
                    image = canonical_rgb(decoded)
                locator = path.relative_to(root.parents[2]).as_posix()
                source_id = f"cifake:{split}/{class_name}/{path.name}"
                yield ImageSample(
                    source_id=source_id,
                    dataset="cifake",
                    native_locator=locator,
                    official_split=split,
                    original_label=label,
                    binary_label=label,
                    role="cross_dataset_eval",
                    source_group_id=source_id,
                    image=image,
                )


def validate_sample_role(sample: ImageSample) -> None:
    if sample.dataset == "wildfake" and sample.role != "external_demo_only":
        raise ValueError("WildFake samples may only use role=external_demo_only")
    if sample.role == "train" and not (
        sample.dataset == "sid" and sample.official_split == "train"
    ):
        raise ValueError("Only official-train SID samples may enter training")
