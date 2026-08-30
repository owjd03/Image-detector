"""Render the bounded Stage 04 near-duplicate report for manual review."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

from PIL import Image, ImageDraw
import pyarrow.parquet as pq


def _load(locator: str) -> Image.Image:
    path_text, row_text = locator.rsplit("#row=", 1)
    row = pq.read_table(Path(path_text), columns=["image"]).slice(int(row_text), 1).to_pylist()[0]
    with Image.open(BytesIO(row["image"]["bytes"])) as decoded:
        return decoded.convert("RGB").copy()


def main() -> int:
    manifests = Path("model/outputs/manifests")
    report = json.loads((manifests / "leakage_report.json").read_text(encoding="utf-8"))
    rows = pq.read_table(manifests / "dataset_manifest.parquet", columns=["source_id", "native_locator"]).to_pylist()
    locators = {row["source_id"]: row["native_locator"] for row in rows}
    pairs = report.get("near_cross_role", [])
    width, image_height, label_height = 320, 220, 48
    canvas = Image.new("RGB", (width * 2, (image_height + label_height) * len(pairs)), "white")
    draw = ImageDraw.Draw(canvas)
    for pair_index, pair in enumerate(pairs):
        for column, key in enumerate(("left", "right")):
            identifier = pair[key]
            image = _load(locators[identifier])
            image.thumbnail((width, image_height), Image.Resampling.LANCZOS)
            x, y = column * width, pair_index * (image_height + label_height)
            canvas.paste(image, (x + (width - image.width) // 2, y))
            draw.text((x + 4, y + image_height + 2), f"d={pair['distance']} {identifier}", fill="black")
    output = Path("model/outputs/transform_grids/near_duplicate_review.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"saved {len(pairs)} near-duplicate pairs to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
