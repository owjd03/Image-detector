"""Create labelled clean/transformed grids from user-selected image files."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from model.src.transforms import GRADED_CONDITIONS, apply_condition, seed_for_source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()
    sample_root, output = Path(args.samples), Path(args.out)
    files = [path for path in sorted(sample_root.rglob("*")) if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}][:args.limit]
    if not files:
        parser.error("No supported sample images found")
    output.mkdir(parents=True, exist_ok=True)
    conditions = ["clean", *sorted(GRADED_CONDITIONS)]
    thumb, label_height, columns = 160, 24, 5
    for sample_path in files:
        with Image.open(sample_path) as decoded:
            original = ImageOps.exif_transpose(decoded).convert("RGB")
        rows = (len(conditions) + columns - 1) // columns
        canvas = Image.new("RGB", (columns * thumb, rows * (thumb + label_height)), "white")
        draw = ImageDraw.Draw(canvas)
        for index, condition_id in enumerate(conditions):
            transformed = apply_condition(original, condition_id, seed=seed_for_source(42, sample_path.as_posix(), index))
            transformed.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
            x, y = (index % columns) * thumb, (index // columns) * (thumb + label_height)
            canvas.paste(transformed, (x + (thumb - transformed.width) // 2, y))
            draw.text((x + 3, y + thumb + 3), condition_id, fill="black")
        canvas.save(output / f"{sample_path.parent.name}_{sample_path.stem}_transform_grid.png")
    print(f"saved {len(files)} transform grids to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
