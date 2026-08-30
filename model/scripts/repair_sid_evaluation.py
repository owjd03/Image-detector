"""Repair validation-role overlap caused by extending an earlier preflight."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from io import BytesIO
from pathlib import Path
import shutil
import sys

import pyarrow.parquet as pq
from PIL import Image

from model.scripts.download_datasets import _sid_stream, _write_shard
from model.src.config import DEFAULT_CONFIG_PATH, load_config


def _rows(directory: Path) -> list[dict]:
    return [row for path in sorted(directory.glob("*.parquet")) for row in pq.read_table(path).to_pylist()]


def _pixel_digest(row: dict) -> str:
    with Image.open(BytesIO(row["image"]["bytes"])) as decoded:
        image = decoded.convert("RGB")
    width, height = image.size
    digest = hashlib.sha256()
    digest.update(width.to_bytes(8, "big"))
    digest.update(height.to_bytes(8, "big"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--shuffle-buffer", type=int, default=1_000)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        root = config.paths.resources_dir / "datasets" / "sid_set" / "tier_a"
        calibration = _rows(root / "calibration")
        calibration_ids = {str(row["img_id"]) for row in calibration}
        calibration_pixels = {_pixel_digest(row) for row in calibration}
        current = _rows(root / "internal_final_evaluation")
        kept = [
            row for row in current
            if str(row["img_id"]) not in calibration_ids and _pixel_digest(row) not in calibration_pixels
        ]
        kept_ids = {str(row["img_id"]) for row in kept}
        kept_pixels = {_pixel_digest(row) for row in kept}
        counts = Counter(int(row["label"]) for row in kept)
        print(f"kept={len(kept)} removed_overlap={len(current) - len(kept)} counts={dict(counts)}", flush=True)
        for raw in _sid_stream(config, "validation", config.project.seed, args.shuffle_buffer):
            identifier, label = str(raw["img_id"]), int(raw["label"])
            if label not in {0, 1} or identifier in calibration_ids or identifier in kept_ids or counts[label] >= 1_000:
                continue
            candidate = {
                "img_id": identifier, "image": raw["image"], "mask": raw["mask"],
                "width": int(raw["width"]), "height": int(raw["height"]), "label": label,
                "official_split": "validation", "role": "internal_final_evaluation",
            }
            candidate_pixel = _pixel_digest(candidate)
            if candidate_pixel in calibration_pixels or candidate_pixel in kept_pixels:
                continue
            kept.append(candidate)
            kept_ids.add(identifier)
            kept_pixels.add(candidate_pixel)
            counts[label] += 1
            if counts[0] == 1_000 and counts[1] == 1_000:
                break
        if counts != Counter({0: 1_000, 1: 1_000}) or len(kept_ids) != 2_000:
            raise RuntimeError(f"Could not restore balanced evaluation quotas: {dict(counts)} unique={len(kept_ids)}")
        temporary = root / "_internal_final_evaluation_repaired"
        if temporary.exists():
            raise RuntimeError(f"Temporary repair directory already exists: {temporary}")
        for index in range(0, len(kept), 500):
            _write_shard(temporary, index // 500, kept[index:index + 500])
        written = _rows(temporary)
        if (
            len(written) != 2_000
            or {str(row["img_id"]) for row in written} & calibration_ids
            or {_pixel_digest(row) for row in written} & calibration_pixels
        ):
            raise RuntimeError("Replacement evaluation validation failed")
        original = root / "internal_final_evaluation"
        backup = root / f"_backup_internal_final_evaluation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        shutil.move(str(original), str(backup))
        shutil.move(str(temporary), str(original))
        print(f"repair complete rows=2000 backup={backup.name}")
        return 0
    except Exception as error:
        print(f"SID evaluation repair failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
