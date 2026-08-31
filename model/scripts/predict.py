"""Recursively predict supported images and write the graded JSON schema."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Callable

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.src.config import DEFAULT_CONFIG_PATH
from model.src.data import canonical_rgb
from model.src.inference import InferenceEngine


SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


def discover(input_dir: Path) -> list[Path]:
    return sorted((path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED), key=lambda path: path.relative_to(input_dir).as_posix().lower())


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verdict_line(image_path: str, probability: float, threshold: float) -> str:
    """Format one calibrated prediction for a human reading the terminal."""

    verdict = "AI-GENERATED" if probability >= threshold else "AUTHENTIC"
    return (
        f"- {image_path}: {verdict} | "
        f"AI probability {probability:.2%} | authentic probability {1.0 - probability:.2%}"
    )


def run(args: argparse.Namespace, engine_factory: Callable[..., InferenceEngine] = InferenceEngine) -> dict[str, object]:
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    paths = discover(input_dir)
    if not paths:
        raise ValueError("Input directory contains no supported images")
    engine = engine_factory(checkpoint=args.checkpoint, config_path=args.config, device=args.device)
    predictions, errors = [], []
    started = time.perf_counter()
    for start in range(0, len(paths), args.batch_size):
        readable_paths, images = [], []
        for path in paths[start:start + args.batch_size]:
            try:
                with Image.open(path) as decoded:
                    if getattr(decoded, "is_animated", False):
                        raise ValueError("animated images are unsupported")
                    images.append(canonical_rgb(decoded)); readable_paths.append(path)
            except (OSError, ValueError, UnidentifiedImageError) as error:
                relative = path.relative_to(input_dir).as_posix()
                errors.append({"image_path": relative, "reason": f"unreadable image: {type(error).__name__}"})
                print(f"warning: skipped {relative}: unreadable image", file=sys.stderr)
        if images:
            values = engine.predict_images(images)
            if len(values) != len(images):
                raise RuntimeError("Inference engine returned the wrong prediction count")
            for path, probability in zip(readable_paths, values):
                if not 0.0 <= probability <= 1.0:
                    raise RuntimeError("Inference engine returned a non-finite/out-of-range probability")
                predictions.append({"image_path": path.relative_to(input_dir).as_posix(), "pred": probability})
    if not predictions:
        raise ValueError("No readable supported images were found")
    predictions.sort(key=lambda row: row["image_path"].lower())
    atomic_json(args.output, predictions)
    error_path = args.output.with_name(args.output.stem + ".errors.json")
    if errors:
        atomic_json(error_path, errors)
    elif error_path.exists():
        error_path.unlink()
    return {"readable": len(predictions), "errors": len(errors), "elapsed_seconds": time.perf_counter() - started, "output": str(args.output), "engine": engine.metadata()}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input_dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--verbose", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        summary = run(args)
        print(json.dumps(summary, indent=2))
        predictions = json.loads(args.output.read_text(encoding="utf-8"))
        engine = summary.get("engine", {})
        threshold = engine.get("threshold") if isinstance(engine, dict) else None
        if not isinstance(threshold, (int, float)):
            raise RuntimeError("Inference engine metadata does not contain a numeric threshold")
        print("\nFinal verdicts:")
        for prediction in predictions:
            print(verdict_line(prediction["image_path"], prediction["pred"], float(threshold)))
        print("\nThese are model estimates, not proof of image provenance.")
        return 0
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
