"""Shared frozen inference engine used by CLI and later API stages."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Sequence

from PIL import Image
import torch

from model.src.clip_model import image_embeddings, load_frozen_clip
from model.src.config import DEFAULT_CONFIG_PATH, REPOSITORY_ROOT, load_config
from model.src.data import canonical_rgb
from model.src.heads import build_head


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class InferenceEngine:
    """Load CLIP and the frozen head once, then predict image batches."""

    def __init__(self, *, checkpoint: Path | None = None, config_path: Path = DEFAULT_CONFIG_PATH, device: str = "auto") -> None:
        config = load_config(config_path)
        final_dir = config.paths.outputs_dir / "final"
        frozen_path = final_dir / "frozen_model.json"
        if not frozen_path.is_file():
            raise ValueError(f"Frozen model configuration does not exist: {frozen_path}")
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        self.checkpoint = checkpoint.resolve() if checkpoint else (final_dir / "checkpoint.pt").resolve()
        if not self.checkpoint.is_file():
            raise ValueError(f"Classifier checkpoint does not exist: {self.checkpoint}")
        if checkpoint is None and (REPOSITORY_ROOT / frozen["checkpoint"]).resolve() != self.checkpoint:
            raise ValueError("Frozen configuration does not reference the final checkpoint")
        if checkpoint is None and file_sha256(self.checkpoint) != frozen["checkpoint_sha256"]:
            raise ValueError("Frozen classifier checkpoint hash does not match")
        metadata_path = self.checkpoint.with_name("metadata.json")
        if not metadata_path.is_file():
            raise ValueError(f"Classifier metadata does not exist: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.head = build_head(metadata["architecture"])
        self.head.load_state_dict(torch.load(self.checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.head.to(self.device).eval()
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.clip, self.processor = load_frozen_clip(config.model.model_id, config.model.revision, config.paths.hf_cache_dir, offline=True, dtype=dtype)
        self.clip.to(self.device).eval()
        self.dtype = dtype
        self.temperature = float(frozen["temperature"])
        self.threshold = float(frozen["threshold"])
        self.model_id = config.model.model_id
        self.revision = config.model.revision
        self.checkpoint_sha256 = file_sha256(self.checkpoint)

    def predict_images(self, images: Sequence[Image.Image]) -> list[float]:
        if not images:
            return []
        prepared = [canonical_rgb(image) for image in images]
        pixels = self.processor(images=prepared, return_tensors="pt")["pixel_values"].to(device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            embeddings = image_embeddings(self.clip, pixels)
            logits = self.head(embeddings.float())
            probabilities = torch.sigmoid(logits / self.temperature)
        return [float(value) for value in probabilities.cpu()]

    def metadata(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "revision": self.revision, "checkpoint_sha256": self.checkpoint_sha256, "temperature": self.temperature, "threshold": self.threshold, "device": self.device}
