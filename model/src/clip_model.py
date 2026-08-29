"""Shared, revision-pinned frozen CLIP vision feature extractor."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from transformers import AutoImageProcessor, CLIPVisionModelWithProjection


PLACEHOLDER_REVISION = "TO_BE_PINNED_IN_STAGE_02"


def validate_pinned_revision(revision: str) -> str:
    """Require a full immutable Hugging Face commit hash."""

    normalized = revision.strip().lower()
    if len(normalized) != 40 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(
            "Model revision must be a 40-character immutable commit hash; "
            "run model.scripts.download_model without --verify-only first"
        )
    return normalized


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """Scale each embedding vector to Euclidean (L2) length one."""

    return functional.normalize(embeddings.float(), p=2, dim=-1)


def load_frozen_clip(
    model_id: str,
    revision: str,
    cache_dir: Path,
    *,
    offline: bool = False,
    dtype: torch.dtype | None = None,
) -> tuple[CLIPVisionModelWithProjection, Any]:
    """Load only CLIP's vision tower and processor, with all weights frozen."""

    pinned_revision = validate_pinned_revision(revision)
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    common = {
        "revision": pinned_revision,
        "cache_dir": str(cache_dir),
        "local_files_only": offline,
    }
    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False, **common)
    model = CLIPVisionModelWithProjection.from_pretrained(
        model_id,
        torch_dtype=dtype,
        **common,
    )
    model.eval()
    model.requires_grad_(False)
    return model, processor


def image_embeddings(
    model: CLIPVisionModelWithProjection,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    """Return normalized 768-value projected image embeddings."""

    return normalize_embeddings(model(pixel_values=pixel_values).image_embeds)
