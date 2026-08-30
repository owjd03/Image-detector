"""Small classifier heads trained on frozen CLIP embeddings."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import torch
from torch import nn


class LinearHead(nn.Module):
    def __init__(self, embedding_dim: int = 768) -> None:
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings).squeeze(-1)


class MLPHead(nn.Module):
    def __init__(self, embedding_dim: int = 768, dropout: float = 0.3) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings).squeeze(-1)


def build_head(architecture: str, embedding_dim: int = 768) -> nn.Module:
    if architecture == "linear":
        return LinearHead(embedding_dim)
    if architecture == "mlp":
        return MLPHead(embedding_dim)
    raise ValueError(f"Unknown head architecture: {architecture}")


def grouped_indices(source_ids: Sequence[str], conditions: Sequence[str]) -> list[list[int]]:
    """Group views by source, placing each clean view first."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, source_id in enumerate(source_ids):
        groups[source_id].append(index)
    result = []
    for source_id in sorted(groups):
        indices = sorted(groups[source_id], key=lambda index: (conditions[index] != "clean", conditions[index]))
        if not indices or conditions[indices[0]] != "clean":
            raise ValueError(f"Source has no clean view: {source_id}")
        result.append(indices)
    return result


def consistency_loss(logits: torch.Tensor, group_ids: Sequence[int], clean_mask: Sequence[bool]) -> torch.Tensor:
    """Mean squared logit difference from each group's clean view."""
    penalties = []
    by_group: dict[int, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        by_group[int(group_id)].append(index)
    for indices in by_group.values():
        clean = [index for index in indices if clean_mask[index]]
        if len(clean) != 1:
            raise ValueError("Each group must contain exactly one clean view")
        transformed = [index for index in indices if index != clean[0]]
        if transformed:
            penalties.append(((logits[transformed] - logits[clean[0]]) ** 2).mean())
    return torch.stack(penalties).mean() if penalties else logits.sum() * 0.0


def better_checkpoint(loss: float, worst_balanced_accuracy: float, best_loss: float, best_worst: float, tolerance: float = 1e-4) -> bool:
    if loss < best_loss - tolerance:
        return True
    return abs(loss - best_loss) <= tolerance and worst_balanced_accuracy > best_worst
