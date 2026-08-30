from PIL import Image
import pytest

from model.src.data import ImageSample, validate_sample_role
from model.src.manifests import evaluation_descriptors, manifest_row, training_descriptors, validate_manifest
from model.src.transforms import HELD_OUT_CONDITIONS


def sample(identifier: str, *, dataset: str = "sid", split: str = "train", role: str = "train", label: int = 0, color: str = "red") -> ImageSample:
    return ImageSample(identifier, dataset, "relative", split, label, label if label in {0, 1} else None, role, identifier, Image.new("RGB", (8, 8), color))


def test_forbidden_training_role_is_rejected() -> None:
    with pytest.raises(ValueError, match="Only official-train SID"):
        validate_sample_role(sample("x", dataset="cifake"))


def test_wildfake_training_is_rejected() -> None:
    with pytest.raises(ValueError, match="external_demo_only"):
        validate_sample_role(sample("x", dataset="wildfake"))


def test_cross_role_exact_duplicate_is_fatal() -> None:
    rows = [manifest_row(sample("train")), manifest_row(sample("eval", split="validation", role="calibration"))]
    with pytest.raises(ValueError, match="Exact cross-role"):
        validate_manifest(rows)


def test_same_role_cross_split_exact_duplicate_is_fatal() -> None:
    rows = [
        manifest_row(sample("train", dataset="cifake", split="train", role="cross_dataset_eval")),
        manifest_row(sample("test", dataset="cifake", split="test", role="cross_dataset_eval")),
    ]
    with pytest.raises(ValueError, match="Exact cross-role"):
        validate_manifest(rows)


def test_training_plan_has_three_views_and_no_heldout() -> None:
    rows = [manifest_row(sample(f"x-{index}", label=index % 2, color=("red" if index % 2 == 0 else "blue"))) for index in range(8)]
    plan = training_descriptors(rows, 42)
    assert len(plan) == 24
    assert all(item["condition_id"] not in HELD_OUT_CONDITIONS for item in plan)
    assert all(sum(candidate["source_id"] == item["source_id"] for candidate in plan) == 3 for item in plan)


def test_full_evaluation_profile_uses_all_available_records() -> None:
    rows = []
    for role in ("calibration", "internal_final_evaluation"):
        for label in (0, 1):
            rows.extend(manifest_row(sample(f"{role}-{label}-{index}", split="validation", role=role, label=label, color=("red" if label == 0 else "blue"))) for index in range(300))
    rows.extend(manifest_row(sample(f"tampered-{index}", split="test", role="exploratory_tampered", label=2, color="green")) for index in range(120))
    for label in (0, 1):
        rows.extend(manifest_row(sample(f"cifake-{label}-{index}", dataset="cifake", split="test", role="cross_dataset_eval", label=label, color=("red" if label == 0 else "blue"))) for index in range(550))
    plan = evaluation_descriptors(rows, 42)
    assert len(plan) == (1_200 * 23) + (120 * 23) + 1_100
    assert sum(item["role"] == "calibration" for item in plan) == 600 * 23
    assert sum(item["role"] == "internal_final_evaluation" for item in plan) == 600 * 23
    assert sum(item["dataset"] == "cifake" for item in plan) == 1100
    assert sum(item["role"] == "exploratory_tampered" for item in plan) == 120 * 23
