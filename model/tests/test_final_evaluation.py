import json
from pathlib import Path

import pytest
import torch

import model.scripts.evaluate as evaluate
from model.src.heads import build_head


def test_final_scope_isolation() -> None:
    rows = [
        {"dataset": "sid", "role": "calibration", "condition": "clean"},
        {"dataset": "sid", "role": "calibration", "condition": "heldout_jpeg_q20"},
        {"dataset": "sid", "role": "internal_final_evaluation", "condition": "clean"},
        {"dataset": "cifake", "role": "cross_dataset_eval", "condition": "clean"},
        {"dataset": "sid", "role": "exploratory_tampered", "condition": "clean"},
    ]
    assert evaluate.select_scope("heldout", rows) == [1]
    assert evaluate.select_scope("test", rows) == [2]
    assert evaluate.select_scope("cifake", rows) == [3]
    assert evaluate.select_scope("tampered", rows) == [4]
    with pytest.raises(ValueError):
        evaluate.select_scope("validation", rows)


def frozen_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    training = tmp_path / "model" / "outputs" / "training" / "selected" / "seed_42"
    evaluation = tmp_path / "model" / "outputs" / "evaluation"
    final = tmp_path / "model" / "outputs" / "final"
    training.mkdir(parents=True)
    evaluation.mkdir(parents=True)
    torch.save(build_head("linear").state_dict(), training / "checkpoint.pt")
    (training / "metadata.json").write_text(json.dumps({"architecture": "linear"}), encoding="utf-8")
    proposal = {
        "experiment": "selected", "seed": 42,
        "checkpoint": "model/outputs/training/selected/seed_42/checkpoint.pt",
        "temperature": 1.25, "threshold": 0.6,
    }
    (evaluation / "proposal.json").write_text(json.dumps(proposal), encoding="utf-8")
    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate, "EVALUATION", evaluation)
    monkeypatch.setattr(evaluate, "FINAL", final)
    return evaluation, final


def test_freeze_packages_verified_runtime_bundle(tmp_path, monkeypatch) -> None:
    _, final = frozen_fixture(tmp_path, monkeypatch)
    frozen = evaluate.freeze_proposal()
    assert frozen["checkpoint"] == "model/outputs/final/checkpoint.pt"
    assert evaluate.file_hash(final / "checkpoint.pt") == frozen["checkpoint_sha256"]
    assert json.loads((final / "metadata.json").read_text())["architecture"] == "linear"
    assert evaluate.freeze_proposal() == frozen


def test_freeze_rejects_conflicting_existing_configuration(tmp_path, monkeypatch) -> None:
    evaluation, _ = frozen_fixture(tmp_path, monkeypatch)
    evaluate.freeze_proposal()
    proposal_path = evaluation / "proposal.json"
    proposal = json.loads(proposal_path.read_text())
    proposal["threshold"] = 0.7
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
    with pytest.raises(ValueError, match="different final model"):
        evaluate.freeze_proposal()


def test_load_model_rejects_corrupted_final_checkpoint(tmp_path, monkeypatch) -> None:
    _, final = frozen_fixture(tmp_path, monkeypatch)
    frozen = evaluate.freeze_proposal()
    (final / "checkpoint.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash changed"):
        evaluate.load_model(frozen, "cpu")
