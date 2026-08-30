"""Rank Stage 7 failures and generate Stage 8 metadata/contact sheets."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import pyarrow.parquet as pq

from model.scripts.extract_embeddings import _load_transformed


ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "model" / "outputs" / "evaluation" / "final" / "test_predictions.csv"
OUTPUT = ROOT / "model" / "outputs" / "error_analysis"
REPORT = ROOT / "report" / "error_analysis.md"


def select_cases(frame: pd.DataFrame, count: int = 10) -> pd.DataFrame:
    clean = frame[frame["condition"] == "clean"].set_index("source_id")
    clean_probability = clean["probability"].to_dict()
    clean_prediction = clean["prediction"].to_dict()
    clean_label = clean["label"].to_dict()
    candidates = []
    false_positive = frame[(frame["label"] == 0) & (frame["prediction"] == 1)].sort_values("probability", ascending=False).drop_duplicates("source_id").head(count)
    false_negative = frame[(frame["label"] == 1) & (frame["prediction"] == 0)].sort_values("probability", ascending=True).drop_duplicates("source_id").head(count)
    transformed = frame[frame["condition"] != "clean"].copy()
    transformed["clean_probability"] = transformed["source_id"].map(clean_probability)
    transformed["clean_prediction"] = transformed["source_id"].map(clean_prediction)
    transformed["clean_label"] = transformed["source_id"].map(clean_label)
    flips = transformed[(transformed["clean_prediction"] == transformed["clean_label"]) & (transformed["prediction"] != transformed["label"])].copy()
    flips["rank_score"] = (flips["probability"] - flips["clean_probability"]).abs()
    flips = flips.sort_values("rank_score", ascending=False).drop_duplicates("source_id").head(count)
    for error_type, selected in (("false_positive", false_positive), ("false_negative", false_negative), ("clean_to_transformed_flip", flips)):
        for _, row in selected.iterrows():
            candidates.append({
                "source_id": row["source_id"], "dataset": row["dataset"], "split": row["split"],
                "condition": row["condition"], "label": int(row["label"]),
                "clean_probability": float(clean_probability[row["source_id"]]),
                "transformed_probability": float(row["probability"]), "error_type": error_type,
            })
    result = pd.DataFrame(candidates)
    result.insert(0, "case_id", [f"EA-{index:03d}" for index in range(1, len(result) + 1)])
    return result


def locator_maps() -> tuple[dict[str, str], dict[tuple[str, str], int]]:
    manifest = pq.read_table(ROOT / "model" / "outputs" / "manifests" / "dataset_manifest.parquet", columns=["source_id", "native_locator"]).to_pylist()
    descriptors = pq.read_table(ROOT / "model" / "outputs" / "manifests" / "evaluation_descriptors.parquet", columns=["source_id", "condition_id", "seed"]).to_pylist()
    return ({row["source_id"]: row["native_locator"] for row in manifest}, {(row["source_id"], row["condition_id"]): int(row["seed"]) for row in descriptors})


def contact_sheet(cases: pd.DataFrame, path: Path) -> None:
    locators, seeds = locator_maps(); width, height = 240, 210; columns = 5
    rows = (len(cases) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), "white"); draw = ImageDraw.Draw(sheet); font = ImageFont.load_default()
    for position, case in cases.iterrows():
        condition = str(case["condition"])
        descriptor = {"native_locator": locators[case["source_id"]], "condition_id": condition, "seed": seeds[(case["source_id"], condition)]}
        image = _load_transformed(descriptor); image.thumbnail((220, 165), Image.Resampling.LANCZOS)
        x = (position % columns) * width + 10; y = (position // columns) * height + 5
        sheet.paste(image, (x + (220 - image.width) // 2, y))
        text = f"{case['case_id']} {case['error_type']}\n{condition}\ny={case['label']} clean={case['clean_probability']:.3f} shown={case['transformed_probability']:.3f}"
        draw.multiline_text((x, y + 168), text, fill="black", font=font, spacing=1)
    path.parent.mkdir(parents=True, exist_ok=True); sheet.save(path)


def write_report(cases: pd.DataFrame, frame: pd.DataFrame) -> None:
    counts = Counter(cases["error_type"]); condition_counts = Counter(cases["condition"])
    false_positive_ids = cases[cases["error_type"] == "false_positive"]["case_id"].tolist()
    false_negative_ids = cases[cases["error_type"] == "false_negative"]["case_id"].tolist()
    flip_ids = cases[cases["error_type"] == "clean_to_transformed_flip"]["case_id"].tolist()
    lines = [
        "# Stage 8 Error Analysis", "", "Status: required tier complete; optional saliency and manual tag vocabulary not run.", "",
        "## Scope", "", f"The frozen SID final-evaluation predictions contained {len(frame)} source-condition rows. This review selected {len(cases)} representative cases while allowing each source at most once per ranked list.", "",
        "- Highest-confidence false positives: " + ", ".join(false_positive_ids),
        "- Highest-confidence false negatives: " + ", ".join(false_negative_ids),
        "- Largest clean-to-transformed correctness flips: " + ", ".join(flip_ids), "",
        "Case metadata: `model/outputs/error_analysis/cases.csv`.", "", "## Recurring patterns", "",
        "The ranked failures concentrate in the conditions listed below. These are associations in the frozen model output, not proof that CLIP used a particular visible feature.", "",
        "| Condition | Selected cases |", "|---|---:|",
    ]
    lines.extend(f"| {condition} | {count} |" for condition, count in condition_counts.most_common())
    lines += ["", "## Visual review observations", "", "The highest-confidence false positives frequently have a polished or stylized appearance: shallow-depth-of-field food/product photographs, saturated portraits, signage or text, and sparse iconic compositions. These authentic images may resemble visual conventions represented unevenly in the training data.", "", "The false negatives include photorealistic people, sports and traffic scenes, decorative objects, and dark low-contrast imagery. Several look visually plausible at contact-sheet scale, so the errors cannot be reduced to one obvious anatomical or physical artifact.", "", "Seven of the ten largest clean-to-transformed flips use held-out WebP compression and three use added noise. Some sources recur across an error list and the flip list, demonstrating that a modest transformation can reverse a highly confident verdict."]
    lines += ["", "Visual review of the local contact sheet should be interpreted conservatively: similar composition, texture, compression, or subject matter can coexist with many unobserved embedding cues. The classifier head cannot be said to identify a specific anatomical or physical defect without a validated explanation method.", "", "## Authentic-image false-positive harms", "", "A false positive wrongly labels an authentic image as likely AI-generated. In moderation, journalism, education, or evidence review this can suppress legitimate work and make an accusation without provenance evidence. The detector should therefore be used as a triage signal, never as the sole basis for removal, penalties, or attribution.", "", "## Transformation sensitivity", "", f"The selected flip list contains {counts.get('clean_to_transformed_flip', 0)} distinct sources that were correct when clean and incorrect after a transformation. This demonstrates that aggregate robustness scores can hide individual instability. The Stage 7 paired evaluation recorded 177 such condition-level flips overall.", "", "## Limits of pixel-only detection", "", "A pixel classifier estimates similarity to patterns in its training distribution; it does not verify who created an image or how it was produced. Resizing, recompression, editing, screenshots, novel generators, and dataset shifts can alter those patterns. Provenance claims require trustworthy metadata, signatures, or content credentials in addition to model scores.", "", "## Licensing and repository status", "", "Raw SID images and the generated contact sheet remain under ignored `model/outputs/error_analysis/images/` and are not intended for Git. The tracked report references cases only by stable IDs and contains no absolute local paths.", "", "## Optional work", "", "- Manual artifact tags: Not run", "- Saliency/attention rollout: Not run", "- Public report imagery: Omitted pending explicit licensing review", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True); REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--count", type=int, default=10); args = parser.parse_args()
    frame = pd.read_csv(PREDICTIONS); cases = select_cases(frame, args.count)
    OUTPUT.mkdir(parents=True, exist_ok=True); cases.to_csv(OUTPUT / "cases.csv", index=False)
    contact_sheet(cases, OUTPUT / "images" / "contact_sheet.png"); write_report(cases, frame)
    print({"cases": len(cases), "counts": cases["error_type"].value_counts().to_dict(), "metadata": "model/outputs/error_analysis/cases.csv", "contact_sheet": "model/outputs/error_analysis/images/contact_sheet.png"})


if __name__ == "__main__":
    main()
