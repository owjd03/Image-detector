import pandas as pd

from model.scripts.analyze_errors import select_cases


def test_rankings_are_source_unique_and_include_flips() -> None:
    rows = []
    for source, label, clean_probability, transformed_probability in (("a", 0, .1, .9), ("b", 1, .9, .1), ("c", 0, .8, .9)):
        rows.append({"source_id": source, "dataset": "sid", "split": "test", "condition": "clean", "label": label, "probability": clean_probability, "prediction": int(clean_probability >= .5)})
        rows.append({"source_id": source, "dataset": "sid", "split": "test", "condition": "jpeg_q30", "label": label, "probability": transformed_probability, "prediction": int(transformed_probability >= .5)})
    selected = select_cases(pd.DataFrame(rows), count=2)
    assert not selected.groupby("error_type")["source_id"].apply(lambda values: values.duplicated().any()).any()
    assert "clean_to_transformed_flip" in set(selected["error_type"])
