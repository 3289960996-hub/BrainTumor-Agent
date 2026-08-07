import csv
import json
import zipfile
from pathlib import Path

import pytest

from scripts.archive_upenn_validation_results import (
    EVALUATION_CSV,
    EVALUATION_JSON,
    EXPECTED_CASES,
    METRICS,
    RESULT_ARCHIVE,
    validate_results,
)


def _write_results(root: Path) -> dict[str, object]:
    cases = [
        {
            "case_id": case_id,
            "whole_tumor": 0.80 + index / 100,
            "tumor_core": 0.70 + index / 100,
            "enhancing_tumor": 0.60 + index / 100,
        }
        for index, case_id in enumerate(EXPECTED_CASES)
    ]
    payload: dict[str, object] = {
        "summary": {
            "num_cases": len(cases),
            **{
                metric: sum(float(case[metric]) for case in cases) / len(cases)
                for metric in METRICS
            },
        },
        "cases": cases,
        "metadata": {
            "device": "cuda",
            "gpu_name": "Tesla T4",
            "inference_folds": "0,1,2,3,4",
            "output_label_profile": "brats19_preserved",
            "validation_doi": "10.7937/TCIA.709X-DN49",
        },
    }
    json_path = root / EVALUATION_JSON
    csv_path = root / EVALUATION_CSV
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("case_id", *METRICS))
        writer.writeheader()
        writer.writerows(cases)
    with zipfile.ZipFile(root / RESULT_ARCHIVE, "w") as archive:
        archive.writestr("evaluation.json", json_path.read_bytes())
        archive.writestr("evaluation_cases.csv", csv_path.read_bytes())
        for case_id in EXPECTED_CASES:
            archive.writestr(f"brats_predictions/{case_id}.nii.gz", b"test-mask")
    return payload


def test_validate_results_accepts_complete_colab_output(tmp_path: Path) -> None:
    _write_results(tmp_path)

    validated, files = validate_results(tmp_path)

    assert tuple(sorted(validated["cases"])) == EXPECTED_CASES
    assert set(files) == {"archive", "json", "csv"}


def test_validate_results_rejects_incorrect_macro_average(tmp_path: Path) -> None:
    payload = _write_results(tmp_path)
    payload["summary"]["tumor_core"] = 0.1
    (tmp_path / EVALUATION_JSON).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="macro average"):
        validate_results(tmp_path)
