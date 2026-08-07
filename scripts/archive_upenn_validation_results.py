"""Validate and archive the 10-case UPENN-GBM Colab evaluation results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

RESULT_ARCHIVE = "BrainTumor-Agent-UPENN-10-five-fold-GPU-validation.zip"
EVALUATION_JSON = "evaluation-upenn-10-cases.json"
EVALUATION_CSV = "evaluation-upenn-10-cases.csv"
EXPECTED_CASES = (
    "UPENN-GBM-00002_11",
    "UPENN-GBM-00006_11",
    "UPENN-GBM-00008_11",
    "UPENN-GBM-00009_11",
    "UPENN-GBM-00011_11",
    "UPENN-GBM-00013_11",
    "UPENN-GBM-00014_11",
    "UPENN-GBM-00016_11",
    "UPENN-GBM-00017_11",
    "UPENN-GBM-00018_11",
)
METRICS = ("whole_tumor", "tumor_core", "enhancing_tumor")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _metric(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{context} must be between 0 and 1")
    return result


def validate_evaluation(payload: Mapping[str, object]) -> dict[str, object]:
    cases = payload.get("cases")
    summary = payload.get("summary")
    metadata = payload.get("metadata")
    if not isinstance(cases, list) or not isinstance(summary, dict):
        raise ValueError("Evaluation JSON must contain cases and summary")
    if not isinstance(metadata, dict):
        raise ValueError("Evaluation JSON must contain metadata")

    case_rows: dict[str, dict[str, float]] = {}
    for raw_case in cases:
        if not isinstance(raw_case, dict) or not isinstance(raw_case.get("case_id"), str):
            raise ValueError("Each case must contain a case_id")
        case_id = raw_case["case_id"]
        if case_id in case_rows:
            raise ValueError(f"Duplicate case in evaluation JSON: {case_id}")
        case_rows[case_id] = {
            metric: _metric(raw_case.get(metric), f"{case_id}.{metric}")
            for metric in METRICS
        }

    if tuple(sorted(case_rows)) != EXPECTED_CASES:
        missing = sorted(set(EXPECTED_CASES) - set(case_rows))
        unexpected = sorted(set(case_rows) - set(EXPECTED_CASES))
        raise ValueError(f"UPENN cohort differs; missing={missing}, unexpected={unexpected}")
    if summary.get("num_cases") != len(EXPECTED_CASES):
        raise ValueError("Summary num_cases must equal 10")

    summary_metrics: dict[str, float] = {}
    for metric in METRICS:
        actual = _metric(summary.get(metric), f"summary.{metric}")
        expected = sum(row[metric] for row in case_rows.values()) / len(case_rows)
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Summary {metric} does not match the case macro average")
        summary_metrics[metric] = actual

    required_metadata = {
        "device": "cuda",
        "inference_folds": "0,1,2,3,4",
        "output_label_profile": "brats19_preserved",
        "validation_doi": "10.7937/TCIA.709X-DN49",
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"metadata.{key} must equal {expected!r}")
    if not isinstance(metadata.get("gpu_name"), str) or not metadata["gpu_name"].strip():
        raise ValueError("metadata.gpu_name is required")

    return {
        "cases": case_rows,
        "summary": summary_metrics,
        "metadata": metadata,
    }


def validate_csv(path: Path, expected_cases: Mapping[str, Mapping[str, float]]) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    actual_ids = [row.get("case_id", "") for row in rows]
    if sorted(actual_ids) != list(EXPECTED_CASES):
        raise ValueError("Evaluation CSV cohort does not match the expected UPENN cases")
    for row in rows:
        case_id = row["case_id"]
        for metric in METRICS:
            actual = _metric(float(row[metric]), f"CSV {case_id}.{metric}")
            if not math.isclose(
                actual,
                expected_cases[case_id][metric],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"CSV and JSON differ for {case_id}.{metric}")


def validate_result_zip(path: Path, json_bytes: bytes, csv_bytes: bytes) -> None:
    expected_masks = {f"brats_predictions/{case_id}.nii.gz" for case_id in EXPECTED_CASES}
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"Corrupt entry in result ZIP: {corrupt}")
        names = set(archive.namelist())
        missing_masks = sorted(expected_masks - names)
        if missing_masks:
            raise ValueError(f"Result ZIP is missing prediction masks: {missing_masks}")
        if archive.read("evaluation.json") != json_bytes:
            raise ValueError("External evaluation JSON differs from the copy inside the ZIP")
        if archive.read("evaluation_cases.csv") != csv_bytes:
            raise ValueError("External evaluation CSV differs from the copy inside the ZIP")


def validate_results(results_dir: Path) -> tuple[dict[str, object], dict[str, Path]]:
    files = {
        "archive": results_dir / RESULT_ARCHIVE,
        "json": results_dir / EVALUATION_JSON,
        "csv": results_dir / EVALUATION_CSV,
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Colab result files:\n" + "\n".join(missing))

    json_bytes = files["json"].read_bytes()
    csv_bytes = files["csv"].read_bytes()
    payload = json.loads(json_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Evaluation JSON root must be an object")
    validated = validate_evaluation(payload)
    validate_csv(files["csv"], validated["cases"])
    validate_result_zip(files["archive"], json_bytes, csv_bytes)
    return validated, files


def build_readme(validated: Mapping[str, object], files: Mapping[str, Path]) -> str:
    cases = validated["cases"]
    summary = validated["summary"]
    metadata = validated["metadata"]
    assert isinstance(cases, dict)
    assert isinstance(summary, dict)
    assert isinstance(metadata, dict)
    lines = [
        "# UPENN-GBM 10-case five-fold GPU validation",
        "",
        f"- GPU: {metadata['gpu_name']}",
        "- Folds: 0, 1, 2, 3, 4",
        "- Dataset: UPENN-GBM expert-reviewed masks",
        "- DOI: 10.7937/TCIA.709X-DN49",
        "- Scope: independent 10-case sample; not clinical validation",
        "",
        "## Dice",
        "",
        "| Case | WT | TC | ET |",
        "| --- | ---: | ---: | ---: |",
    ]
    for case_id in EXPECTED_CASES:
        row = cases[case_id]
        lines.append(
            f"| `{case_id}` | {row['whole_tumor']:.6f} | "
            f"{row['tumor_core']:.6f} | {row['enhancing_tumor']:.6f} |"
        )
    lines.extend(
        [
            f"| **Macro average** | **{summary['whole_tumor']:.6f}** | "
            f"**{summary['tumor_core']:.6f}** | **{summary['enhancing_tumor']:.6f}** |",
            "",
            "## SHA-256",
            "",
            "| File | SHA-256 |",
            "| --- | --- |",
        ]
    )
    for path in files.values():
        lines.append(f"| `{path.name}` | `{sha256(path)}` |")
    return "\n".join(lines) + "\n"


def archive_results(results_dir: Path, archive_dir: Path) -> Path:
    validated, files = validate_results(results_dir.resolve())
    destination = archive_dir.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty archive directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    copied = {key: shutil.copy2(path, destination / path.name) for key, path in files.items()}
    copied_paths = {key: Path(path) for key, path in copied.items()}
    (destination / "README.md").write_text(
        build_readme(validated, copied_paths),
        encoding="utf-8",
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.archive_dir is None:
        validated, files = validate_results(args.results_dir.resolve())
        summary = validated["summary"]
        print(f"Validated {len(validated['cases'])} UPENN cases")
        print(
            "Macro Dice: "
            f"WT={summary['whole_tumor']:.6f}, "
            f"TC={summary['tumor_core']:.6f}, "
            f"ET={summary['enhancing_tumor']:.6f}"
        )
        for path in files.values():
            print(f"{path.name}: {sha256(path)}")
    else:
        destination = archive_results(args.results_dir, args.archive_dir)
        print(f"Validated and archived results: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
