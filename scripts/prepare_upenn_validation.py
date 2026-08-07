"""Download and validate a small independent UPENN-GBM evaluation cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import requests

HF_REPO = "MedOtter/UPENN-GBM"
HF_REVISION = "662f85bd477f2f6306c0702d2e44301b716ae45c"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}"
TCIA_COLLECTION = "https://www.cancerimagingarchive.net/collection/upenn-gbm/"
TCIA_DOI = "10.7937/TCIA.709X-DN49"


@dataclass(frozen=True)
class CaseFiles:
    flair: int
    t1: int
    t1gd: int
    t2: int
    seg: int


# Sizes were read from TCIA Faspex package 604 before selecting the cohort.
CASES: dict[str, CaseFiles] = {
    "UPENN-GBM-00002_11": CaseFiles(1900090, 2054596, 2151118, 1775300, 51227),
    "UPENN-GBM-00006_11": CaseFiles(2366528, 2416985, 2443921, 2247836, 31872),
    "UPENN-GBM-00008_11": CaseFiles(1807501, 1899781, 1917654, 1609983, 25285),
    "UPENN-GBM-00009_11": CaseFiles(2450932, 2469512, 2511599, 2371660, 45429),
    "UPENN-GBM-00011_11": CaseFiles(2200627, 2270072, 2290159, 2031305, 36583),
    "UPENN-GBM-00013_11": CaseFiles(2036183, 2120229, 2166239, 1992761, 24881),
    "UPENN-GBM-00014_11": CaseFiles(2134091, 2230594, 2254170, 2047797, 46472),
    "UPENN-GBM-00016_11": CaseFiles(1954535, 2034843, 2069712, 1900622, 27243),
    "UPENN-GBM-00017_11": CaseFiles(2433383, 2436076, 2472195, 2223509, 26295),
    "UPENN-GBM-00018_11": CaseFiles(2000687, 2050001, 2083163, 1917694, 59067),
}

MODALITIES = {
    "flair": ("FLAIR", "flair"),
    "t1": ("T1", "t1"),
    "t1gd": ("T1GD", "t1ce"),
    "t2": ("T2", "t2"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runtime/external_validation/UPENN-GBM"),
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--colab-zip", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_file(case_id: str, role: str, root: Path) -> tuple[Path, str, int]:
    sizes = CASES[case_id]
    if role == "seg":
        name = f"{case_id}_segm.nii.gz"
        relative = f"images_segm/{name}"
        expected = sizes.seg
    else:
        source_suffix, _ = MODALITIES[role]
        name = f"{case_id}_{source_suffix}.nii.gz"
        relative = f"images_structural/{case_id}/{name}"
        expected = getattr(sizes, role)
    return root / "raw" / case_id / name, relative, expected


def download_one(case_id: str, role: str, root: Path) -> dict[str, object]:
    target, relative, expected = raw_file(case_id, role, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != expected:
            raise RuntimeError(f"Existing file has unexpected size: {target}")
    else:
        temp = target.with_suffix(target.suffix + ".partial")
        if temp.exists():
            raise RuntimeError(f"Partial file already exists: {temp}")
        url = f"{HF_BASE}/{relative}?download=true"
        with requests.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with temp.open("xb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        stream.write(chunk)
        if temp.stat().st_size != expected:
            raise RuntimeError(
                f"Downloaded size mismatch for {relative}: "
                f"{temp.stat().st_size} != {expected}"
            )
        temp.replace(target)
    return {
        "case_id": case_id,
        "role": role,
        "source_path": relative,
        "expected_tcia_bytes": expected,
        "actual_bytes": target.stat().st_size,
        "sha256": sha256(target),
        "raw_path": str(target.relative_to(root)).replace("\\", "/"),
    }


def link_standardized_files(root: Path) -> None:
    cases_root = root / "cases"
    for case_id in CASES:
        destinations: list[tuple[Path, Path]] = []
        for role, (_, target_suffix) in MODALITIES.items():
            source, _, _ = raw_file(case_id, role, root)
            destinations.append(
                (source, cases_root / case_id / f"{case_id}_{target_suffix}.nii.gz")
            )
        source, _, _ = raw_file(case_id, "seg", root)
        destinations.append((source, cases_root / case_id / f"{case_id}_seg.nii.gz"))
        for source, destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.stat().st_size != source.stat().st_size:
                    raise RuntimeError(f"Existing standardized file differs: {destination}")
                continue
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)


def validate_nifti(root: Path) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    allowed_labels = {0, 1, 2, 4}
    for case_id in CASES:
        loaded: dict[str, nib.spatialimages.SpatialImage] = {}
        for role in (*MODALITIES, "seg"):
            path, _, _ = raw_file(case_id, role, root)
            loaded[role] = nib.load(path)
        reference = loaded["flair"]
        for role, image in loaded.items():
            if image.shape != reference.shape:
                raise RuntimeError(f"Shape mismatch in {case_id}: {role}")
            if not np.allclose(image.affine, reference.affine, atol=1e-5):
                raise RuntimeError(f"Affine mismatch in {case_id}: {role}")
        labels = {int(value) for value in np.unique(np.asanyarray(loaded["seg"].dataobj))}
        if not labels.issubset(allowed_labels) or not labels - {0}:
            raise RuntimeError(f"Invalid labels in {case_id}: {sorted(labels)}")
        results.append(
            {
                "case_id": case_id,
                "shape": list(reference.shape),
                "voxel_sizes_mm": [float(x) for x in reference.header.get_zooms()[:3]],
                "labels": sorted(labels),
                "grid_consistent": True,
            }
        )
    return results


def write_manifests(
    root: Path,
    files: list[dict[str, object]],
    validation: list[dict[str, object]],
) -> None:
    manifest_root = root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    source = {
        "dataset": "UPENN-GBM",
        "cohort": "10 baseline cases with manual/expert segmentations",
        "case_ids": list(CASES),
        "tcia_collection": TCIA_COLLECTION,
        "tcia_doi": TCIA_DOI,
        "tcia_faspex_package_id": "604",
        "license": "CC BY 4.0",
        "mirror": f"https://huggingface.co/datasets/{HF_REPO}",
        "mirror_revision": HF_REVISION,
        "ground_truth": "images_segm manual/expert masks",
        "label_scheme": {
            "0": "background",
            "1": "NCR/NET",
            "2": "edema/infiltrated tissue",
            "4": "enhancing tumor",
        },
        "evaluation_regions": {"WT": [1, 2, 4], "TC": [1, 4], "ET": [4]},
        "training_overlap_statement": (
            "UPENN-GBM is an institutional cohort independent of the old BraTS/MSD "
            "sources used by Dataset002_BRATS19. Case identifiers are preserved."
        ),
    }
    (manifest_root / "source_and_license.json").write_text(
        json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    payload = {
        "source": source,
        "total_bytes": sum(int(item["actual_bytes"]) for item in files),
        "files": sorted(files, key=lambda item: (str(item["case_id"]), str(item["role"]))),
        "nifti_validation": validation,
    }
    (manifest_root / "files_sha256.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_colab_zip(root: Path, target: Path) -> None:
    target = target.resolve()
    if target.exists():
        validate_colab_zip(root, target)
        print(f"Validated existing Colab archive: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted((root / "cases").rglob("*.nii.gz")):
            relative = path.relative_to(root / "cases")
            archive.write(path, Path("validation-cases") / relative)
        for path in sorted((root / "manifests").glob("*.json")):
            archive.write(path, Path("validation-manifests") / path.name)
    validate_colab_zip(root, target)


def validate_colab_zip(root: Path, target: Path) -> None:
    expected = {
        str(Path("validation-cases") / path.relative_to(root / "cases")).replace("\\", "/")
        for path in (root / "cases").rglob("*.nii.gz")
    }
    expected.update(
        str(Path("validation-manifests") / path.name).replace("\\", "/")
        for path in (root / "manifests").glob("*.json")
    )
    with zipfile.ZipFile(target) as archive:
        corrupt = archive.testzip()
        actual = {name for name in archive.namelist() if not name.endswith("/")}
    if corrupt is not None:
        raise RuntimeError(f"Corrupt file in Colab archive: {corrupt}")
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(
            f"Colab archive contents differ; missing={missing}, unexpected={unexpected}"
        )


def main() -> None:
    args = parse_args()
    root = args.output.resolve()
    jobs = [(case_id, role) for case_id in CASES for role in (*MODALITIES, "seg")]
    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(download_one, case_id, role, root): (case_id, role)
            for case_id, role in jobs
        }
        for future in as_completed(futures):
            case_id, role = futures[future]
            records.append(future.result())
            print(f"ready {case_id} {role}")
    validation = validate_nifti(root)
    link_standardized_files(root)
    write_manifests(root, records, validation)
    if args.colab_zip is not None:
        write_colab_zip(root, args.colab_zip)
    total = sum(int(item["actual_bytes"]) for item in records)
    print(f"Prepared {len(CASES)} cases / {len(records)} files / {total / 1024**2:.2f} MiB")
    print(f"Standardized cases: {root / 'cases'}")
    print(f"Manifest: {root / 'manifests' / 'files_sha256.json'}")
    if args.colab_zip is not None:
        print(f"Colab archive: {args.colab_zip.resolve()}")


if __name__ == "__main__":
    main()
