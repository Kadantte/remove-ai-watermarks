"""Contracts for the synthetic watermark benchmark cohort builder."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "watermark_benchmark_cohort.py"
SPEC = importlib.util.spec_from_file_location("watermark_benchmark_cohort", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_dwt_cohort_has_matched_positive_attack_and_hard_negative_arms(tmp_path: Path) -> None:
    output = tmp_path / "cohort"

    manifest = MODULE.build_cohort(
        output,
        adapters=("dwt-dct",),
        carriers=(MODULE.CarrierSpec("texture", 7),),
        dwt_schemes=("sdxl",),
        attacks=("jpeg-q90",),
    )
    rows = MODULE.load_manifest(manifest)

    assert len(rows) == 4
    assert {(row.arm, row.state, row.expected) for row in rows} == {
        ("matched_negative", "clean", "not_detected"),
        ("positive", "marked", "detected"),
        ("positive", "attacked", "unresolved"),
        ("hard_negative", "clean", "not_detected"),
    }
    assert all(row.source_revision.startswith("watermark-benchmark-cohort-v1@sha256:") for row in rows)
    assert all(row.path.is_relative_to(output) for row in rows)
    attacked = next(row for row in rows if row.state == "attacked")
    assert attacked.transform.parameters["carrier"] == "texture"
    assert attacked.transform.parameters["scheme"] == "sdxl"


def test_cohort_refuses_an_existing_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "cohort"
    output.mkdir()

    with pytest.raises(FileExistsError, match="output directory already exists"):
        MODULE.build_cohort(output, adapters=("dwt-dct",))


def test_cohort_size_is_explicit_in_artifacts_and_case_identity(tmp_path: Path) -> None:
    output = tmp_path / "cohort"

    manifest = MODULE.build_cohort(
        output,
        adapters=("dwt-dct",),
        carriers=(MODULE.CarrierSpec("texture", 7),),
        dwt_schemes=("sdxl",),
        attacks=("resize-75",),
        size=256,
    )
    rows = MODULE.load_manifest(manifest)

    assert all("256px" in row.case_id for row in rows)
    assert all(row.transform.parameters["size"] == [256, 256] for row in rows)
    for row in rows:
        with Image.open(row.path) as image:
            assert image.size == (256, 256)


def test_resize_attack_scales_relative_to_the_carrier() -> None:
    image = MODULE._carrier(MODULE.CarrierSpec("texture", 7), size=256)

    observed = MODULE.attack_image(image, "resize-75")
    expected = image.resize((192, 192), Image.Resampling.LANCZOS).resize((256, 256), Image.Resampling.LANCZOS)
    old_fixed_size = image.resize((384, 384), Image.Resampling.LANCZOS).resize((256, 256), Image.Resampling.LANCZOS)

    assert observed.tobytes() == expected.tobytes()
    assert observed.tobytes() != old_fixed_size.tobytes()


@pytest.mark.parametrize("size", [255, 258])
def test_cohort_rejects_unsupported_size(tmp_path: Path, size: int) -> None:
    with pytest.raises(ValueError, match="at least 256 and divisible by 16"):
        MODULE.build_cohort(tmp_path / f"cohort-{size}", adapters=("dwt-dct",), size=size)


def test_trustmark_cohort_includes_removed_observation(tmp_path: Path) -> None:
    if not MODULE.trustmark_available():
        pytest.skip("trustmark not installed")
    output = tmp_path / "cohort"

    manifest = MODULE.build_cohort(
        output,
        adapters=("trustmark",),
        carriers=(MODULE.CarrierSpec("gradient", 11),),
        attacks=(),
    )
    rows = MODULE.load_manifest(manifest)

    assert {(row.arm, row.state, row.expected) for row in rows} == {
        ("matched_negative", "clean", "not_detected"),
        ("positive", "marked", "detected"),
        ("positive", "removed", "unresolved"),
        ("hard_negative", "clean", "not_detected"),
    }
    removed = next(row for row in rows if row.state == "removed")
    assert removed.reference_path is not None
    assert removed.transform.name == "trustmark-remove"
