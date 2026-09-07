"""Contracts for the development-only watermark benchmark kernel."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "watermark_benchmark.py"
OFFICIAL_TRUSTMARK = Path(__file__).resolve().parents[1] / "data/fixtures/provenance/adobe-trustmark-p.png"
SPEC = importlib.util.spec_from_file_location("watermark_benchmark", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image(path: Path, value: int) -> Path:
    pixels = np.full((32, 40, 3), value, dtype=np.uint8)
    Image.fromarray(pixels, "RGB").save(path)
    return path


def _row(
    artifact: Path,
    *,
    reference: Path | None = None,
    case_id: str = "case-1",
    arm: str = "positive",
    state: str = "marked",
    expected: str = "detected",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "case_id": case_id,
        "pair_id": "pair-1",
        "media_type": "image",
        "adapter": "fake",
        "arm": arm,
        "state": state,
        "path": artifact.name,
        "sha256": _sha256(artifact),
        "reference_path": reference.name if reference is not None else None,
        "reference_sha256": _sha256(reference) if reference is not None else None,
        "source_revision": "fixture-set@abc123",
        "transform": {
            "name": "identity",
            "revision": "builtin-v1",
            "parameters": {},
        },
        "seed": 7,
        "expected": expected,
    }


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


class _FakeAdapter:
    name = "fake"
    source_file = SCRIPT

    def __init__(self, status: str = "detected", label: str | None = "fake mark") -> None:
        self.status = status
        self.label = label
        self.calls: list[Path] = []
        self.images: list[Any] = []

    @property
    def available(self) -> bool:
        return self.status != "unavailable"

    def detect(self, path: Path, image: Any) -> Any:
        self.calls.append(path)
        self.images.append(image)
        return MODULE.DetectorOutcome(status=self.status, label=self.label)


def test_manifest_preserves_explicit_arms_states_and_revisions(tmp_path: Path) -> None:
    artifact = _image(tmp_path / "marked.png", 100)
    reference = _image(tmp_path / "clean.png", 90)
    manifest = _write_manifest(
        tmp_path / "manifest.jsonl",
        [
            _row(artifact, reference=reference),
            _row(
                reference,
                case_id="case-2",
                arm="hard_negative",
                state="clean",
                expected="not_detected",
            ),
        ],
    )

    rows = MODULE.load_manifest(manifest)

    assert [row.arm for row in rows] == ["positive", "hard_negative"]
    assert [row.state for row in rows] == ["marked", "clean"]
    assert rows[0].path == artifact.resolve()
    assert rows[0].reference_path == reference.resolve()
    assert rows[0].source_revision == "fixture-set@abc123"
    assert rows[0].transform.revision == "builtin-v1"
    assert rows[0].seed == 7


@pytest.mark.parametrize("arm", ["positive", "matched_negative", "wrong_key", "hard_negative"])
def test_manifest_supports_every_evidence_arm(tmp_path: Path, arm: str) -> None:
    artifact = _image(tmp_path / f"{arm}.png", 80)
    expected = "detected" if arm == "positive" else "not_detected"
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [_row(artifact, arm=arm, expected=expected)])

    assert MODULE.load_manifest(manifest)[0].arm == arm


def test_manifest_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    first = _image(tmp_path / "first.png", 1)
    second = _image(tmp_path / "second.png", 2)
    manifest = _write_manifest(
        tmp_path / "manifest.jsonl",
        [_row(first), _row(second)],
    )

    with pytest.raises(ValueError, match="duplicate case_id"):
        MODULE.load_manifest(manifest)


def test_manifest_rejects_artifact_hash_drift(tmp_path: Path) -> None:
    artifact = _image(tmp_path / "marked.png", 100)
    row = _row(artifact)
    row["sha256"] = "0" * 64
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [row])

    with pytest.raises(ValueError, match="sha256 mismatch"):
        MODULE.load_manifest(manifest)


def test_manifest_requires_reference_path_and_hash_together(tmp_path: Path) -> None:
    artifact = _image(tmp_path / "marked.png", 100)
    row = _row(artifact)
    row["reference_path"] = "clean.png"
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [row])

    with pytest.raises(ValueError, match="reference_path and reference_sha256"):
        MODULE.load_manifest(manifest)


def test_manifest_rejects_unknown_fields(tmp_path: Path) -> None:
    artifact = _image(tmp_path / "marked.png", 100)
    row = _row(artifact)
    row["implicit_default"] = True
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [row])

    with pytest.raises(ValueError, match="expected exactly"):
        MODULE.load_manifest(manifest)


@pytest.mark.parametrize(
    ("manifest_text", "message"),
    [
        ('{"schema_version":1,"schema_version":1}\n', "duplicate field"),
        ('{"schema_version":1,"seed":NaN}\n', "non-finite number"),
    ],
)
def test_manifest_rejects_non_strict_json(tmp_path: Path, manifest_text: str, message: str) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(manifest_text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        MODULE.load_manifest(manifest)


def test_evaluation_keeps_detection_removal_and_fidelity_separate(tmp_path: Path) -> None:
    reference = _image(tmp_path / "clean.png", 90)
    removed = _image(tmp_path / "removed.png", 100)
    manifest = _write_manifest(
        tmp_path / "manifest.jsonl",
        [_row(removed, reference=reference, state="removed", expected="not_detected")],
    )
    case = MODULE.load_manifest(manifest)[0]
    adapter = _FakeAdapter(status="not_detected", label=None)
    times = iter((100, 500))

    record = MODULE.evaluate_case(
        case,
        adapters={"fake": adapter},
        repository={"commit": "abc123", "dirty": False},
        clock_ns=lambda: next(times),
    )

    assert adapter.calls == [removed.resolve()]
    assert adapter.images[0].shape == (32, 40, 3)
    assert record["detection"] == {
        "status": "not_detected",
        "label": None,
        "expected": "not_detected",
        "matches_expected": True,
        "positive_evidence": False,
        "adapter_elapsed_ms": 0.0004,
    }
    assert record["removal"] == {
        "attempted": True,
        "status": "no_recognized_signal_after_removal",
        "certifies_erasure": False,
    }
    assert record["fidelity"]["status"] == "measured"
    assert record["fidelity"]["identical"] is False
    assert record["fidelity"]["mae_8bit"] == pytest.approx(10.0)
    assert record["fidelity"]["changed_fraction"] == pytest.approx(1.0)
    assert record["fidelity"]["psnr_db"] is not None
    assert "clean" not in record


def test_identical_pair_marks_psnr_as_unbounded_not_zero(tmp_path: Path) -> None:
    artifact = _image(tmp_path / "same.png", 90)
    manifest = _write_manifest(
        tmp_path / "manifest.jsonl",
        [_row(artifact, reference=artifact, state="clean", expected="not_detected")],
    )

    record = MODULE.evaluate_case(
        MODULE.load_manifest(manifest)[0],
        adapters={"fake": _FakeAdapter(status="not_detected", label=None)},
        repository={"commit": "abc123", "dirty": False},
    )

    assert record["fidelity"] == {
        "status": "measured",
        "identical": True,
        "mae_8bit": 0.0,
        "changed_fraction": 0.0,
        "psnr_db": None,
        "psnr_status": "unbounded_identical",
    }


def test_identical_artifact_and_reference_are_decoded_once(tmp_path: Path) -> None:
    artifact = _image(tmp_path / "same.png", 90)
    case = MODULE.load_manifest(
        _write_manifest(
            tmp_path / "manifest.jsonl",
            [_row(artifact, reference=artifact, state="clean", expected="not_detected")],
        )
    )[0]
    calls: list[Path] = []

    def decode(path: Path) -> Any:
        calls.append(path)
        return np.full((32, 40, 3), 90, dtype=np.uint8)

    MODULE.evaluate_case(
        case,
        adapters={"fake": _FakeAdapter(status="not_detected", label=None)},
        repository={"commit": "abc123", "dirty": False},
        artifact_decoder=decode,
        reference_decoder=lambda _path: pytest.fail("identical reference was decoded again"),
    )

    assert calls == [artifact.resolve()]


def test_missing_reference_is_explicitly_unmeasured(tmp_path: Path) -> None:
    artifact = _image(tmp_path / "marked.png", 100)
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [_row(artifact)])

    record = MODULE.evaluate_case(
        MODULE.load_manifest(manifest)[0],
        adapters={"fake": _FakeAdapter()},
        repository={"commit": "abc123", "dirty": True},
    )

    assert record["fidelity"] == {"status": "not_measured", "reason": "reference_not_provided"}
    assert record["run"]["repository"] == {"commit": "abc123", "dirty": True}


def test_unavailable_adapter_is_not_reported_as_negative(tmp_path: Path) -> None:
    artifact = _image(tmp_path / "marked.png", 100)
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [_row(artifact)])
    adapter = _FakeAdapter(status="unavailable", label=None)

    record = MODULE.evaluate_case(
        MODULE.load_manifest(manifest)[0],
        adapters={"fake": adapter},
        repository={"commit": "abc123", "dirty": False},
    )

    assert record["detection"]["status"] == "unavailable"
    assert record["detection"]["matches_expected"] is None
    assert record["detection"]["adapter_elapsed_ms"] is None
    assert adapter.calls == []


def test_undecodable_artifact_is_error_not_negative(tmp_path: Path) -> None:
    artifact = tmp_path / "broken.png"
    artifact.write_bytes(b"not an image")
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [_row(artifact)])
    adapter = _FakeAdapter(status="not_detected", label=None)

    record = MODULE.evaluate_case(
        MODULE.load_manifest(manifest)[0],
        adapters={"fake": adapter},
        repository={"commit": "abc123", "dirty": False},
    )

    assert record["detection"]["status"] == "error"
    assert record["detection"]["matches_expected"] is None
    assert record["detection"]["error"] == "artifact could not be decoded"
    assert record["detection"]["adapter_elapsed_ms"] is None
    assert adapter.calls == []


def test_detector_disagreement_is_recorded_not_reinterpreted(tmp_path: Path) -> None:
    artifact = _image(tmp_path / "marked.png", 100)
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [_row(artifact, expected="detected")])

    record = MODULE.evaluate_case(
        MODULE.load_manifest(manifest)[0],
        adapters={"fake": _FakeAdapter(status="not_detected", label=None)},
        repository={"commit": "abc123", "dirty": False},
    )

    assert record["detection"]["status"] == "not_detected"
    assert record["detection"]["matches_expected"] is False


def test_shape_mismatch_is_incomparable(tmp_path: Path) -> None:
    reference = _image(tmp_path / "clean.png", 90)
    artifact = tmp_path / "different-shape.png"
    Image.fromarray(np.full((16, 20, 3), 90, dtype=np.uint8), "RGB").save(artifact)
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [_row(artifact, reference=reference)])

    record = MODULE.evaluate_case(
        MODULE.load_manifest(manifest)[0],
        adapters={"fake": _FakeAdapter()},
        repository={"commit": "abc123", "dirty": False},
    )

    assert record["fidelity"] == {
        "status": "incomparable",
        "reason": "shape_mismatch",
        "artifact_shape": [16, 20, 3],
        "reference_shape": [32, 40, 3],
    }


def test_non_8_bit_pair_is_incomparable(tmp_path: Path) -> None:
    reference = tmp_path / "clean.png"
    artifact = tmp_path / "removed.png"
    Image.fromarray(np.full((16, 20), 900, dtype=np.uint16)).save(reference)
    Image.fromarray(np.full((16, 20), 901, dtype=np.uint16)).save(artifact)
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [_row(artifact, reference=reference)])

    record = MODULE.evaluate_case(
        MODULE.load_manifest(manifest)[0],
        adapters={"fake": _FakeAdapter()},
        repository={"commit": "abc123", "dirty": False},
    )

    assert record["fidelity"] == {
        "status": "incomparable",
        "reason": "unsupported_dtype",
        "artifact_dtype": "uint16",
        "reference_dtype": "uint16",
    }


def test_jsonl_writer_refuses_overwrite_and_emits_strict_json(tmp_path: Path) -> None:
    output = tmp_path / "report.jsonl"
    records = [{"schema_version": 1, "value": None}]

    MODULE.write_jsonl(output, records)
    assert output.read_text(encoding="utf-8") == '{"schema_version": 1, "value": null}\n'

    with pytest.raises(FileExistsError):
        MODULE.write_jsonl(output, records)


def test_builtin_adapters_name_their_exact_source_files() -> None:
    adapters = MODULE.default_adapters()

    assert set(adapters) == {"dwt-dct", "trustmark"}
    assert adapters["dwt-dct"].source_file.name == "invisible_watermark.py"
    assert adapters["trustmark"].source_file.name == "trustmark_detector.py"


def test_run_benchmark_caches_only_recurring_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recurring = _image(tmp_path / "recurring.png", 90)
    unique = _image(tmp_path / "unique.png", 100)
    manifest = _write_manifest(
        tmp_path / "manifest.jsonl",
        [
            _row(recurring, case_id="case-1", state="clean", expected="not_detected"),
            _row(recurring, case_id="case-2", state="clean", expected="not_detected"),
            _row(unique, case_id="case-3", state="clean", expected="not_detected"),
        ],
    )
    real_decode = MODULE._decode_image
    calls: list[Path] = []

    def decode(path: Path) -> Any:
        calls.append(path)
        return real_decode(path)

    monkeypatch.setattr(MODULE, "_decode_image", decode)
    monkeypatch.setattr(MODULE, "default_adapters", lambda: {"fake": _FakeAdapter("not_detected", None)})
    monkeypatch.setattr(MODULE, "repository_state", lambda: {"commit": "abc123", "dirty": False})

    MODULE.run_benchmark(manifest, tmp_path / "results.jsonl")

    assert calls.count(recurring.resolve()) == 1
    assert calls.count(unique.resolve()) == 1


def test_run_benchmark_writes_official_trustmark_result(tmp_path: Path) -> None:
    if not MODULE.default_adapters()["trustmark"].available:
        pytest.skip("trustmark not installed")
    row = _row(OFFICIAL_TRUSTMARK)
    row.update(
        {
            "adapter": "trustmark",
            "path": str(OFFICIAL_TRUSTMARK),
            "source_revision": "adobe/trustmark@0ed40cbe8188f664fd9cbbeacd969807de27440a",
        }
    )
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [row])
    output = tmp_path / "result.jsonl"

    counts = MODULE.run_benchmark(manifest, output)
    record = json.loads(output.read_text(encoding="utf-8"))

    assert counts == {"detected": 1}
    assert record["detection"]["status"] == "detected"
    assert record["detection"]["label"] == "Adobe TrustMark (variant P, schema 1)"
    assert record["case_id"] == "case-1"
