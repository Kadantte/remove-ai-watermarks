"""Contracts for the registered visible-mark calibration harness."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "registered_mark_calibrate.py"
SPEC = importlib.util.spec_from_file_location("registered_mark_calibrate", SCRIPT)
assert SPEC
assert SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_manifest_preserves_evidence_arms(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"path": "carrier.png", "arm": "positive"}),
                json.dumps({"path": "provider.png", "arm": "metadata"}),
                json.dumps({"path": "comparison.png", "arm": "control"}),
            ]
        ),
        encoding="utf-8",
    )

    rows = module.load_manifest(manifest)

    assert [row["arm"] for row in rows] == ["positive", "metadata", "control"]
    assert all(Path(row["path"]).is_absolute() for row in rows)


def test_manifest_rejects_one_file_in_multiple_arms(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"path": "same.png", "arm": "positive"}),
                json.dumps({"path": "same.png", "arm": "control"}),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate image path"):
        module.load_manifest(manifest)


def test_summary_never_relabels_a_control() -> None:
    summary = module.summarize([0.1, 0.4], unreadable=1, fires=1)

    assert summary["n"] == 2
    assert summary["unreadable"] == 1
    assert summary["fires"] == 1
