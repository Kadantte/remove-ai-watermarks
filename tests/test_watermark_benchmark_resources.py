"""Contracts for the watermark benchmark process-resource profiler."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "watermark_benchmark_resources.py"
SPEC = importlib.util.spec_from_file_location("watermark_benchmark_resources", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
BENCHMARK = sys.modules["watermark_benchmark"]


def _manifest(path: Path) -> Path:
    artifact = path.parent / "clean.png"
    Image.fromarray(np.full((256, 256, 3), 127, dtype=np.uint8), "RGB").save(artifact)
    row = {
        "schema_version": 1,
        "case_id": "dwt-dct--resource--256px--clean",
        "pair_id": "dwt-dct--resource--256px",
        "media_type": "image",
        "adapter": "dwt-dct",
        "arm": "hard_negative",
        "state": "clean",
        "path": artifact.name,
        "sha256": BENCHMARK.sha256_file(artifact),
        "reference_path": None,
        "reference_sha256": None,
        "source_revision": "resource-test@v1",
        "transform": {
            "name": "synthetic-carrier",
            "revision": "resource-test-v1",
            "parameters": {"size": [256, 256]},
        },
        "seed": 7,
        "expected": "not_detected",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def test_profiler_runs_benchmark_in_a_fresh_process(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.jsonl")
    output = tmp_path / "profile"

    rows = MODULE.profile_manifests([manifest], output, repeat=1)

    assert len(rows) == 1
    assert rows[0]["case_count"] == 1
    assert rows[0]["max_width"] == 256
    assert rows[0]["max_height"] == 256
    assert rows[0]["process"]["wall_ms"] > 0
    assert rows[0]["profiler_source_sha256"] == BENCHMARK.sha256_file(SCRIPT)
    assert rows[0]["process"]["worker_pid"] != os.getpid()
    if sys.platform in ("darwin", "linux"):
        assert rows[0]["process"]["peak_rss_mib"] > 0
    assert sum(rows[0]["status_counts"].values()) == 1
    assert (output / "manifest-01-run-01.jsonl").is_file()
    assert (output / "resources.jsonl").is_file()
    report = (output / "resources.md").read_text(encoding="utf-8")
    assert "absolute high-water mark" in report
    assert "256x256" in report


def test_profiler_rejects_invalid_repeat_before_creating_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repeat must be at least 1"):
        MODULE.profile_manifests([tmp_path / "missing.jsonl"], tmp_path / "profile", repeat=0)

    assert not (tmp_path / "profile").exists()


def test_profiler_validates_manifests_before_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "profile"

    with pytest.raises(FileNotFoundError):
        MODULE.profile_manifests([tmp_path / "missing.jsonl"], output, repeat=1)

    assert not output.exists()


def test_resource_summary_preserves_requested_manifest_order() -> None:
    def row(name: str, digest: str, width: int) -> dict[str, object]:
        return {
            "manifest": {"path": name, "sha256": digest},
            "profiler_source_sha256": "a" * 64,
            "case_count": 1,
            "unique_artifacts": 1,
            "max_width": width,
            "max_height": width,
            "max_megapixels": width * width / 1_000_000,
            "process": {"wall_ms": 1.0, "peak_rss_mib": 2.0},
            "status_counts": {"detected": 1},
        }

    summaries = MODULE.summarize_resources(
        [
            row("small", "f" * 64, 256),
            row("large", "0" * 64, 1024),
        ]
    )

    assert [summary["manifest"] for summary in summaries] == ["small", "large"]
