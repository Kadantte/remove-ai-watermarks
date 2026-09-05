"""Basic command-line contracts for standalone maintainer scripts."""

from __future__ import annotations

import subprocess
import sys
from os import environ
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script",
    [
        "visible_groundtruth.py",
        "visible_recall_sample.py",
        "visible_sheets.py",
        "registered_mark_calibrate.py",
        "watermark_benchmark.py",
    ],
)
def test_script_help_exits_cleanly(script: str) -> None:
    result = subprocess.run(  # noqa: S603 -- fixed interpreter and repository-owned script path
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_visible_groundtruth_help_is_cp1252_safe() -> None:
    env = environ.copy()
    env["PYTHONIOENCODING"] = "cp1252"
    result = subprocess.run(  # noqa: S603 -- fixed interpreter and repository-owned script path
        [sys.executable, str(ROOT / "scripts" / "visible_groundtruth.py"), "--help"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
