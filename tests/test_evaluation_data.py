"""Cross-corpus integrity checks for tracked evaluation tables."""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATIONS = ROOT / "data" / "evaluations"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def test_every_recorded_evaluation_sha256_is_well_formed() -> None:
    checked = 0
    for path in sorted(EVALUATIONS.rglob("*.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            for line_number, row in enumerate(csv.DictReader(stream), start=2):
                for field, value in row.items():
                    if field is not None and field.endswith("sha256") and value:
                        assert SHA256.fullmatch(value), f"{path.relative_to(ROOT)}:{line_number} {field}={value!r}"
                        checked += 1
    assert checked > 0
