"""Measure a registered visible-mark detector on independently labeled arms.

The input is a JSONL manifest with exactly two fields per row:

``path``
    Absolute path, or a path relative to the manifest.
``arm``
    ``positive`` for a visually confirmed carrier, ``metadata`` for a
    provenance-only cohort, ``negative`` for an independently adjudicated
    no-mark image, or ``control`` for an unlabeled comparison image with no
    known local signal.

The arms stay separate because metadata names a provider, not the presence of a
visible mark, and missing local signals do not make an image a true negative. The
script imports the registered engine and reads its shipped gate, so calibration
cannot silently use a copied configuration.

Input images and manifests are read-only. Keep private inputs and generated
manifests outside the repository or in a gitignored evaluation directory.

    uv run python scripts/registered_mark_calibrate.py microsoft manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from remove_ai_watermarks import watermark_registry  # noqa: E402
from remove_ai_watermarks.image_io import imread  # noqa: E402

Arm = Literal["positive", "metadata", "negative", "control"]
_ARMS: tuple[Arm, ...] = ("positive", "metadata", "negative", "control")


class ManifestRow(TypedDict):
    path: str
    arm: Arm


class ArmSummary(TypedDict):
    n: int
    unreadable: int
    min: float | None
    p50: float | None
    p90: float | None
    p99: float | None
    max: float | None
    fires: int


def _percentile(ordered: list[float], fraction: float) -> float | None:
    """Return a nearest-rank percentile from values sorted in ascending order."""
    if not ordered:
        return None
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def summarize(values: list[float], *, unreadable: int, fires: int) -> ArmSummary:
    """Summarize one independently defined arm without inferring its label."""
    ordered = sorted(values)
    return {
        "n": len(values),
        "unreadable": unreadable,
        "min": _percentile(ordered, 0.0),
        "p50": _percentile(ordered, 0.5),
        "p90": _percentile(ordered, 0.9),
        "p99": _percentile(ordered, 0.99),
        "max": _percentile(ordered, 1.0),
        "fires": fires,
    }


def load_manifest(path: Path) -> list[ManifestRow]:
    """Load and validate a manifest, resolving relative paths beside it."""
    rows: list[ManifestRow] = []
    seen: set[Path] = set()
    with path.open(encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict) or set(raw) != {"path", "arm"}:
                raise ValueError(f"{path}:{line_number}: expected exactly path and arm")
            raw_path = raw["path"]
            raw_arm = raw["arm"]
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(f"{path}:{line_number}: path must be a non-empty string")
            if raw_arm not in _ARMS:
                raise ValueError(f"{path}:{line_number}: arm must be one of {', '.join(_ARMS)}")
            image_path = Path(raw_path)
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            image_path = image_path.resolve()
            if image_path in seen:
                raise ValueError(f"{path}:{line_number}: duplicate image path {image_path}")
            seen.add(image_path)
            rows.append({"path": str(image_path), "arm": cast("Arm", raw_arm)})
    if not rows:
        raise ValueError(f"{path}: manifest is empty")
    return rows


def measure(mark: str, rows: list[ManifestRow]) -> tuple[float, dict[Arm, ArmSummary]]:
    """Score every row with the registered engine and its current shipped gate."""
    registered_mark = watermark_registry.get_mark(mark)
    # Calibration intentionally resolves the registry's concrete engine so it
    # cannot drift onto a copied configuration.
    engine = watermark_registry._engine(mark)
    config = getattr(engine, "config", None)
    gate = getattr(config, "detect_ncc_threshold", None)
    if not isinstance(gate, int | float):
        raise ValueError(f"registered mark {mark!r} does not expose a text-detector NCC gate")

    scores: dict[Arm, list[float]] = {arm: [] for arm in _ARMS}
    unreadable: Counter[Arm] = Counter()
    fires: Counter[Arm] = Counter()
    for row in rows:
        image = imread(row["path"])
        if image is None:
            unreadable[row["arm"]] += 1
            continue
        detection: Any = registered_mark.detect(image)
        scores[row["arm"]].append(float(detection.confidence))
        fires[row["arm"]] += bool(detection.detected)

    threshold = float(gate)
    return threshold, {arm: summarize(scores[arm], unreadable=unreadable[arm], fires=fires[arm]) for arm in _ARMS}


def _format_stat(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def print_report(mark: str, gate: float, summaries: dict[Arm, ArmSummary]) -> None:
    """Print a compact human-readable calibration report."""
    print(f"registered mark: {mark}  gate: {gate:.3f}")
    print(
        f"{'arm':10s} {'n':>5s} {'bad':>5s} {'min':>7s} {'p50':>7s} {'p90':>7s} {'p99':>7s} {'max':>7s} {'fires':>7s}"
    )
    for arm in _ARMS:
        row = summaries[arm]
        print(
            f"{arm:10s} {row['n']:5d} {row['unreadable']:5d} "
            f"{_format_stat(row['min']):>7s} {_format_stat(row['p50']):>7s} "
            f"{_format_stat(row['p90']):>7s} {_format_stat(row['p99']):>7s} "
            f"{_format_stat(row['max']):>7s} {row['fires']:7d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mark", choices=watermark_registry.mark_keys())
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    gate, summaries = measure(args.mark, rows)
    if args.json:
        print(json.dumps({"mark": args.mark, "gate": gate, "arms": summaries}, sort_keys=True))
    else:
        print_report(args.mark, gate, summaries)


if __name__ == "__main__":
    main()
