#!/usr/bin/env python3
"""Run reproducible, development-only watermark benchmark cases.

The input is a strict JSONL manifest. Every row names one immutable artifact,
its evidence arm and processing state, the detector adapter, source and
transform revisions, an explicit seed, and an optional decoded-pixel reference.
Artifacts and references are hash-checked before any detector runs.

The output keeps three questions separate:

* ``detection`` records only what one named adapter recognized;
* ``removal`` records the post-removal observation without certifying erasure;
* ``fidelity`` measures decoded-pixel distance from an explicit reference.

This is a maintainer tool, not an installed command. It calls no provenance
oracle or provider API, does not download corpora, and refuses to overwrite an
existing report. The optional TrustMark dependency may fetch its official model
weights when its local package cache is incomplete.

    uv run python scripts/watermark_benchmark.py manifest.jsonl --output report.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from numpy.typing import NDArray

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from remove_ai_watermarks._internal.schema import require_schema_version  # noqa: E402

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
Arm = Literal["positive", "matched_negative", "wrong_key", "hard_negative"]
State = Literal["clean", "marked", "attacked", "removed"]
ExpectedDetection = Literal["detected", "not_detected", "unresolved"]
DetectorStatus = Literal["detected", "not_detected", "unavailable", "error"]

_ARMS: tuple[Arm, ...] = ("positive", "matched_negative", "wrong_key", "hard_negative")
_STATES: tuple[State, ...] = ("clean", "marked", "attacked", "removed")
_EXPECTED: tuple[ExpectedDetection, ...] = ("detected", "not_detected", "unresolved")
DETECTOR_STATUSES: tuple[DetectorStatus, ...] = ("detected", "not_detected", "unavailable", "error")
_FIELDS = {
    "schema_version",
    "case_id",
    "pair_id",
    "media_type",
    "adapter",
    "arm",
    "state",
    "path",
    "sha256",
    "reference_path",
    "reference_sha256",
    "source_revision",
    "transform",
    "seed",
    "expected",
}
_TRANSFORM_FIELDS = {"name", "revision", "parameters"}


@dataclass(frozen=True)
class Transform:
    """Exact transform identity carried by one benchmark case."""

    name: str
    revision: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkCase:
    """One validated manifest row with absolute, hash-checked paths."""

    case_id: str
    pair_id: str
    media_type: Literal["image"]
    adapter: str
    arm: Arm
    state: State
    path: Path
    sha256: str
    reference_path: Path | None
    reference_sha256: str | None
    source_revision: str
    transform: Transform
    seed: int | None
    expected: ExpectedDetection


@dataclass(frozen=True)
class DetectorOutcome:
    """Provider-scoped detector observation, never a general clean verdict."""

    status: DetectorStatus
    label: str | None
    error: str | None = None


class DetectorAdapter(Protocol):
    """Small local detector seam used by the benchmark kernel."""

    @property
    def name(self) -> str: ...

    @property
    def source_file(self) -> Path: ...

    @property
    def available(self) -> bool: ...

    def detect(self, path: Path, image: NDArray[Any]) -> DetectorOutcome: ...


@dataclass(frozen=True)
class FunctionAdapter:
    """Adapt one local optional detector without copying its implementation."""

    name: str
    source_file: Path
    available: bool
    detector: Callable[[Path, NDArray[Any]], str | None]

    def detect(self, path: Path, image: NDArray[Any]) -> DetectorOutcome:
        label = self.detector(path, image)
        return DetectorOutcome(status="detected" if label is not None else "not_detected", label=label)


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@cache
def _code_sha256(path: Path) -> str:
    return sha256_file(path)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite number {value}")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate field {key!r}")
        result[key] = value
    return result


def parse_strict_json(value: str) -> object:
    """Parse JSON while rejecting duplicate fields and non-finite numbers."""
    return json.loads(
        value,
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )


def require_nonempty_string(value: object, *, field: str, location: str) -> str:
    """Return a validated non-empty string or raise with its source location."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: {field} must be a non-empty string")
    return value


def require_sha256(value: object, *, field: str, location: str) -> str:
    """Return a validated lowercase SHA-256 digest."""
    digest = require_nonempty_string(value, field=field, location=location)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{location}: {field} must be a lowercase sha256")
    return digest


def _artifact(
    path_value: object,
    digest_value: object,
    *,
    manifest: Path,
    prefix: str,
    location: str,
    digest_cache: dict[Path, str],
) -> tuple[Path, str]:
    raw_path = require_nonempty_string(path_value, field=f"{prefix}path", location=location)
    expected_digest = require_sha256(digest_value, field=f"{prefix}sha256", location=location)
    path = Path(raw_path)
    if not path.is_absolute():
        path = manifest.parent / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{location}: {prefix}path is not a file: {path}")
    actual_digest = digest_cache.get(path)
    if actual_digest is None:
        actual_digest = sha256_file(path)
        digest_cache[path] = actual_digest
    if actual_digest != expected_digest:
        raise ValueError(
            f"{location}: {prefix}sha256 mismatch for {path}: expected {expected_digest}, got {actual_digest}"
        )
    return path, expected_digest


def _transform(value: object, *, location: str) -> Transform:
    if not isinstance(value, dict):
        raise ValueError(f"{location}: transform must contain exactly name, revision, and parameters")
    transform = cast("dict[str, object]", value)
    if set(transform) != _TRANSFORM_FIELDS:
        raise ValueError(f"{location}: transform must contain exactly name, revision, and parameters")
    name = require_nonempty_string(transform["name"], field="transform.name", location=location)
    revision = require_nonempty_string(transform["revision"], field="transform.revision", location=location)
    parameters = transform["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError(f"{location}: transform.parameters must be an object")
    return Transform(name=name, revision=revision, parameters=cast("dict[str, Any]", parameters))


def _case(
    raw: object,
    *,
    manifest: Path,
    line_number: int,
    digest_cache: dict[Path, str],
) -> BenchmarkCase:
    location = f"{manifest}:{line_number}"
    if not isinstance(raw, dict):
        raise ValueError(f"{location}: expected exactly {', '.join(sorted(_FIELDS))}")
    values = cast("dict[str, object]", raw)
    if set(values) != _FIELDS:
        raise ValueError(f"{location}: expected exactly {', '.join(sorted(_FIELDS))}")
    try:
        require_schema_version(
            values["schema_version"],
            contract="watermark benchmark",
            supported=(SCHEMA_VERSION,),
        )
    except ValueError as exc:
        raise ValueError(f"{location}: {exc}") from exc

    media_type = values["media_type"]
    if media_type != "image":
        raise ValueError(f"{location}: media_type must be image in schema v{SCHEMA_VERSION}")
    arm = values["arm"]
    if arm not in _ARMS:
        raise ValueError(f"{location}: arm must be one of {', '.join(_ARMS)}")
    state = values["state"]
    if state not in _STATES:
        raise ValueError(f"{location}: state must be one of {', '.join(_STATES)}")
    expected = values["expected"]
    if expected not in _EXPECTED:
        raise ValueError(f"{location}: expected must be one of {', '.join(_EXPECTED)}")
    seed = values["seed"]
    if seed is not None and type(seed) is not int:
        raise ValueError(f"{location}: seed must be an integer or null")

    path, sha256 = _artifact(
        values["path"],
        values["sha256"],
        manifest=manifest,
        prefix="",
        location=location,
        digest_cache=digest_cache,
    )
    reference_path_value = values["reference_path"]
    reference_sha256_value = values["reference_sha256"]
    if (reference_path_value is None) != (reference_sha256_value is None):
        raise ValueError(f"{location}: reference_path and reference_sha256 must be provided together")
    reference_path: Path | None = None
    reference_sha256: str | None = None
    if reference_path_value is not None:
        reference_path, reference_sha256 = _artifact(
            reference_path_value,
            reference_sha256_value,
            manifest=manifest,
            prefix="reference_",
            location=location,
            digest_cache=digest_cache,
        )

    return BenchmarkCase(
        case_id=require_nonempty_string(values["case_id"], field="case_id", location=location),
        pair_id=require_nonempty_string(values["pair_id"], field="pair_id", location=location),
        media_type="image",
        adapter=require_nonempty_string(values["adapter"], field="adapter", location=location),
        arm=arm,
        state=state,
        path=path,
        sha256=sha256,
        reference_path=reference_path,
        reference_sha256=reference_sha256,
        source_revision=require_nonempty_string(values["source_revision"], field="source_revision", location=location),
        transform=_transform(values["transform"], location=location),
        seed=seed,
        expected=expected,
    )


def load_manifest(path: Path) -> list[BenchmarkCase]:
    """Load a strict JSONL manifest and verify every artifact digest."""
    rows: list[BenchmarkCase] = []
    case_ids: set[str] = set()
    digest_cache: dict[Path, str] = {}
    with path.open(encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, 1):
            if not line.strip():
                continue
            try:
                raw = parse_strict_json(line)
            except (json.JSONDecodeError, ValueError) as exc:
                detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
                raise ValueError(f"{path}:{line_number}: invalid JSON: {detail}") from exc
            row = _case(raw, manifest=path, line_number=line_number, digest_cache=digest_cache)
            if row.case_id in case_ids:
                raise ValueError(f"{path}:{line_number}: duplicate case_id {row.case_id!r}")
            case_ids.add(row.case_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: manifest is empty")
    return rows


def _detect_dwt_dct(path: Path, image: NDArray[Any]) -> str | None:
    from remove_ai_watermarks.image_io import to_bgr
    from remove_ai_watermarks.invisible_watermark import detect_invisible_watermark

    return detect_invisible_watermark(path, image=to_bgr(image))


def _detect_trustmark(path: Path, _image: NDArray[Any]) -> str | None:
    from remove_ai_watermarks.trustmark_detector import detect_trustmark

    return detect_trustmark(path)


def default_adapters() -> dict[str, DetectorAdapter]:
    """Return the first two local, open image-watermark adapters."""
    from remove_ai_watermarks import invisible_watermark, trustmark_detector

    dwt_source = Path(invisible_watermark.__file__).resolve()
    trustmark_source = Path(trustmark_detector.__file__).resolve()
    return {
        "dwt-dct": FunctionAdapter(
            name="dwt-dct",
            source_file=dwt_source,
            available=invisible_watermark.is_available(),
            detector=_detect_dwt_dct,
        ),
        "trustmark": FunctionAdapter(
            name="trustmark",
            source_file=trustmark_source,
            available=trustmark_detector.is_available(),
            detector=_detect_trustmark,
        ),
    }


def repository_state() -> dict[str, str | bool]:
    """Identify the repository commit and whether tracked files differ from it."""
    git_executable = shutil.which("git")
    if git_executable is None:
        raise RuntimeError("cannot resolve repository revision: git is not installed")

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - resolved git executable and fixed repository cwd
            [git_executable, *args],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )

    revision = git("rev-parse", "HEAD")
    if revision.returncode != 0 or not revision.stdout.strip():
        raise RuntimeError(f"cannot resolve repository revision: {revision.stderr.strip()}")
    tracked = git("status", "--porcelain", "--untracked-files=no")
    if tracked.returncode != 0:
        raise RuntimeError(f"cannot inspect repository state: {tracked.stderr.strip()}")
    return {"commit": revision.stdout.strip(), "dirty": bool(tracked.stdout.strip())}


def _decode_image(path: Path) -> NDArray[Any] | None:
    import cv2

    from remove_ai_watermarks.image_io import imread

    return imread(path, cv2.IMREAD_UNCHANGED)


def _image_fidelity(
    case: BenchmarkCase,
    artifact: NDArray[Any] | None,
    reference_decoder: Callable[[Path], NDArray[Any] | None],
) -> dict[str, Any]:
    if case.reference_path is None:
        return {"status": "not_measured", "reason": "reference_not_provided"}

    import numpy as np

    reference = reference_decoder(case.reference_path)
    if artifact is None or reference is None:
        return {"status": "not_measured", "reason": "decoded_pixels_unavailable"}
    if artifact.shape != reference.shape:
        return {
            "status": "incomparable",
            "reason": "shape_mismatch",
            "artifact_shape": list(artifact.shape),
            "reference_shape": list(reference.shape),
        }
    if artifact.dtype != np.uint8 or reference.dtype != np.uint8:
        return {
            "status": "incomparable",
            "reason": "unsupported_dtype",
            "artifact_dtype": str(artifact.dtype),
            "reference_dtype": str(reference.dtype),
        }

    delta = artifact.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(delta)
    mse = float(np.mean(np.square(delta)))
    changed: np.ndarray[Any, np.dtype[np.bool_]] = (
        artifact != reference if artifact.ndim == 2 else np.any(artifact != reference, axis=-1)
    )
    identical = mse == 0.0
    return {
        "status": "measured",
        "identical": identical,
        "mae_8bit": float(np.mean(absolute)),
        "changed_fraction": float(np.mean(changed)),
        "psnr_db": None if identical else 20.0 * math.log10(255.0) - 10.0 * math.log10(mse),
        "psnr_status": "unbounded_identical" if identical else "measured",
    }


def _detect(adapter: DetectorAdapter, path: Path, image: NDArray[Any]) -> DetectorOutcome:
    try:
        return adapter.detect(path, image)
    except Exception as exc:
        log.warning("Benchmark adapter %s failed for %s: %s", adapter.name, path, exc)
        return DetectorOutcome(status="error", label=None, error=f"{type(exc).__name__}: {exc}")


def _detection_record(
    outcome: DetectorOutcome,
    expected: ExpectedDetection,
    adapter_elapsed_ms: float | None,
) -> dict[str, Any]:
    comparable = outcome.status in ("detected", "not_detected") and expected != "unresolved"
    record: dict[str, Any] = {
        "status": outcome.status,
        "label": outcome.label,
        "expected": expected,
        "matches_expected": outcome.status == expected if comparable else None,
        "positive_evidence": outcome.status == "detected",
        "adapter_elapsed_ms": adapter_elapsed_ms,
    }
    if outcome.error is not None:
        record["error"] = outcome.error
    return record


def _removal_record(state: State, outcome: DetectorOutcome) -> dict[str, str | bool]:
    if state != "removed":
        status = "not_applicable"
    elif outcome.status == "detected":
        status = "signal_detected_after_removal"
    elif outcome.status == "not_detected":
        status = "no_recognized_signal_after_removal"
    else:
        status = "unmeasured"
    return {
        "attempted": state == "removed",
        "status": status,
        "certifies_erasure": False,
    }


def evaluate_case(
    case: BenchmarkCase,
    *,
    adapters: Mapping[str, DetectorAdapter],
    repository: Mapping[str, str | bool],
    reference_decoder: Callable[[Path], NDArray[Any] | None] = _decode_image,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Evaluate one case while keeping detector and fidelity claims distinct."""
    try:
        adapter = adapters[case.adapter]
    except KeyError as exc:
        raise ValueError(f"unknown adapter {case.adapter!r}") from exc
    artifact = _decode_image(case.path)
    adapter_elapsed_ms: float | None = None
    if artifact is None:
        outcome = DetectorOutcome(status="error", label=None, error="artifact could not be decoded")
    elif not adapter.available:
        outcome = DetectorOutcome(status="unavailable", label=None)
    else:
        started_ns = clock_ns()
        outcome = _detect(adapter, case.path, artifact)
        adapter_elapsed_ms = (clock_ns() - started_ns) / 1e6
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case.case_id,
        "pair_id": case.pair_id,
        "media_type": case.media_type,
        "adapter": case.adapter,
        "arm": case.arm,
        "state": case.state,
        "artifact": {
            "sha256": case.sha256,
            "source_revision": case.source_revision,
        },
        "reference": {"sha256": case.reference_sha256} if case.reference_sha256 is not None else None,
        "transform": {
            "name": case.transform.name,
            "revision": case.transform.revision,
            "parameters": case.transform.parameters,
            "seed": case.seed,
        },
        "run": {
            "repository": dict(repository),
            "kernel_source_sha256": _code_sha256(Path(__file__)),
            "adapter_source_sha256": _code_sha256(adapter.source_file),
        },
        "detection": _detection_record(outcome, case.expected, adapter_elapsed_ms),
        "removal": _removal_record(case.state, outcome),
        "fidelity": _image_fidelity(case, artifact, reference_decoder),
    }


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Publish strict JSONL atomically and refuse to replace an existing report."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite existing report: {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_benchmark(manifest: Path, output: Path) -> Counter[str]:
    """Evaluate a manifest with the built-in local adapters and write JSONL."""
    cases = load_manifest(manifest)
    adapters = default_adapters()
    unknown = sorted({case.adapter for case in cases} - adapters.keys())
    if unknown:
        raise ValueError(f"unknown adapters: {', '.join(unknown)}")
    repository = repository_state()
    counts: Counter[str] = Counter()
    reference_decoder = lru_cache(maxsize=8)(_decode_image)

    def records() -> Iterable[dict[str, Any]]:
        for case in cases:
            record = evaluate_case(
                case,
                adapters=adapters,
                repository=repository,
                reference_decoder=reference_decoder,
            )
            counts[record["detection"]["status"]] += 1
            yield record

    write_jsonl(output, records())
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="strict benchmark JSONL manifest")
    parser.add_argument("--output", type=Path, required=True, help="new JSONL result path; never overwritten")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        counts = run_benchmark(args.manifest, args.output)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    log.info(
        "Wrote %s cases to %s: %s detected, %s not detected, %s unavailable, %s errors",
        counts.total(),
        args.output,
        counts["detected"],
        counts["not_detected"],
        counts["unavailable"],
        counts["error"],
    )


if __name__ == "__main__":
    main()
