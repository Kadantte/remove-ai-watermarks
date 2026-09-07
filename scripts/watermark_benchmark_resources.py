#!/usr/bin/env python3
"""Measure cold-process cost for watermark benchmark manifests.

Each repetition runs ``watermark_benchmark.py`` in a fresh Python process and
records parent-observed wall time plus the child's absolute peak RSS. Detector
results remain ordinary benchmark JSONL files and can be aggregated with
``watermark_benchmark_report.py``. No provider API or provenance oracle is used.

    uv run python scripts/watermark_benchmark_resources.py \
      .local-eval/cohort-256/manifest.jsonl \
      .local-eval/cohort-512/manifest.jsonl \
      --repeat 3 --output-dir .local-eval/resource-profile
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from watermark_benchmark import (  # noqa: E402
    DETECTOR_STATUSES,
    load_manifest,
    parse_strict_json,
    run_benchmark,
    sha256_file,
    write_jsonl,
)
from watermark_benchmark_report import markdown_table, nearest_rank, write_report  # noqa: E402

log = logging.getLogger(__name__)

RESOURCE_SCHEMA_VERSION = 1


def _peak_rss_mib() -> float | None:
    """Return this process's absolute peak RSS in MiB on known Unix ABIs."""
    try:
        import resource
    except ImportError:
        return None

    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return peak / (1024 * 1024)
    if sys.platform.startswith("linux"):
        return peak / 1024
    return None


def _worker(manifest: Path, result: Path, metrics: Path) -> int:
    counts = run_benchmark(manifest, result)
    write_jsonl(
        metrics,
        [
            {
                "peak_rss_mib": _peak_rss_mib(),
                "status_counts": {status: counts[status] for status in DETECTOR_STATUSES},
                "worker_pid": os.getpid(),
            }
        ],
    )
    return 0


def _read_worker_metrics(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{path}: worker metrics must contain exactly one row")
    raw = parse_strict_json(lines[0])
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: worker metrics fields are invalid")
    metrics = cast("dict[str, object]", raw)
    if set(metrics) != {"peak_rss_mib", "status_counts", "worker_pid"}:
        raise ValueError(f"{path}: worker metrics fields are invalid")
    peak = metrics["peak_rss_mib"]
    if peak is not None and (
        not isinstance(peak, (int, float)) or isinstance(peak, bool) or not math.isfinite(peak) or peak < 0
    ):
        raise ValueError(f"{path}: peak_rss_mib must be a finite non-negative number or null")
    raw_counts = metrics["status_counts"]
    if not isinstance(raw_counts, dict):
        raise ValueError(f"{path}: status_counts fields are invalid")
    counts = cast("dict[str, object]", raw_counts)
    if set(counts) != set(DETECTOR_STATUSES):
        raise ValueError(f"{path}: status_counts fields are invalid")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts.values()):
        raise ValueError(f"{path}: status_counts values must be non-negative integers")
    worker_pid = metrics["worker_pid"]
    if not isinstance(worker_pid, int) or isinstance(worker_pid, bool) or worker_pid < 1:
        raise ValueError(f"{path}: worker_pid must be a positive integer")
    return {
        "peak_rss_mib": peak,
        "status_counts": {status: cast("int", counts[status]) for status in DETECTOR_STATUSES},
        "worker_pid": worker_pid,
    }


def _manifest_geometry(manifest: Path) -> dict[str, int | float]:
    from PIL import Image

    cases = load_manifest(manifest)
    artifacts = {case.sha256: case.path for case in cases}
    dimensions: list[tuple[int, int]] = []
    for path in artifacts.values():
        with Image.open(path) as image:
            dimensions.append(image.size)
    max_width, max_height = max(dimensions, key=lambda size: size[0] * size[1])
    return {
        "case_count": len(cases),
        "unique_artifacts": len(artifacts),
        "max_width": max_width,
        "max_height": max_height,
        "max_megapixels": max_width * max_height / 1_000_000,
    }


def _run_child(manifest: Path, result: Path, metrics: Path) -> tuple[float, dict[str, Any]]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(manifest),
        "--_worker",
        "--_result",
        str(result),
        "--_metrics",
        str(metrics),
    ]
    started_ns = time.perf_counter_ns()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1e6
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark worker failed for {manifest} with exit {completed.returncode}; "
            f"stdout={completed.stdout!r}; stderr={completed.stderr!r}"
        )
    return elapsed_ms, _read_worker_metrics(metrics)


def profile_manifests(manifests: Sequence[Path], output_dir: Path, *, repeat: int) -> list[dict[str, Any]]:
    """Profile each manifest in fresh child processes and write their results."""
    if not manifests:
        raise ValueError("at least one manifest is required")
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    resolved = [manifest.resolve() for manifest in manifests]
    if len(set(resolved)) != len(resolved):
        raise ValueError("duplicate manifest")
    manifest_profiles = [(manifest, _manifest_geometry(manifest), sha256_file(manifest)) for manifest in resolved]
    output_dir.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    profiler_source_sha256 = sha256_file(Path(__file__))
    for manifest_index, (manifest, geometry, manifest_digest) in enumerate(manifest_profiles, 1):
        for run_index in range(1, repeat + 1):
            result = output_dir / f"manifest-{manifest_index:02d}-run-{run_index:02d}.jsonl"
            with tempfile.TemporaryDirectory(prefix="watermark-resource-") as temporary:
                metrics_path = Path(temporary) / "metrics.jsonl"
                wall_ms, worker_metrics = _run_child(manifest, result, metrics_path)
            counts = cast("dict[str, int]", worker_metrics["status_counts"])
            if sum(counts.values()) != geometry["case_count"]:
                raise ValueError(f"worker status count does not match manifest case count: {manifest}")
            rows.append(
                {
                    "schema_version": RESOURCE_SCHEMA_VERSION,
                    "profiler_source_sha256": profiler_source_sha256,
                    "manifest": {"path": str(manifest), "sha256": manifest_digest},
                    "run_index": run_index,
                    **geometry,
                    "process": {
                        "wall_ms": wall_ms,
                        "peak_rss_mib": worker_metrics["peak_rss_mib"],
                        "platform": sys.platform,
                        "machine": platform.machine(),
                        "python": sys.version.split()[0],
                        "worker_pid": worker_metrics["worker_pid"],
                    },
                    "status_counts": counts,
                    "result": {"path": str(result.resolve()), "sha256": sha256_file(result)},
                }
            )
            log.info(
                "Profiled %s run %s/%s: %.2f ms, peak RSS %s MiB",
                manifest,
                run_index,
                repeat,
                wall_ms,
                worker_metrics["peak_rss_mib"],
            )

    write_jsonl(output_dir / "resources.jsonl", rows)
    write_report(output_dir / "resources.md", render_resource_report(summarize_resources(rows)))
    return rows


def _metric_stats(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "p50": nearest_rank(ordered, 0.50),
        "p90": nearest_rank(ordered, 0.90),
        "max": ordered[-1],
    }


def summarize_resources(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate process metrics by exact manifest digest."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        manifest = cast("Mapping[str, Any]", row["manifest"])
        grouped[str(manifest["sha256"])].append(row)

    summaries: list[dict[str, Any]] = []
    for digest, group in grouped.items():
        first = group[0]
        process_rows = [cast("Mapping[str, Any]", row["process"]) for row in group]
        wall = _metric_stats([float(row["wall_ms"]) for row in process_rows])
        rss_values = [float(row["peak_rss_mib"]) for row in process_rows if row["peak_rss_mib"] is not None]
        rss = _metric_stats(rss_values) if rss_values else None
        status_signatures = {json.dumps(row["status_counts"], sort_keys=True) for row in group}
        manifest = cast("Mapping[str, Any]", first["manifest"])
        summaries.append(
            {
                "manifest": str(manifest["path"]),
                "cohort": Path(str(manifest["path"])).parent.name,
                "sha256": digest,
                "profiler_source_sha256": first["profiler_source_sha256"],
                "runs": len(group),
                "cases": first["case_count"],
                "unique_artifacts": first["unique_artifacts"],
                "max_geometry": f"{first['max_width']}x{first['max_height']}",
                "max_megapixels": first["max_megapixels"],
                "wall_p50_ms": wall["p50"],
                "wall_p90_ms": wall["p90"],
                "wall_max_ms": wall["max"],
                "rss_p50_mib": rss["p50"] if rss is not None else None,
                "rss_p90_mib": rss["p90"] if rss is not None else None,
                "rss_max_mib": rss["max"] if rss is not None else None,
                "status_counts_stable": len(status_signatures) == 1,
            }
        )
    return summaries


def render_resource_report(summaries: Sequence[Mapping[str, Any]]) -> str:
    """Render an absolute process-cost report."""
    columns = (
        ("cohort", "Cohort"),
        ("max_geometry", "Max geometry"),
        ("max_megapixels", "Max MP"),
        ("cases", "Cases"),
        ("runs", "Runs"),
        ("wall_p50_ms", "Wall p50 ms"),
        ("wall_p90_ms", "Wall p90 ms"),
        ("wall_max_ms", "Wall max ms"),
        ("rss_p50_mib", "Peak RSS p50 MiB"),
        ("rss_p90_mib", "Peak RSS p90 MiB"),
        ("rss_max_mib", "Peak RSS max MiB"),
        ("status_counts_stable", "Statuses stable"),
    )
    lines = [
        "# Watermark benchmark resource profile",
        "",
        (
            "Each run is a fresh Python process. Wall time includes interpreter startup, imports, image decoding, "
            "detector calls, fidelity metrics, and result writing. Peak RSS is the process's absolute high-water "
            "mark, not incremental memory attributable only to validation."
        ),
        "",
        *markdown_table(columns, summaries, missing="unavailable"),
    ]
    lines.extend(
        [
            "",
            "Detector verdicts remain in the sibling `manifest-XX-run-YY.jsonl` files. Aggregate them with "
            "`watermark_benchmark_report.py`; this table answers only the process-cost question.",
            "",
            "Profiler source SHA-256: " + ", ".join(sorted({str(row["profiler_source_sha256"]) for row in summaries})),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path, help="one or more strict benchmark manifests")
    parser.add_argument("--output-dir", type=Path, help="new directory for result JSONL and resource reports")
    parser.add_argument("--repeat", type=int, default=3, help="fresh processes per manifest (default: 3)")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_metrics", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args._worker:
            if len(args.manifests) != 1 or args._result is None or args._metrics is None:
                raise ValueError("worker mode requires one manifest, --_result, and --_metrics")
            return _worker(args.manifests[0], args._result, args._metrics)
        if args.output_dir is None:
            raise ValueError("--output-dir is required")
        if args._result is not None or args._metrics is not None:
            raise ValueError("internal worker outputs require --_worker")
        profile_manifests(args.manifests, args.output_dir, repeat=args.repeat)
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
