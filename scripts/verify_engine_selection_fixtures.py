#!/usr/bin/env python3
"""Verify the tracked content matrix used by the auto-engine study."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from engine_selection_manifest import load_content_manifest
from watermark_benchmark import sha256_file

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE_ROOT = _ROOT / "data" / "evaluations" / "engine-selection"
_MANIFEST = _FIXTURE_ROOT / "content-manifest.csv"
_CARRIER_MANIFEST = _FIXTURE_ROOT / "carrier-manifest.csv"
_EXPECTED_PROVIDERS = {"meta", "openai"}


def main() -> int:
    errors: list[str] = []
    try:
        rows = load_content_manifest(_MANIFEST)
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
        rows = []
    with _CARRIER_MANIFEST.open(newline="", encoding="utf-8") as stream:
        carrier_rows = list(csv.DictReader(stream))

    providers = {row.provider for row in rows}
    if rows and providers != _EXPECTED_PROVIDERS:
        errors.append(f"content providers: expected {_EXPECTED_PROVIDERS}, got {providers}")

    expected_pair_ids = {f"{index:02d}" for index in range(19)}
    pair_ids = {row.pair_id for row in rows}
    if rows and pair_ids != expected_pair_ids:
        errors.append(f"pair ids: expected {sorted(expected_pair_ids)}, got {sorted(pair_ids)}")

    paths = {row.path.relative_to(_FIXTURE_ROOT) for row in rows}
    committed_files = {
        path.relative_to(_FIXTURE_ROOT) for path in (_FIXTURE_ROOT / "originals").glob("**/*") if path.is_file()
    }
    unlisted = committed_files - paths
    if unlisted:
        errors.append(f"unlisted files: {sorted(str(path) for path in unlisted)}")

    carrier_providers: set[str] = set()
    for row in carrier_rows:
        relative_path = Path(row["file"])
        path = (_FIXTURE_ROOT / relative_path).resolve()
        carrier_providers.add(row["provider"])
        if not path.is_relative_to(_ROOT / "data"):
            errors.append(f"carrier outside data/: {relative_path}")
            continue
        if not path.is_file():
            errors.append(f"missing carrier: {relative_path}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != row["sha256"]:
            errors.append(f"carrier hash mismatch for {relative_path}: expected {row['sha256']}, got {actual_hash}")
    if carrier_providers != {"google", "meta", "openai"}:
        errors.append(f"carrier providers: expected google/meta/openai; got {sorted(carrier_providers)}")

    if errors:
        for error in errors:
            log.error("%s", error)
        return 1

    log.info(
        "Verified %s content files in %s matched pairs and %s canonical carriers",
        len(rows),
        len(pair_ids),
        len(carrier_rows),
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
