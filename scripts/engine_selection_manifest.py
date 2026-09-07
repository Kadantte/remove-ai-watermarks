"""Shared validation for the publication-cleared engine-selection image matrix."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from watermark_benchmark import sha256_file

REUSE_BASIS = "project-maintainer-public-test-clearance"
CONTENT_FIELDS = (
    "pair_id",
    "provider",
    "model",
    "file",
    "width",
    "height",
    "content_stratum",
    "sha256",
    "source_payload_sha256",
    "reuse_basis",
    "prompt",
)
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9-]*")


@dataclass(frozen=True)
class ContentFixture:
    """One validated source row from the publication-cleared content matrix."""

    pair_id: str
    provider: str
    model: str
    path: Path
    width: int
    height: int
    content_stratum: str
    sha256: str
    source_payload_sha256: str
    reuse_basis: str
    prompt: str

    @property
    def name(self) -> str:
        """Return a stable filename-safe carrier identity."""
        return f"{self.pair_id}-{self.provider}"

    @property
    def seed(self) -> int:
        """Derive a stable 60-bit seed from the source bytes."""
        return int(self.sha256[:15], 16)


def _require_text(row: dict[str, str | None], field: str, *, location: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: {field} must be non-empty")
    return value.strip()


def _require_sha256(row: dict[str, str | None], field: str, *, location: str) -> str:
    value = _require_text(row, field, location=location)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{location}: {field} must be a lowercase sha256")
    return value


def load_content_manifest(source_manifest: Path) -> list[ContentFixture]:
    """Load and validate one balanced, content-stratified source manifest."""
    manifest = source_manifest.resolve()
    with manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CONTENT_FIELDS:
            raise ValueError(f"{manifest}: expected columns {', '.join(CONTENT_FIELDS)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{manifest}: source manifest contains no rows")

    fixtures: list[ContentFixture] = []
    paths: set[Path] = set()
    hashes: set[str] = set()
    pair_provider: set[tuple[str, str]] = set()
    stratum_provider: set[tuple[str, str]] = set()
    pairs: dict[str, list[ContentFixture]] = defaultdict(list)

    for line_number, row in enumerate(rows, start=2):
        location = f"{manifest}:{line_number}"
        if None in row:
            raise ValueError(f"{location}: row contains extra columns")
        pair_id = _require_text(row, "pair_id", location=location)
        provider = _require_text(row, "provider", location=location)
        if not _SAFE_ID.fullmatch(pair_id) or not _SAFE_ID.fullmatch(provider):
            raise ValueError(f"{location}: pair_id and provider must be lowercase filename-safe identifiers")
        source_sha256 = _require_sha256(row, "sha256", location=location)
        source_payload_sha256 = _require_sha256(row, "source_payload_sha256", location=location)
        reuse_basis = _require_text(row, "reuse_basis", location=location)
        if reuse_basis != REUSE_BASIS:
            raise ValueError(f"{location}: unsupported reuse_basis {reuse_basis!r}")
        try:
            width = int(_require_text(row, "width", location=location))
            height = int(_require_text(row, "height", location=location))
        except ValueError as exc:
            raise ValueError(f"{location}: width and height must be positive integers") from exc
        if width <= 0 or height <= 0:
            raise ValueError(f"{location}: width and height must be positive integers")

        relative_path = Path(_require_text(row, "file", location=location))
        path = (manifest.parent / relative_path).resolve()
        if not path.is_relative_to(manifest.parent):
            raise ValueError(f"{location}: file must stay inside {manifest.parent}")
        if path in paths:
            raise ValueError(f"{location}: duplicate source path {relative_path}")
        paths.add(path)
        if source_sha256 in hashes:
            raise ValueError(f"{location}: duplicate source sha256 {source_sha256}")
        hashes.add(source_sha256)
        identity = (pair_id, provider)
        if identity in pair_provider:
            raise ValueError(f"{location}: duplicate pair/provider {pair_id}/{provider}")
        pair_provider.add(identity)

        content_stratum = _require_text(row, "content_stratum", location=location)
        stratum_identity = (content_stratum, provider)
        if stratum_identity in stratum_provider:
            raise ValueError(f"{location}: duplicate stratum/provider {content_stratum}/{provider}")
        stratum_provider.add(stratum_identity)

        if not path.is_file():
            raise ValueError(f"{location}: source file is missing: {relative_path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != source_sha256:
            raise ValueError(
                f"{location}: source sha256 mismatch for {relative_path}: expected {source_sha256}, got {actual_sha256}"
            )
        with Image.open(path) as image:
            actual_size = image.size
            actual_format = image.format
        if actual_size != (width, height):
            raise ValueError(f"{location}: expected size {(width, height)}, got {actual_size}")
        if actual_format != "PNG":
            raise ValueError(f"{location}: expected PNG source, got {actual_format}")

        fixture = ContentFixture(
            pair_id=pair_id,
            provider=provider,
            model=_require_text(row, "model", location=location),
            path=path,
            width=width,
            height=height,
            content_stratum=content_stratum,
            sha256=source_sha256,
            source_payload_sha256=source_payload_sha256,
            reuse_basis=reuse_basis,
            prompt=_require_text(row, "prompt", location=location),
        )
        fixtures.append(fixture)
        pairs[pair_id].append(fixture)

    providers = {fixture.provider for fixture in fixtures}
    if len(providers) < 2:
        raise ValueError(f"{manifest}: a stratified cohort requires at least two providers")
    for pair_id, pair in sorted(pairs.items()):
        if {fixture.provider for fixture in pair} != providers:
            raise ValueError(f"{manifest}: pair {pair_id} does not contain every provider")
        if len({fixture.prompt for fixture in pair}) != 1 or len({fixture.content_stratum for fixture in pair}) != 1:
            raise ValueError(f"{manifest}: pair {pair_id} does not preserve one prompt and content stratum")
    return fixtures
