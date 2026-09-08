#!/usr/bin/env python3
"""Synchronize the packaged C2PA soft-binding registry snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_URL = (
    "https://raw.githubusercontent.com/c2pa-org/softbinding-algorithm-list/main/softbinding-algorithm-list.json"
)
PINNED_SOURCE_URL = (
    "https://raw.githubusercontent.com/c2pa-org/softbinding-algorithm-list/{revision}/softbinding-algorithm-list.json"
)
DEFAULT_REVISION_URL = (
    "https://api.github.com/repos/c2pa-org/softbinding-algorithm-list/commits"
    "?path=softbinding-algorithm-list.json&per_page=1"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "src/remove_ai_watermarks/_internal/_generated_c2pa_soft_bindings.py"
SOURCE_LICENSE = "CC BY 4.0"
SOURCE_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
_ALGORITHM_PATTERN = re.compile(r"^[A-Za-z]{2,63}(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)+\.?$")
_DECODED_MEDIA_TYPES = frozenset({"application", "audio", "image", "model", "text", "video"})


@dataclass(frozen=True, slots=True)
class RegistryRow:
    """The runtime subset of one validated upstream registry entry."""

    identifier: int
    algorithm: str
    kind: str
    decoded_media_types: tuple[str, ...]
    encoded_media_types: tuple[str, ...]
    display_label: str
    date_entered: str
    resolution_apis: tuple[str, ...]
    deprecated: bool


def _read_bytes(source: str) -> bytes:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        request = Request(source, headers={"User-Agent": "remove-ai-watermarks-c2pa-registry-sync"})
        with urlopen(request, timeout=30) as response:
            return response.read()
    return Path(source).read_bytes()


def _read_json(source: str) -> Any:
    return json.loads(_read_bytes(source))


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or not value.isprintable():
        raise ValueError(f"{field} must be a non-empty printable string")
    return value


def _media_types(entry: dict[str, Any], identifier: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    decoded_value = entry.get("decodedMediaTypes", [])
    encoded_value = entry.get("encodedMediaTypes", [])
    if not isinstance(decoded_value, list) or not all(isinstance(item, str) for item in decoded_value):
        raise ValueError(f"entry {identifier}: decodedMediaTypes must be a string array")
    if not isinstance(encoded_value, list) or not all(isinstance(item, str) for item in encoded_value):
        raise ValueError(f"entry {identifier}: encodedMediaTypes must be a string array")
    decoded = tuple(cast("list[str]", decoded_value))
    encoded = tuple(cast("list[str]", encoded_value))
    if not decoded and not encoded:
        raise ValueError(f"entry {identifier}: one media-type array is required")
    unknown = set(decoded) - _DECODED_MEDIA_TYPES
    if unknown:
        raise ValueError(f"entry {identifier}: unsupported decoded media types: {sorted(unknown)}")
    if any("/" not in media_type or any(character.isspace() for character in media_type) for media_type in encoded):
        raise ValueError(f"entry {identifier}: invalid encoded media type")
    return decoded, encoded


def _validated_urls(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a string array")
    urls = tuple(cast("list[str]", value))
    if any(not _is_http_url(url) for url in urls):
        raise ValueError(f"{field} contains an invalid URL")
    return urls


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _display_label(description: str) -> str:
    """Keep the official description's compact first sentence as its display label."""
    return description.partition(". ")[0].rstrip(".")


def parse_registry(payload: object) -> tuple[RegistryRow, ...]:
    """Validate the upstream shape and return its normalized runtime rows."""
    if not isinstance(payload, list):
        raise ValueError("registry root must be an array")
    rows: list[RegistryRow] = []
    for raw_entry in cast("list[object]", payload):
        if not isinstance(raw_entry, dict):
            raise ValueError("every registry entry must be an object")
        entry = cast("dict[str, Any]", raw_entry)
        identifier = entry.get("identifier")
        if not isinstance(identifier, int) or isinstance(identifier, bool) or not 0 <= identifier <= 65535:
            raise ValueError("identifier must be an integer from 0 through 65535")
        algorithm = _require_text(entry.get("alg"), f"entry {identifier}: alg")
        if _ALGORITHM_PATTERN.fullmatch(algorithm) is None:
            raise ValueError(f"entry {identifier}: invalid alg {algorithm!r}")
        kind = entry.get("type")
        if kind not in {"watermark", "fingerprint"}:
            raise ValueError(f"entry {identifier}: invalid type {kind!r}")
        decoded, encoded = _media_types(entry, identifier)
        metadata_value = entry.get("entryMetadata")
        if not isinstance(metadata_value, dict):
            raise ValueError(f"entry {identifier}: entryMetadata must be an object")
        metadata = cast("dict[str, Any]", metadata_value)
        description = _require_text(metadata.get("description"), f"entry {identifier}: description")
        date_entered = _require_text(metadata.get("dateEntered"), f"entry {identifier}: dateEntered")
        try:
            datetime.fromisoformat(date_entered.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"entry {identifier}: invalid dateEntered {date_entered!r}") from error
        contact = _require_text(metadata.get("contact"), f"entry {identifier}: contact")
        if "@" not in contact or any(character.isspace() for character in contact):
            raise ValueError(f"entry {identifier}: invalid contact")
        information_url = _require_text(metadata.get("informationalUrl"), f"entry {identifier}: informationalUrl")
        if not _is_http_url(information_url):
            raise ValueError(f"entry {identifier}: invalid informationalUrl")
        resolution_apis = _validated_urls(
            entry.get("softBindingResolutionApis", []),
            f"entry {identifier}: softBindingResolutionApis",
        )
        deprecated = entry.get("deprecated", False)
        if not isinstance(deprecated, bool):
            raise ValueError(f"entry {identifier}: deprecated must be a boolean")
        rows.append(
            RegistryRow(
                identifier=identifier,
                algorithm=algorithm,
                kind=cast("str", kind),
                decoded_media_types=decoded,
                encoded_media_types=encoded,
                display_label=_display_label(description),
                date_entered=date_entered,
                resolution_apis=resolution_apis,
                deprecated=deprecated,
            )
        )

    identifiers = [row.identifier for row in rows]
    algorithms = [row.algorithm for row in rows]
    if identifiers != sorted(identifiers):
        raise ValueError("registry entries must be ordered by identifier")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("registry identifiers must be unique")
    if len(algorithms) != len(set(algorithms)):
        raise ValueError("registry algorithms must be unique")
    return tuple(rows)


def _latest_revision() -> str:
    revisions = _read_json(DEFAULT_REVISION_URL)
    if not isinstance(revisions, list) or not revisions or not isinstance(revisions[0], dict):
        raise ValueError("GitHub did not return a registry revision")
    revision = cast("dict[str, Any]", revisions[0]).get("sha")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("GitHub returned an invalid registry revision")
    return revision


def _source_snapshot(source: str, explicit_revision: str | None) -> tuple[bytes, str]:
    if source == DEFAULT_SOURCE_URL:
        revision = explicit_revision or _latest_revision()
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError("the official source revision must be a 40-character commit SHA")
        # Fetch through the immutable revision. Reading `main` before its revision
        # creates a race where the data and recorded commit name different states.
        return _read_bytes(PINNED_SOURCE_URL.format(revision=revision)), revision
    source_bytes = _read_bytes(source)
    revision = explicit_revision or f"sha256:{hashlib.sha256(source_bytes).hexdigest()}"
    return source_bytes, revision


def _quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _string_tuple(values: tuple[str, ...]) -> str:
    if not values:
        return "()"
    body = ", ".join(_quoted(value) for value in values)
    return f"({body}{',' if len(values) == 1 else ''})"


def _string_field(value: str, *, indent: str = "        ") -> list[str]:
    literal = _quoted(value)
    if len(indent) + len(literal) + 1 <= 120:
        return [f"{indent}{literal},"]

    chunks: list[str] = []
    remaining = value
    while len(_quoted(remaining)) > 92:
        boundary = remaining.rfind(" ", 0, 88)
        if boundary <= 0:
            boundary = 88
        chunks.append(remaining[: boundary + (boundary < len(remaining) and remaining[boundary] == " ")])
        remaining = remaining[len(chunks[-1]) :]
    chunks.append(remaining)
    return [f"{indent}(", *(f"{indent}    {_quoted(chunk)}" for chunk in chunks), f"{indent}),"]


def render_module(rows: tuple[RegistryRow, ...], *, source: str, revision: str) -> str:
    """Render a deterministic, import-only Python snapshot."""
    lines = [
        '"""Generated C2PA soft-binding registry; do not edit by hand.',
        "",
        f"Source: {source}",
        f"Revision: {revision}",
        f"Source license: {SOURCE_LICENSE} ({SOURCE_LICENSE_URL})",
        "Changes: validated, ordered, and reduced to fields used at runtime; descriptions",
        "are shortened to their first sentence for display.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "C2PA_SOFT_BINDING_SOURCE_URL = (",
        f"    {_quoted(source)}",
        ")",
        f"C2PA_SOFT_BINDING_SOURCE_REVISION = {_quoted(revision)}",
        f"C2PA_SOFT_BINDING_SOURCE_LICENSE = {_quoted(SOURCE_LICENSE)}",
        "",
        "C2PA_SOFT_BINDING_ROWS: tuple[",
        "    tuple[int, str, str, tuple[str, ...], tuple[str, ...], str, str, tuple[str, ...], bool], ...",
        "] = (",
    ]
    for row in rows:
        lines.extend(
            [
                "    (",
                f"        {row.identifier},",
                f"        {_quoted(row.algorithm)},",
                f"        {_quoted(row.kind)},",
                f"        {_string_tuple(row.decoded_media_types)},",
                f"        {_string_tuple(row.encoded_media_types)},",
            ]
        )
        lines.extend(_string_field(row.display_label))
        lines.extend(
            [
                f"        {_quoted(row.date_entered)},",
                f"        {_string_tuple(row.resolution_apis)},",
                f"        {row.deprecated!r},",
                "    ),",
            ]
        )
    lines.extend((")", ""))
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE_URL, help="Upstream JSON URL or local file")
    parser.add_argument("--revision", help="Source revision; inferred for the official URL")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Generated Python module")
    parser.add_argument("--check", action="store_true", help="Fail if the generated snapshot is stale")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        source_bytes, revision = _source_snapshot(args.source, args.revision)
        rows = parse_registry(json.loads(source_bytes))
        rendered = render_module(rows, source=args.source, revision=revision)
        if args.check:
            if not args.output.exists() or args.output.read_text() != rendered:
                log.error("C2PA soft-binding snapshot is stale; run this script without --check")
                return 1
            log.info("C2PA soft-binding snapshot is current: %s entries at %s", len(rows), revision)
            return 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        log.info("Updated %s with %s entries at %s", args.output, len(rows), revision)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        log.error("C2PA soft-binding sync failed: %s", error)
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
