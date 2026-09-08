#!/usr/bin/env python3
"""Reproduce the deterministic Content Seal crop, resize, and JPEG variants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "contentseal"
MANIFEST = CORPUS / "manifest.csv"


def _rows() -> dict[str, dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        return {row["name"]: row for row in csv.DictReader(stream)}


def _write_and_verify(image: Image.Image, path: Path, row: dict[str, str], *, format: str, quality: int) -> None:
    image.save(path, format=format, quality=quality)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != row["sha256"]:
        raise RuntimeError(f"{row['name']} hash mismatch: expected {row['sha256']}, got {digest}")
    log.info("Verified %s", path)


def reproduce_transforms(output_dir: Path) -> list[Path]:
    """Write and hash-check the eight deterministic manifest variants."""
    rows = _rows()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for prefix, source_name in (("fox", "gen_fox_forest"), ("text", "gen_text_poster")):
        with Image.open(CORPUS / rows[source_name]["file"]) as opened:
            source = opened.convert("RGB")

        for fraction in (0.5, 0.33):
            width = int(source.width * fraction)
            height = int(source.height * fraction)
            left = (source.width - width) // 2
            top = (source.height - height) // 2
            name = f"{prefix}_crop{int(fraction * 100)}"
            path = output_dir / f"{name}.webp"
            crop = source.crop((left, top, left + width, top + height))
            _write_and_verify(crop, path, rows[name], format="WEBP", quality=95)
            outputs.append(path)

        scale = 512 / max(source.size)
        resized = source.resize(
            (round(source.width * scale), round(source.height * scale)),
            Image.Resampling.LANCZOS,
        )
        name = f"{prefix}_res512"
        path = output_dir / f"{name}.webp"
        _write_and_verify(resized, path, rows[name], format="WEBP", quality=95)
        outputs.append(path)

        name = f"{prefix}_jpeg85"
        path = output_dir / f"{name}.jpg"
        _write_and_verify(source, path, rows[name], format="JPEG", quality=85)
        outputs.append(path)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Directory for regenerated variants")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    reproduce_transforms(args.output_dir)


if __name__ == "__main__":
    main()
