#!/usr/bin/env python3
"""Build a local, synthetic image-watermark benchmark cohort.

The builder creates deterministic clean carriers, matched DWT-DCT and TrustMark
embeds, fixed attack transforms, TrustMark remover outputs, and hard negatives.
It writes every generated artifact once under a new output directory and emits
the strict JSONL manifest consumed by ``watermark_benchmark.py``.

Generated artifacts and reports belong under ``.local-eval/`` and are never
committed. No provider oracle or API is used.

    uv run python scripts/watermark_benchmark_cohort.py \
      --output-dir .local-eval/watermark-benchmark-cohort-v1
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import sys
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from watermark_benchmark import SCHEMA_VERSION, load_manifest, sha256_file, write_jsonl  # noqa: E402

log = logging.getLogger(__name__)

RECIPE_VERSION = "watermark-benchmark-cohort-v1"
AdapterName = Literal["dwt-dct", "trustmark"]
AttackName = Literal["jpeg-q90", "resize-75", "crop-5"]

DEFAULT_CARRIERS = (
    # The content spread is deliberate: the legacy DWT-DCT decoder is strongly
    # carrier-dependent even before an attack.
    # Keep seeds explicit so adding a carrier never changes an existing one.
    ("texture", 7),
    ("gradient", 11),
    ("low-detail", 19),
)
DEFAULT_DWT_SCHEMES = ("sdxl", "flux", "sd1")
DEFAULT_ATTACKS: tuple[AttackName, ...] = ("jpeg-q90", "resize-75", "crop-5")
_ADAPTERS: tuple[AdapterName, ...] = ("dwt-dct", "trustmark")
_DWT_SCHEMES = frozenset(DEFAULT_DWT_SCHEMES)
_ATTACKS = frozenset(DEFAULT_ATTACKS)


@dataclass(frozen=True)
class CarrierSpec:
    """Identity of one deterministic synthetic carrier."""

    name: str
    seed: int


def trustmark_available() -> bool:
    """Return whether the optional TrustMark package can be imported."""
    from remove_ai_watermarks.trustmark_detector import is_available

    return is_available()


@cache
def _artifact_sha256(path: Path) -> str:
    """Cache hashes for immutable artifacts created by this builder."""
    return sha256_file(path)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def _dependencies() -> dict[str, str]:
    return {
        "numpy": _package_version("numpy"),
        "pillow": _package_version("pillow"),
        "invisible-watermark": _package_version("invisible-watermark"),
        "trustmark": _package_version("trustmark"),
    }


def _source_revision() -> str:
    return f"{RECIPE_VERSION}@sha256:{_artifact_sha256(Path(__file__))}"


def _carrier(spec: CarrierSpec, *, size: int = 512) -> Image.Image:
    import numpy as np
    from PIL import ImageDraw

    if spec.name == "texture":
        y, x = np.mgrid[:size, :size]
        rng = np.random.default_rng(spec.seed)
        pixels = np.stack(
            (
                x * 255 // (size - 1),
                y * 255 // (size - 1),
                (x + y) * 127 // (size - 1),
            ),
            axis=2,
        ).astype(np.int16)
        pixels += cast("Any", rng).integers(-20, 21, pixels.shape, dtype=np.int16)
        return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")

    if spec.name == "gradient":
        y, x = np.mgrid[:size, :size]
        pixels = np.stack(
            (
                32 + x * 190 // (size - 1),
                48 + y * 160 // (size - 1),
                64 + (x + y) * 140 // (2 * size - 2),
            ),
            axis=2,
        ).astype(np.uint8)
        image = Image.fromarray(pixels, "RGB")
        draw = ImageDraw.Draw(image)
        draw.ellipse((96, 112, 304, 320), fill=(205, 96, 72), outline=(245, 230, 210), width=9)
        draw.rectangle((330, 72, 455, 430), fill=(58, 121, 184), outline=(230, 240, 250), width=7)
        return image

    if spec.name == "low-detail":
        rng = np.random.default_rng(spec.seed)
        bands = cast("Any", rng).integers(0, 9, (16, 16, 3), dtype=np.uint8)
        pixels = np.repeat(np.repeat(bands, size // 16, axis=0), size // 16, axis=1)
        return Image.fromarray(np.clip(pixels.astype(np.int16) + 220, 0, 255).astype(np.uint8), "RGB")

    if spec.name == "hard-checker":
        y, x = np.mgrid[:size, :size]
        checker = ((x // 4 + y // 4 + spec.seed) % 2) * 255
        pixels = np.stack((checker, np.roll(checker, 2, axis=0), np.roll(checker, 2, axis=1)), axis=2)
        return Image.fromarray(pixels.astype(np.uint8), "RGB")

    raise ValueError(f"unknown carrier {spec.name!r}")


def _save(image: Image.Image, path: Path) -> Path:
    image.convert("RGB").save(path, format="PNG")
    return path


def _attack(image: Image.Image, attack: AttackName) -> Image.Image:
    image = image.convert("RGB")
    if attack == "jpeg-q90":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, subsampling=0)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()
    if attack == "resize-75":
        reduced = image.resize((384, 384), Image.Resampling.LANCZOS)
        return reduced.resize(image.size, Image.Resampling.LANCZOS)
    if attack == "crop-5":
        margin_x = round(image.width * 0.05)
        margin_y = round(image.height * 0.05)
        cropped = image.crop((margin_x, margin_y, image.width - margin_x, image.height - margin_y))
        return cropped.resize(image.size, Image.Resampling.LANCZOS)
    raise ValueError(f"unknown attack {attack!r}")


def _embed_dwt_dct(image: Image.Image, scheme: str) -> Image.Image:
    import cv2
    import numpy as np

    watermark_patterns = import_module("remove_ai_watermarks.invisible_watermark")
    patterns = vars(watermark_patterns)
    bits_48 = cast("dict[str, int]", patterns["_BITS_48"])
    sd1_string = cast("bytes", patterns["_SD1_STRING"])
    WatermarkEncoder = import_module("imwatermark").WatermarkEncoder
    encoder = WatermarkEncoder()
    if scheme == "sdxl":
        encoder.set_watermark("bits", [int(bit) for bit in format(bits_48["Stable Diffusion XL"], "048b")])
    elif scheme == "flux":
        encoder.set_watermark(
            "bits",
            [int(bit) for bit in format(bits_48["FLUX.2 (Black Forest Labs)"], "048b")],
        )
    elif scheme == "sd1":
        encoder.set_watermark("bytes", sd1_string)
    else:
        raise ValueError(f"unknown DWT-DCT scheme {scheme!r}")

    rgb = np.asarray(image.convert("RGB"))
    marked_bgr = np.asarray(encoder.encode(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), "dwtDct"), dtype=np.uint8)
    return Image.fromarray(cv2.cvtColor(marked_bgr, cv2.COLOR_BGR2RGB), "RGB")


def _trustmark_runtime() -> Any:
    if not trustmark_available():
        raise RuntimeError("TrustMark cohort requested but the trustmark extra is unavailable")

    TrustMark = import_module("trustmark").TrustMark
    return TrustMark(
        verbose=False,
        model_type="P",
        encoding_type=TrustMark.Encoding.BCH_5,
        loadRemover=True,
    )


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _row(
    *,
    root: Path,
    case_id: str,
    pair_id: str,
    adapter: AdapterName,
    arm: str,
    state: str,
    path: Path,
    reference: Path | None,
    source_revision: str,
    transform_name: str,
    transform_revision: str,
    parameters: dict[str, Any],
    seed: int,
    expected: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "pair_id": pair_id,
        "media_type": "image",
        "adapter": adapter,
        "arm": arm,
        "state": state,
        "path": _relative(path, root),
        "sha256": _artifact_sha256(path),
        "reference_path": _relative(reference, root) if reference is not None else None,
        "reference_sha256": _artifact_sha256(reference) if reference is not None else None,
        "source_revision": source_revision,
        "transform": {
            "name": transform_name,
            "revision": transform_revision,
            "parameters": parameters,
        },
        "seed": seed,
        "expected": expected,
    }


def _attack_parameters(attack: AttackName) -> dict[str, Any]:
    if attack == "jpeg-q90":
        values: dict[str, Any] = {"quality": 90, "subsampling": 0}
    elif attack == "resize-75":
        values = {"scale": 0.75, "restore_size": True, "resampling": "lanczos"}
    else:
        values = {"crop_fraction_per_edge": 0.05, "restore_size": True, "resampling": "lanczos"}
    return values


def _append_hard_negative(
    rows: list[dict[str, Any]],
    *,
    root: Path,
    artifacts: Path,
    adapter: AdapterName,
    source_revision: str,
    dependencies: dict[str, str],
) -> None:
    spec = CarrierSpec("hard-checker", 101)
    path = artifacts / "hard-checker.png"
    if not path.exists():
        _save(_carrier(spec), path)
    rows.append(
        _row(
            root=root,
            case_id=f"{adapter}--hard-checker--clean",
            pair_id=f"{adapter}--hard-checker",
            adapter=adapter,
            arm="hard_negative",
            state="clean",
            path=path,
            reference=None,
            source_revision=source_revision,
            transform_name="synthetic-carrier",
            transform_revision=RECIPE_VERSION,
            parameters={"carrier": spec.name, "dependencies": dependencies},
            seed=spec.seed,
            expected="not_detected",
        )
    )


def _append_dwt_rows(
    rows: list[dict[str, Any]],
    *,
    root: Path,
    artifacts: Path,
    carriers: Sequence[tuple[CarrierSpec, Image.Image]],
    schemes: Sequence[str],
    attacks: Sequence[AttackName],
    source_revision: str,
    dependencies: dict[str, str],
) -> None:
    for spec, clean in carriers:
        clean_path = _save(clean, artifacts / f"{spec.name}--clean.png")
        for scheme in schemes:
            pair_id = f"dwt-dct--{scheme}--{spec.name}"
            base_parameters = {"scheme": scheme, "carrier": spec.name, "dependencies": dependencies}
            rows.append(
                _row(
                    root=root,
                    case_id=f"{pair_id}--clean",
                    pair_id=pair_id,
                    adapter="dwt-dct",
                    arm="matched_negative",
                    state="clean",
                    path=clean_path,
                    reference=clean_path,
                    source_revision=source_revision,
                    transform_name="synthetic-carrier",
                    transform_revision=RECIPE_VERSION,
                    parameters=base_parameters,
                    seed=spec.seed,
                    expected="not_detected",
                )
            )
            marked = _embed_dwt_dct(clean, scheme)
            marked_path = _save(marked, artifacts / f"{spec.name}--dwt-{scheme}--marked.png")
            rows.append(
                _row(
                    root=root,
                    case_id=f"{pair_id}--marked",
                    pair_id=pair_id,
                    adapter="dwt-dct",
                    arm="positive",
                    state="marked",
                    path=marked_path,
                    reference=clean_path,
                    source_revision=source_revision,
                    transform_name="dwt-dct-embed",
                    transform_revision=f"invisible-watermark@{dependencies['invisible-watermark']}",
                    parameters=base_parameters,
                    seed=spec.seed,
                    expected="detected",
                )
            )
            for attack in attacks:
                attacked_path = _save(
                    _attack(marked, attack),
                    artifacts / f"{spec.name}--dwt-{scheme}--{attack}.png",
                )
                rows.append(
                    _row(
                        root=root,
                        case_id=f"{pair_id}--{attack}",
                        pair_id=pair_id,
                        adapter="dwt-dct",
                        arm="positive",
                        state="attacked",
                        path=attacked_path,
                        reference=clean_path,
                        source_revision=source_revision,
                        transform_name=f"dwt-dct-embed-then-{attack}",
                        transform_revision=RECIPE_VERSION,
                        parameters=base_parameters | _attack_parameters(attack),
                        seed=spec.seed,
                        expected="unresolved",
                    )
                )


def _append_trustmark_rows(
    rows: list[dict[str, Any]],
    *,
    root: Path,
    artifacts: Path,
    carriers: Sequence[tuple[CarrierSpec, Image.Image]],
    attacks: Sequence[AttackName],
    source_revision: str,
    dependencies: dict[str, str],
) -> None:
    runtime = _trustmark_runtime()
    for spec, clean in carriers:
        clean_path = artifacts / f"{spec.name}--clean.png"
        if not clean_path.exists():
            _save(clean, clean_path)
        pair_id = f"trustmark--p-schema1--{spec.name}"
        base_parameters = {
            "variant": "P",
            "schema": 1,
            "carrier": spec.name,
            "dependencies": dependencies,
        }
        rows.append(
            _row(
                root=root,
                case_id=f"{pair_id}--clean",
                pair_id=pair_id,
                adapter="trustmark",
                arm="matched_negative",
                state="clean",
                path=clean_path,
                reference=clean_path,
                source_revision=source_revision,
                transform_name="synthetic-carrier",
                transform_revision=RECIPE_VERSION,
                parameters=base_parameters,
                seed=spec.seed,
                expected="not_detected",
            )
        )
        payload = format(spec.seed, "061b")
        marked = runtime.encode(clean, payload, MODE="binary")
        marked_path = _save(marked, artifacts / f"{spec.name}--trustmark-p--marked.png")
        rows.append(
            _row(
                root=root,
                case_id=f"{pair_id}--marked",
                pair_id=pair_id,
                adapter="trustmark",
                arm="positive",
                state="marked",
                path=marked_path,
                reference=clean_path,
                source_revision=source_revision,
                transform_name="trustmark-embed",
                transform_revision=f"trustmark@{dependencies['trustmark']}",
                parameters=base_parameters | {"payload_sha256": hashlib.sha256(payload.encode()).hexdigest()},
                seed=spec.seed,
                expected="detected",
            )
        )
        for attack in attacks:
            attacked_path = _save(
                _attack(marked, attack),
                artifacts / f"{spec.name}--trustmark-p--{attack}.png",
            )
            rows.append(
                _row(
                    root=root,
                    case_id=f"{pair_id}--{attack}",
                    pair_id=pair_id,
                    adapter="trustmark",
                    arm="positive",
                    state="attacked",
                    path=attacked_path,
                    reference=clean_path,
                    source_revision=source_revision,
                    transform_name=f"trustmark-embed-then-{attack}",
                    transform_revision=RECIPE_VERSION,
                    parameters=base_parameters | _attack_parameters(attack),
                    seed=spec.seed,
                    expected="unresolved",
                )
            )
        removed_path = _save(
            runtime.remove_watermark(marked),
            artifacts / f"{spec.name}--trustmark-p--removed.png",
        )
        rows.append(
            _row(
                root=root,
                case_id=f"{pair_id}--removed",
                pair_id=pair_id,
                adapter="trustmark",
                arm="positive",
                state="removed",
                path=removed_path,
                reference=clean_path,
                source_revision=source_revision,
                transform_name="trustmark-remove",
                transform_revision=f"trustmark@{dependencies['trustmark']}",
                parameters=base_parameters | {"input": "matched-marked"},
                seed=spec.seed,
                expected="unresolved",
            )
        )


def build_cohort(
    output_dir: Path,
    *,
    adapters: Sequence[AdapterName] = _ADAPTERS,
    carriers: Sequence[CarrierSpec] = tuple(CarrierSpec(name, seed) for name, seed in DEFAULT_CARRIERS),
    dwt_schemes: Sequence[str] = DEFAULT_DWT_SCHEMES,
    attacks: Sequence[AttackName] = DEFAULT_ATTACKS,
) -> Path:
    """Build and validate a new local cohort directory, returning its manifest."""
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    unknown_adapters = sorted(set(adapters) - set(_ADAPTERS))
    if not adapters or unknown_adapters:
        detail = ", ".join(unknown_adapters) if unknown_adapters else "none selected"
        raise ValueError(f"invalid adapters: {detail}")
    unknown_schemes = sorted(set(dwt_schemes) - _DWT_SCHEMES)
    if unknown_schemes:
        raise ValueError(f"unknown DWT-DCT schemes: {', '.join(unknown_schemes)}")
    unknown_attacks = sorted(set(attacks) - _ATTACKS)
    if unknown_attacks:
        raise ValueError(f"unknown attacks: {', '.join(unknown_attacks)}")

    output_dir.mkdir(parents=True)
    artifacts = output_dir / "artifacts"
    artifacts.mkdir()
    rows: list[dict[str, Any]] = []
    source_revision = _source_revision()
    dependencies = _dependencies()
    carrier_images = tuple((spec, _carrier(spec)) for spec in carriers)

    if "dwt-dct" in adapters:
        _append_dwt_rows(
            rows,
            root=output_dir,
            artifacts=artifacts,
            carriers=carrier_images,
            schemes=dwt_schemes,
            attacks=attacks,
            source_revision=source_revision,
            dependencies=dependencies,
        )
        _append_hard_negative(
            rows,
            root=output_dir,
            artifacts=artifacts,
            adapter="dwt-dct",
            source_revision=source_revision,
            dependencies=dependencies,
        )
    if "trustmark" in adapters:
        _append_trustmark_rows(
            rows,
            root=output_dir,
            artifacts=artifacts,
            carriers=carrier_images,
            attacks=attacks,
            source_revision=source_revision,
            dependencies=dependencies,
        )
        _append_hard_negative(
            rows,
            root=output_dir,
            artifacts=artifacts,
            adapter="trustmark",
            source_revision=source_revision,
            dependencies=dependencies,
        )

    manifest = output_dir / "manifest.jsonl"
    write_jsonl(manifest, rows)
    load_manifest(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="new local cohort directory")
    parser.add_argument(
        "--adapter",
        action="append",
        choices=_ADAPTERS,
        dest="adapters",
        help="adapter to include; repeat to select both (default: both)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        manifest = build_cohort(args.output_dir, adapters=tuple(args.adapters or _ADAPTERS))
    except (FileExistsError, ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    log.info("Wrote validated benchmark cohort to %s", manifest)


if __name__ == "__main__":
    main()
