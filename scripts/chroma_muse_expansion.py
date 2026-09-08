# /// script
# requires-python = ">=3.11"
# dependencies = ["modal>=1.5"]
# ///
"""Chroma1 ladder for a diverse Meta Muse subset (content-adaptive research).

The first expansion harvested seven files from a spaces dump of 495 Meta
candidates. That set was not diverse: one control lost the seal, one was a
byte-identical copy of the committed lighthouse fixture, and four of the
remaining five were near-duplicate photoreal portraits of one subject.
This script instead ladders six API-generated Muse Image files from the
2026-08-26 61-image corpus, chosen to span flat_ratio and content class
(architecture, product, text, portrait, illustration, busy scene).

Outputs stay under out/ (gitignored). Oracle verdicts are recorded in
docs/chroma1-engine-research.md.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

app = modal.App("chroma-muse-expansion")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "diffusers>=0.38,<1",
        "transformers>=4.53",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "pillow",
        "numpy",
    )
    .env({"HF_HOME": "/cache"})
)

cache = modal.Volume.from_name("chroma-hf-cache", create_if_missing=True)

CHROMA_MODEL_ID = "lodestones/Chroma1-HD"
RUNGS = (0.015, 0.03, 0.045, 0.06, 0.08, 0.10, 0.12)
EFFECTIVE_STEPS = 4
_PROMPT = "high quality, sharp, detailed, faithful to the original"
_NEGATIVE = "blurry, lowres, distorted text, garbled text, artifacts"

# Six files from the 2026-08-26 Muse Image API corpus. None of them hashes
# to a committed contentseal original. Names are corpus stems.
FIXTURES = (
    "architecture_mosque",
    "product_sneaker",
    "text_poster_bakery",
    "portrait_weathered_fisherman",
    "illustration_watercolor_botanical",
    "scene_tokyo_alley",
)

_CORPUS_IMAGES = (
    Path.home() / "Documents/GitHub/remove-ai-watermarks/data/research/corpora/" / "meta-muse-corpus-2026-08-26/images"
)


def _fixture_source(name: str) -> Path:
    matches = sorted(_CORPUS_IMAGES.glob(f"{name}-*.webp"))
    if not matches:
        raise FileNotFoundError(f"No corpus image for {name} under {_CORPUS_IMAGES}")
    return matches[0]


def stage(staged_root: Path) -> None:
    """Write lossless PNG originals and metadata-stripped pixel-identical controls."""
    from PIL import Image

    import remove_ai_watermarks.metadata as metadata

    if Path(metadata.__file__).resolve().parts[: len(_SRC.resolve().parts)] != _SRC.resolve().parts:
        raise RuntimeError(f"metadata loaded from {metadata.__file__}, not {_SRC}")

    staged_root.mkdir(parents=True, exist_ok=True)
    for name in FIXTURES:
        source = _fixture_source(name)
        dest_dir = staged_root / name
        dest_dir.mkdir(parents=True, exist_ok=True)
        original = dest_dir / "original.png"
        with Image.open(source) as opened:
            opened.convert("RGB").save(original, format="PNG")
        control = dest_dir / "control.png"
        control.write_bytes(original.read_bytes())
        metadata.remove_ai_metadata(control, control, keep_standard=True)
        print(f"staged {name} from {source.name}", flush=True)


@app.function(image=image, gpu="H100", volumes={"/cache": cache}, timeout=2400)
def run(payload: dict[str, bytes]) -> dict:
    import io
    import time

    import torch
    from diffusers import ChromaImg2ImgPipeline
    from PIL import Image

    chroma = ChromaImg2ImgPipeline.from_pretrained(CHROMA_MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
    out: dict = {}
    for name, data in payload.items():
        source = Image.open(io.BytesIO(data)).convert("RGB")
        width = max(16, (source.width // 16) * 16)
        height = max(16, (source.height // 16) * 16)
        for strength in RUNGS:
            started = time.perf_counter()
            generator = torch.Generator(device="cpu").manual_seed(0)
            image = chroma(
                prompt=_PROMPT,
                negative_prompt=_NEGATIVE,
                image=source,
                width=width,
                height=height,
                strength=strength,
                num_inference_steps=max(1, math.ceil(EFFECTIVE_STEPS / strength)),
                guidance_scale=5.0,
                generator=generator,
            ).images[0]
            if image.size != source.size:
                image = image.resize(source.size, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            out[f"{name}/chroma_s{strength:.4f}"] = {
                "png": buffer.getvalue(),
                "seconds": round(time.perf_counter() - started, 2),
            }
            print(f"{name} s={strength} done", flush=True)
    cache.commit()
    return out


@app.local_entrypoint()
def main(staged: str = "out/cohort-calibration/meta-api") -> None:
    staged_root = Path(staged)
    if not all((staged_root / name / "original.png").exists() for name in FIXTURES):
        stage(staged_root)
    payload = {name: (staged_root / name / "original.png").read_bytes() for name in FIXTURES}
    results = run.remote(payload)
    timing = {}
    for key, item in results.items():
        destination = staged_root / f"{key}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(item["png"])
        timing[key] = item["seconds"]
    (staged_root / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(json.dumps(timing, indent=2))
