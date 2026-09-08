# /// script
# requires-python = ">=3.11"
# dependencies = ["modal>=1.5"]
# ///
"""Seed-1/2 probes at measured first-clean rungs (research, gitignored out/).

Mosque and botanical are the new Meta 0.10 worst cases. The OpenAI 9-face
grid is the harder OpenAI first-clean (0.075). Lighthouse @ 0.10 seeds 1-2
already ran clean in preship validation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parent.parent
app = modal.App("chroma-seed-probes")
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
_PROMPT = "high quality, sharp, detailed, faithful to the original"
_NEGATIVE = "blurry, lowres, distorted text, garbled text, artifacts"


@app.function(image=image, gpu="H100", volumes={"/cache": cache}, timeout=1800)
def run(payload: dict[str, bytes]) -> dict:
    import io
    import time

    import torch
    from diffusers import ChromaImg2ImgPipeline
    from PIL import Image

    chroma = ChromaImg2ImgPipeline.from_pretrained(CHROMA_MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
    jobs = (
        ("mosque", 0.10),
        ("botanical", 0.10),
        ("openai_faces", 0.075),
    )
    out: dict = {}
    for name, strength in jobs:
        source = Image.open(io.BytesIO(payload[name])).convert("RGB")
        width = max(16, (source.width // 16) * 16)
        height = max(16, (source.height // 16) * 16)
        for seed in (1, 2):
            started = time.perf_counter()
            image = chroma(
                prompt=_PROMPT,
                negative_prompt=_NEGATIVE,
                image=source,
                width=width,
                height=height,
                strength=strength,
                num_inference_steps=max(1, math.ceil(4 / strength)),
                guidance_scale=5.0,
                generator=torch.Generator(device="cpu").manual_seed(seed),
            ).images[0]
            if image.size != source.size:
                image = image.resize(source.size, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            key = f"{name}_s{strength:.4f}_seed{seed}"
            out[key] = buffer.getvalue()
            print(f"{key} {time.perf_counter() - started:.1f}s", flush=True)
    cache.commit()
    return out


@app.local_entrypoint()
def main(out: str = "out/preship-chroma/seed-probes") -> None:
    sources = {
        "mosque": _ROOT / "out/cohort-calibration/meta-api/architecture_mosque/original.png",
        "botanical": _ROOT / "out/cohort-calibration/meta-api/illustration_watercolor_botanical/original.png",
        "openai_faces": _ROOT / "data/synthid/originals/ChatGPT Image May 30, 2026, 10_31_08 AM.png",
    }
    missing = [k for k, p in sources.items() if not p.exists()]
    if missing:
        raise SystemExit(f"missing sources: {missing}")
    results = run.remote({k: p.read_bytes() for k, p in sources.items()})
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    for key, png in results.items():
        (dest / f"{key}.png").write_bytes(png)
    print(json.dumps(sorted(results), indent=2))
