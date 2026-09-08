# /// script
# requires-python = ">=3.11"
# dependencies = ["modal>=1.5"]
# ///
"""Extra high Chroma1 rungs (0.25-0.40) for the Google cohort fixtures.

The 0.08-0.20 ladder came back detected end-to-end on the first fixture, so
this appends the high rungs for every Google fixture without touching the
already-oracled lower outputs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parent.parent

app = modal.App("chroma-google-high-rungs")

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
RUNGS = (0.25, 0.30, 0.35, 0.40)
EFFECTIVE_STEPS = 4
_PROMPT = "high quality, sharp, detailed, faithful to the original"
_NEGATIVE = "blurry, lowres, distorted text, garbled text, artifacts"

FIXTURES = (
    "Gemini_Generated_Image_633uuy633uuy633u.png",
    "Gemini_Generated_Image_akdbeiakdbeiakdb.png",
    "Gemini_Generated_Image_y48j3cy48j3cy48j.png",
    "Gemini_Generated_Image_3mc4t93mc4t93mc4.png",
)
MAX_SIDE = 1536


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
        if MAX_SIDE and max(source.size) > MAX_SIDE:
            scale = MAX_SIDE / max(source.size)
            source = source.resize(
                (round(source.width * scale), round(source.height * scale)),
                Image.Resampling.LANCZOS,
            )
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
            out[f"{Path(name).stem}/chroma_s{strength:.2f}"] = {
                "png": buffer.getvalue(),
                "seconds": round(time.perf_counter() - started, 2),
            }
            print(f"{name} s={strength} done", flush=True)
    cache.commit()
    return out


@app.local_entrypoint()
def main(out: str = "out/cohort-calibration/google") -> None:
    payload = {name: (_ROOT / "data" / "synthid" / "originals" / name).read_bytes() for name in FIXTURES}
    results = run.remote(payload)
    timing = {}
    for key, item in results.items():
        destination = Path(out) / f"{key}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(item["png"])
        timing[key] = item["seconds"]
    print(json.dumps(timing, indent=2))
