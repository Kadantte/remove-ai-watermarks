# /// script
# requires-python = ">=3.11"
# dependencies = ["modal>=1.5"]
# ///
"""Chroma1 ladder for extra OpenAI SynthID carriers (content-adaptive research).

The three committed ChatGPT originals are two zero-face text cards (first-clean
0.06) and one 9-face grid (0.075). Images API gpt-image-1 / gpt-image-1.5 stamp
C2PA but not SynthID, so they are not carriers. These four files are spaces
uploads with OpenAI C2PA whose pixel SynthID still reads DETECTED on
POST /v1/content_provenance_checks. Outputs stay under out/ (gitignored).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parent.parent
app = modal.App("chroma-openai-expansion")
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
RUNGS = (0.04, 0.05, 0.06, 0.075, 0.09, 0.12)
_PROMPT = "high quality, sharp, detailed, faithful to the original"
_NEGATIVE = "blurry, lowres, distorted text, garbled text, artifacts"
FIXTURES = ("dense_ed68", "midflat_038c", "textlike_3b17", "faces2_4dab")


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
            image = chroma(
                prompt=_PROMPT,
                negative_prompt=_NEGATIVE,
                image=source,
                width=width,
                height=height,
                strength=strength,
                num_inference_steps=max(1, math.ceil(4 / strength)),
                guidance_scale=5.0,
                generator=torch.Generator(device="cpu").manual_seed(0),
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
def main(staged: str = "out/cohort-calibration/openai-expand") -> None:
    staged_root = Path(staged)
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
