# /// script
# requires-python = ">=3.11"
# dependencies = ["modal>=1.5"]
# ///
"""Chroma1 ladder for the Microsoft InvisMark cohort (three valid Paint sources)
plus the matching qwen-zimage production runs at the shipped 0.15 floor.

The sources are the same three Paint carriers behind the qwen cohort floor
(staged under out/cohort-calibration/microsoft/<label>/original.png with
pixel-identical metadata-stripped controls). They are real user uploads from
the raiw corpus and must never be committed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parent.parent

app = modal.App("chroma-microsoft-ladder")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "torchvision>=0.20.0",
        "diffusers>=0.38,<1",
        "diffsynth>=2.0.17,<3",
        "transformers>=4.53",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "pillow",
        "numpy",
        "opencv-python-headless",
    )
    .env({"HF_HOME": "/cache"})
    .add_local_dir(_ROOT / "src" / "remove_ai_watermarks", remote_path="/pkg/remove_ai_watermarks")
)

cache = modal.Volume.from_name("chroma-hf-cache", create_if_missing=True)

CHROMA_MODEL_ID = "lodestones/Chroma1-HD"
CHROMA_RUNGS = (0.04, 0.06, 0.08, 0.10, 0.125, 0.15, 0.20)
QWEN_FLOOR = 0.15
EFFECTIVE_STEPS = 4
_SOURCES = ("paint-1", "paint-2", "paint-3")
_PROMPT = "high quality, sharp, detailed, faithful to the original"
_NEGATIVE = "blurry, lowres, distorted text, garbled text, artifacts"


@app.function(image=image, gpu="H100", volumes={"/cache": cache}, timeout=2400)
def run(payload: dict[str, bytes]) -> dict:
    import io
    import sys
    import time

    import torch
    from PIL import Image

    sys.path.insert(0, "/pkg")

    def chroma_call(chroma: object, source: Image.Image, strength: float, seed: int) -> Image.Image:
        width = max(16, (source.width // 16) * 16)
        height = max(16, (source.height // 16) * 16)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        out = chroma(
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
        return out if out.size == source.size else out.resize(source.size, Image.Resampling.LANCZOS)

    def png_bytes(image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    from diffusers import ChromaImg2ImgPipeline

    chroma = ChromaImg2ImgPipeline.from_pretrained(CHROMA_MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")

    from remove_ai_watermarks._internal.qwen_zimage_pipeline import QwenZImagePipeline

    qwen = QwenZImagePipeline(
        device="cuda",
        torch_dtype=torch.bfloat16,
        keep_global_models_on_device=True,
        keep_face_models_on_device=True,
    )

    out: dict = {}
    for label in _SOURCES:
        source = Image.open(io.BytesIO(payload[label])).convert("RGB")
        rungs: dict = {}
        for strength in CHROMA_RUNGS:
            started = time.perf_counter()
            image = chroma_call(chroma, source, strength, 0)
            rungs[f"{strength:.4f}"] = {
                "png": png_bytes(image),
                "seconds": round(time.perf_counter() - started, 2),
            }
            print(f"{label} chroma s={strength} done", flush=True)
        started = time.perf_counter()
        qwen_out = qwen.run(source, strength=QWEN_FLOOR, seed=0)
        out[label] = {
            "chroma": rungs,
            "qwen_seconds": round(time.perf_counter() - started, 2),
            "qwen_png": png_bytes(qwen_out),
        }
        print(f"{label} qwen s={QWEN_FLOOR} done", flush=True)

    cache.commit()
    return out


@app.local_entrypoint()
def main(staged: str = "out/cohort-calibration/microsoft") -> None:
    staged_root = Path(staged)
    payload = {label: (staged_root / label / "original.png").read_bytes() for label in _SOURCES}
    results = run.remote(payload)
    summary: dict = {}
    for label, entry in results.items():
        target = staged_root / label
        for strength, item in entry["chroma"].items():
            (target / f"chroma_s{float(strength):.4f}.png").write_bytes(item["png"])
        (target / "qwen_floor.png").write_bytes(entry["qwen_png"])
        summary[label] = {
            "chroma_seconds": {k: v["seconds"] for k, v in entry["chroma"].items()},
            "qwen_seconds": entry["qwen_seconds"],
        }
    print(json.dumps(summary, indent=2))
