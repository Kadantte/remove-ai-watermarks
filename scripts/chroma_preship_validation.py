# /// script
# requires-python = ">=3.11"
# dependencies = ["modal>=1.5"]
# ///
"""Pre-ship validation outputs for the chroma-zimage profile.

1. Seed sensitivity of the measured boundaries: chroma at the worst-fixture
   first-clean rungs (google 633uuy @ 0.25, meta lighthouse @ 0.10) with
   seeds 1 and 2, to check the boundary is not a seed-0 fluke.
2. Face-stage interaction: the FULL chroma-zimage pipeline (Chroma1 global at
   the shipped Google floor 0.40, then the inherited YuNet/SAM/Z-Image face
   stage) on the 5-face fixture 3mc4t9, to verify the composited face regions
   do not re-introduce a detectable signal on top of the floor.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parent.parent

app = modal.App("chroma-preship-validation")

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
_PROMPT = "high quality, sharp, detailed, faithful to the original"
_NEGATIVE = "blurry, lowres, distorted text, garbled text, artifacts"
GOOGLE_FLOOR = 0.40
MAX_SIDE = 1536


@app.function(image=image, gpu="H100", volumes={"/cache": cache}, timeout=2400)
def run(payload: dict[str, bytes]) -> dict:
    import io
    import time

    import torch
    from diffusers import ChromaImg2ImgPipeline
    from PIL import Image

    sys.path.insert(0, "/pkg")

    def chroma_call(pipe: object, source: Image.Image, strength: float, seed: int) -> Image.Image:
        width = max(16, (source.width // 16) * 16)
        height = max(16, (source.height // 16) * 16)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        out = pipe(
            prompt=_PROMPT,
            negative_prompt=_NEGATIVE,
            image=source,
            width=width,
            height=height,
            strength=strength,
            num_inference_steps=max(1, math.ceil(4 / strength)),
            guidance_scale=5.0,
            generator=generator,
        ).images[0]
        return out if out.size == source.size else out.resize(source.size, Image.Resampling.LANCZOS)

    def png_bytes(image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    out: dict = {}

    chroma = ChromaImg2ImgPipeline.from_pretrained(CHROMA_MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")

    # Seed sensitivity at the worst-fixture boundaries.
    for key, strength in (("google_633uuy", 0.25), ("meta_lighthouse", 0.10)):
        source = Image.open(io.BytesIO(payload[key])).convert("RGB")
        if MAX_SIDE and max(source.size) > MAX_SIDE:
            scale = MAX_SIDE / max(source.size)
            source = source.resize(
                (round(source.width * scale), round(source.height * scale)),
                Image.Resampling.LANCZOS,
            )
        for seed in (1, 2):
            started = time.perf_counter()
            image = chroma_call(chroma, source, strength, seed)
            out[f"seedprobe_{key}_s{strength}_seed{seed}"] = png_bytes(image)
            print(f"{key} s={strength} seed={seed} ({time.perf_counter() - started:.1f}s)", flush=True)

    del chroma
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    # Full chroma-zimage pipeline on the 5-face fixture at the shipped floor.
    from remove_ai_watermarks._internal.chroma_zimage_pipeline import ChromaZImagePipeline

    full = ChromaZImagePipeline(
        device="cuda",
        torch_dtype=torch.bfloat16,
        keep_global_models_on_device=True,
        keep_face_models_on_device=True,
    )
    source = Image.open(io.BytesIO(payload["google_3mc4t9"])).convert("RGB")
    if MAX_SIDE and max(source.size) > MAX_SIDE:
        scale = MAX_SIDE / max(source.size)
        source = source.resize(
            (round(source.width * scale), round(source.height * scale)),
            Image.Resampling.LANCZOS,
        )
    started = time.perf_counter()
    result = full.run(source, strength=GOOGLE_FLOOR, seed=0)
    out["fullpath_3mc4t9_s0.40"] = png_bytes(result)
    print(f"full pipeline 3mc4t9 @ {GOOGLE_FLOOR} ({time.perf_counter() - started:.1f}s)", flush=True)

    cache.commit()
    return out


@app.local_entrypoint()
def main(out: str = "out/preship-chroma") -> None:
    sources = {
        "google_633uuy": _ROOT / "data/synthid/originals/Gemini_Generated_Image_633uuy633uuy633u.png",
        "meta_lighthouse": _ROOT / "data/contentseal/originals/gen_lighthouse_watercolor.webp",
        "google_3mc4t9": _ROOT / "data/synthid/originals/Gemini_Generated_Image_3mc4t93mc4t93mc4.png",
    }
    results = run.remote({k: p.read_bytes() for k, p in sources.items()})
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    for key, png in results.items():
        (root / f"{key}.png").write_bytes(png)
    print(json.dumps(sorted(results), indent=2))
