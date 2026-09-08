# /// script
# requires-python = ">=3.11"
# dependencies = ["modal>=1.5"]
# ///
"""Isolated Modal GPU prototype: does a low-strength Chroma1 img2img pass scrub
the invisible watermark while keeping text/faces faithful?

Chroma1 (lodestones, Apache-2.0) is the FLUX.1 architecture re-trained on open
data, and diffusers ships it a native strength img2img pipeline, which is exactly
what FLUX.2 lacks (see docs/synthid.md and issue #88). This script is the same
shape as qwen_scrub_prototype.py: standalone, not imported by the package and
not in uv.lock, so the locked environment never moves for an experiment.

Run from the repository root (weights ~34 GB download once into a Modal volume,
then every later run starts warm):

    uvx modal run scripts/chroma_scrub_prototype.py \
        --source data/synthid/originals/'ChatGPT Image May 31, 2026, 02_03_55 PM.png' \
        --out out/chroma --strengths 0.05,0.1,0.15,0.2

A different GPU (default H100, ~$3.95/h; the 34 GB bf16 stack also fits
A100-80GB at ~$2.50/h):

    CHROMA_GPU=A100-80GB uvx modal run scripts/chroma_scrub_prototype.py ...

What to look for (same protocol as the Qwen study, docs/synthid.md 2.2):

  * SCRUB: the oracle no longer reports the watermark at some strength
    (openai.com/verify for OpenAI carriers, the Gemini app for Google ones).
  * FIDELITY: text and faces stay faithful at that same strength; score with
    scripts/fidelity_metrics.py after the oracle pass.
The smallest strength that clears the oracle while keeping fidelity is the
number to compare against the qwen-zimage floors (OpenAI 0.07675 / Google 0.25).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import modal

_GPU = os.environ.get("CHROMA_GPU", "H100")

app = modal.App("chroma-scrub-proto")

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

# Model weights persist here across runs, so only the first container downloads.
cache = modal.Volume.from_name("chroma-hf-cache", create_if_missing=True)

CHROMA_MODEL_ID = "lodestones/Chroma1-HD"

# A neutral, faithful-regeneration prompt (scrub, not restyle); the same intent
# as the Qwen prototype and the shipped global prompts.
_PROMPT = "high quality, sharp, detailed, faithful to the original"
_NEGATIVE = "blurry, lowres, distorted text, garbled text, artifacts"

_pipe = None


def _load_pipe() -> Any:
    global _pipe
    if _pipe is not None:
        return _pipe
    import torch
    from diffusers import ChromaImg2ImgPipeline

    print(f"loading {CHROMA_MODEL_ID} (bf16)...", flush=True)
    _pipe = ChromaImg2ImgPipeline.from_pretrained(CHROMA_MODEL_ID, torch_dtype=torch.bfloat16)
    _pipe = _pipe.to("cuda")
    return _pipe


def _psnr(a: Any, b: Any) -> float:
    import numpy as np

    x = np.asarray(a.convert("RGB"), dtype=np.float64)
    y = np.asarray(b.convert("RGB"), dtype=np.float64)
    mse = float(((x - y) ** 2).mean())
    return 99.0 if mse <= 1e-12 else 10.0 * float(np.log10(255.0**2 / mse))


@app.function(
    image=image,
    gpu=_GPU,
    volumes={"/cache": cache},
    timeout=2400,
)
def scrub(
    source_bytes: bytes,
    strengths: list[float],
    steps: int,
    guidance: float,
    seed: int,
    max_side: int,
) -> list[tuple[str, bytes, float]]:
    """Sweep one fixture over the strengths; return name, PNG bytes, and PSNR."""
    import io

    import torch
    from PIL import Image

    pipe = _load_pipe()
    source = Image.open(io.BytesIO(source_bytes)).convert("RGB")
    if max_side and max(source.size) > max_side:
        scale = max_side / max(source.size)
        source = source.resize(
            (round(source.width * scale), round(source.height * scale)),
            Image.Resampling.LANCZOS,
        )
    # The Flux-family img2img pipelines default to 1024x1024 when height/width
    # are not passed, silently discarding the input aspect. Floor to the /16
    # latent-patch grid, the same rule the shipped profiles use.
    width = max(16, (source.width // 16) * 16)
    height = max(16, (source.height // 16) * 16)

    results: list[tuple[str, bytes, float]] = []
    for strength in strengths:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        print(f"strength={strength:.2f} on {width}x{height}...", flush=True)
        output = pipe(
            prompt=_PROMPT,
            negative_prompt=_NEGATIVE,
            image=source,
            width=width,
            height=height,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        ).images[0]
        if output.size != source.size:
            output = output.resize(source.size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        output.save(buffer, format="PNG")
        results.append((f"s{strength:.2f}", buffer.getvalue(), _psnr(source, output)))
        print(f"  psnr vs source: {results[-1][2]:.2f} dB", flush=True)
    cache.commit()
    return results


@app.local_entrypoint()
def main(
    source: str,
    strengths: str = "0.05,0.1,0.15,0.2",
    out: str = "out/chroma",
    steps: int = 35,
    guidance: float = 5.0,
    seed: int = 0,
    max_side: int = 1536,
) -> None:
    """Sweep Chroma1 img2img strength over SOURCE and save one output per strength."""
    source_path = Path(source)
    if not source_path.exists():
        raise SystemExit(f"source not found: {source_path}")
    values = [float(s) for s in strengths.split(",") if s.strip()]
    stem = source_path.stem

    print(f"gpu={_GPU} model={CHROMA_MODEL_ID} steps={steps} guidance={guidance} seed={seed}")
    outputs = scrub.remote(source_path.read_bytes(), values, steps, guidance, seed, max_side)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload, psnr in outputs:
        destination = out_dir / f"{stem}_chroma_{name}.png"
        destination.write_bytes(payload)
        print(f"saved {destination} (psnr {psnr:.2f} dB vs source)")

    print(
        "\nDone. Submit each output to the matching oracle (openai.com/verify for "
        "OpenAI carriers, the Gemini app for Google ones). The smallest strength "
        "that clears the oracle while keeping fidelity is the Chroma1 floor."
    )
