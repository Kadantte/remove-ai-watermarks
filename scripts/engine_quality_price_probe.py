# /// script
# requires-python = ">=3.11"
# dependencies = ["modal>=1.5"]
# ///
"""Head-to-head probe: qwen-zimage (the shipped production path) vs Chroma1
img2img on one fixture -- warm per-image wall time, VRAM peak, and outputs for
the local fidelity metrics, so quality and price can be compared on the same
pixels.

qwen-zimage runs through the package's own ``QwenZImagePipeline`` (DiffSynth,
Lightning LoRA, Canny ControlNet, face stage) at its certified OpenAI operating
point (0.07675) with both stacks resident, exactly as the deployed worker does.
Chroma1 runs the same neutral prompt with the SDXL-style step compensation
(``requested_steps`` semantics: always spend 4 effective denoising steps), so a
low strength is not silently 1 step of 35.

Standalone like the other prototypes: not imported by the package, not in
uv.lock. The package source is mounted at runtime, never installed.

Run from the repository root (first run downloads both stacks, ~100 GB total,
into the shared volume; later runs start warm):

    uvx modal run scripts/engine_quality_price_probe.py \
        --source data/synthid/originals/'ChatGPT Image May 31, 2026, 02_03_55 PM.png'

Oracle the Chroma ladder outputs afterwards (openai.com/verify) and score both
engines' floor outputs with scripts/fidelity_metrics.py locally.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import modal

_GPU = "H100"
# Verified 2026-08-29 from modal.com/pricing.
_H100_PER_HOUR = 3.95
_A100_80_PER_HOUR = 2.50

app = modal.App("engine-quality-price-probe")

_ROOT = Path(__file__).resolve().parent.parent

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

# Carries both engines' weights; named for the first prototype that filled it.
cache = modal.Volume.from_name("chroma-hf-cache", create_if_missing=True)

QWEN_OPENAI_FLOOR = 0.07675  # watermark_profiles.QWEN_ZIMAGE_OPENAI_STRENGTH
CHROMA_MODEL_ID = "lodestones/Chroma1-HD"
CHROMA_STRENGTHS = (0.05, 0.06, 0.075, 0.09, 0.10)
CHROMA_EFFECTIVE_STEPS = 4

_PROMPT = "high quality, sharp, detailed, faithful to the original"
_NEGATIVE = "blurry, lowres, distorted text, garbled text, artifacts"


def requested_steps(effective_steps: int, strength: float) -> int:
    """Diffusers truncates the step COUNT; spend effective_steps regardless."""
    import math

    return max(1, math.ceil(effective_steps / max(float(strength), 1e-6)))


def _psnr(a: Any, b: Any) -> float:
    import numpy as np

    x = np.asarray(a.convert("RGB"), dtype=np.float64)
    y = np.asarray(b.convert("RGB"), dtype=np.float64)
    mse = float(((x - y) ** 2).mean())
    return 99.0 if mse <= 1e-12 else 10.0 * float(np.log10(255.0**2 / mse))


def _png(image: Any) -> bytes:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@app.function(image=image, gpu=_GPU, volumes={"/cache": cache}, timeout=3000)
def probe(source_bytes: bytes, max_side: int) -> dict:
    """Run both engines; return outputs, timings, and VRAM peaks."""
    import gc
    import io
    import time

    import torch
    from PIL import Image

    sys.path.insert(0, "/pkg")

    source = Image.open(io.BytesIO(source_bytes)).convert("RGB")
    if max_side and max(source.size) > max_side:
        scale = max_side / max(source.size)
        source = source.resize(
            (round(source.width * scale), round(source.height * scale)),
            Image.Resampling.LANCZOS,
        )

    report: dict = {"source_size": source.size, "engines": {}}

    # --- qwen-zimage, the exact production path ---------------------------------
    print("[qwen] loading the production two-stage pipeline (resident stacks)...", flush=True)
    from remove_ai_watermarks._internal.qwen_zimage_pipeline import QwenZImagePipeline

    qwen = QwenZImagePipeline(
        device="cuda",
        torch_dtype=torch.bfloat16,
        keep_global_models_on_device=True,
        keep_face_models_on_device=True,
    )
    print(f"[qwen] warmup pass at strength={QWEN_OPENAI_FLOOR}...", flush=True)
    qwen.run(source, strength=QWEN_OPENAI_FLOOR, seed=0)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    qwen_out = qwen.run(source, strength=QWEN_OPENAI_FLOOR, seed=0)
    qwen_seconds = time.perf_counter() - started
    qwen_vram_gib = torch.cuda.max_memory_allocated() / (1024**3)
    report["engines"]["qwen-zimage"] = {
        "strength": QWEN_OPENAI_FLOOR,
        "warm_seconds": round(qwen_seconds, 2),
        "vram_peak_gib": round(qwen_vram_gib, 1),
        "psnr": round(_psnr(source, qwen_out), 2),
        "png": _png(qwen_out),
    }
    qwen_report = report["engines"]["qwen-zimage"]
    print(
        f"[qwen] warm {qwen_seconds:.2f}s, vram peak {qwen_vram_gib:.1f} GiB, psnr {qwen_report['psnr']:.2f} dB",
        flush=True,
    )

    del qwen
    gc.collect()
    torch.cuda.empty_cache()

    # --- Chroma1, compensated-step ladder ---------------------------------------
    from diffusers import ChromaImg2ImgPipeline

    print(f"[chroma] loading {CHROMA_MODEL_ID} (bf16)...", flush=True)
    chroma = ChromaImg2ImgPipeline.from_pretrained(CHROMA_MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")

    width = max(16, (source.width // 16) * 16)
    height = max(16, (source.height // 16) * 16)
    ladder: list[dict] = []
    for strength in CHROMA_STRENGTHS:
        steps = requested_steps(CHROMA_EFFECTIVE_STEPS, strength)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        generator = torch.Generator(device="cpu").manual_seed(0)
        out = chroma(
            prompt=_PROMPT,
            negative_prompt=_NEGATIVE,
            image=source,
            width=width,
            height=height,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=5.0,
            generator=generator,
        ).images[0]
        seconds = time.perf_counter() - started
        vram_gib = torch.cuda.max_memory_allocated() / (1024**3)
        if out.size != source.size:
            out = out.resize(source.size, Image.Resampling.LANCZOS)
        ladder.append(
            {
                "strength": strength,
                "requested_steps": steps,
                "warm_seconds": round(seconds, 2),
                "vram_peak_gib": round(vram_gib, 1),
                "psnr": round(_psnr(source, out), 2),
                "png": _png(out),
            }
        )
        print(
            f"[chroma] s={strength:.3f} steps={steps}/{steps} warm {seconds:.2f}s "
            f"vram {vram_gib:.1f} GiB psnr {ladder[-1]['psnr']:.2f} dB",
            flush=True,
        )
    report["engines"]["chroma1"] = {"ladder": ladder}

    cache.commit()
    return report


@app.local_entrypoint()
def main(
    source: str,
    out: str = "out/engine-comparison",
    max_side: int = 1536,
) -> None:
    source_path = Path(source)
    if not source_path.exists():
        raise SystemExit(f"source not found: {source_path}")

    report = probe.remote(source_path.read_bytes(), max_side)

    out_dir = Path(out) / source_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for engine, payload in report["engines"].items():
        if engine == "chroma1":
            for run in payload["ladder"]:
                (out_dir / f"chroma_s{run['strength']:.3f}.png").write_bytes(run["png"])
        else:
            (out_dir / f"{engine}_s{payload['strength']}.png").write_bytes(payload["png"])

    summary = {
        "source": str(source_path),
        "source_size": report["source_size"],
        "gpu": _GPU,
        "usd_per_hour": {"H100": _H100_PER_HOUR, "A100-80GB": _A100_80_PER_HOUR},
        "engines": {
            engine: (
                {k: v for k, v in payload.items() if k != "png"}
                if engine != "chroma1"
                else {"ladder": [{k: v for k, v in run.items() if k != "png"} for run in payload["ladder"]]}
            )
            for engine, payload in report["engines"].items()
        },
    }
    (out_dir / "timing.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\noutputs in {out_dir}; oracle the chroma ladder, then score locally with:")
    print(f"  uv run scripts/fidelity_metrics.py compare --original '{source}' \\")
    print(f"    --variant qwen={out_dir / 'qwen-zimage_s0.0768.png'} --variant chroma={out_dir}/chroma_sXXX.png")
