# /// script
# requires-python = ">=3.11"
# dependencies = ["modal>=1.5"]
# ///
"""Chroma1 floor calibration across the committed vendor cohorts, plus the
matching qwen-zimage production runs at each cohort's shipped floor, for the
engine quality/price comparison.

Every Chroma1 run spends a fixed 4 effective denoising steps (the requested
count is scaled up, because Diffusers truncates the step COUNT), matching the
semantics the shipped profiles use. Google fixtures are capped to a 1536 px
long side, the same practical path the qwen floors were certified on; Meta
sources run at native size like their cohort measurement.

The Microsoft InvisMark cohort is NOT here: its carrier fixtures are not
committed to this repository (they were private sources), so no Chroma1 floor
can be measured for it until carriers exist. Until then a chroma profile would
treat Microsoft like sdxl-zimage treats Meta today: fall back to the
conservative unknown floor.

Run from the repository root (both stacks are warm in the shared volume):

    uvx modal run scripts/engine_cohort_calibration.py

Outputs land in out/cohort-calibration/<vendor>/<fixture>/ . Oracle each
fixture's rungs in ASCENDING strength order and stop at the first clean one:
that pair brackets the floor with the fewest checks.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import modal

_GPU = "H100"

app = modal.App("engine-cohort-calibration")

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

cache = modal.Volume.from_name("chroma-hf-cache", create_if_missing=True)

CHROMA_MODEL_ID = "lodestones/Chroma1-HD"
QWEN_OPENAI_FLOOR = 0.07675
QWEN_GOOGLE_FLOOR = 0.27
QWEN_META_FLOOR = 0.1
CHROMA_EFFECTIVE_STEPS = 4

_PROMPT = "high quality, sharp, detailed, faithful to the original"
_NEGATIVE = "blurry, lowres, distorted text, garbled text, artifacts"

# vendor -> fixture -> {chroma rungs, qwen floor strength, max side}
FIXTURES: dict[str, dict[str, dict]] = {
    "openai": {
        "ChatGPT Image May 30, 2026, 10_31_08 AM.png": {  # June study 9-face grid, photoreal
            "chroma": (0.04, 0.05, 0.06, 0.075, 0.09, 0.12),
            "qwen": QWEN_OPENAI_FLOOR,
            "max_side": 1536,
        },
        "ChatGPT Image May 31, 2026, 02_02_23 PM.png": {  # full-pipeline quality fixture
            "chroma": (0.04, 0.05, 0.06, 0.075, 0.09),
            "qwen": QWEN_OPENAI_FLOOR,
            "max_side": 1536,
        },
    },
    "google": {
        "Gemini_Generated_Image_633uuy633uuy633u.png": {
            "chroma": (0.08, 0.10, 0.12, 0.15, 0.20),
            "qwen": QWEN_GOOGLE_FLOOR,
            "max_side": 1536,
        },
        "Gemini_Generated_Image_akdbeiakdbeiakdb.png": {
            "chroma": (0.08, 0.10, 0.12, 0.15, 0.20),
            "qwen": QWEN_GOOGLE_FLOOR,
            "max_side": 1536,
        },
        "Gemini_Generated_Image_y48j3cy48j3cy48j.png": {
            "chroma": (0.08, 0.10, 0.12, 0.15, 0.20),
            "qwen": QWEN_GOOGLE_FLOOR,
            "max_side": 1536,
        },
        "Gemini_Generated_Image_3mc4t93mc4t93mc4.png": {  # the 18-face portrait grid
            "chroma": (0.08, 0.10, 0.12, 0.15, 0.20),
            "qwen": QWEN_GOOGLE_FLOOR,
            "max_side": 1536,
        },
    },
    "meta": {
        name: {"chroma": (0.03, 0.045, 0.06, 0.08), "qwen": QWEN_META_FLOOR, "max_side": 0}
        for name in (
            "gen_fox_forest.webp",
            "gen_text_poster.webp",
            "gen_lighthouse_watercolor.webp",
            "gen_night_city.webp",
            "gen_studio_mug.webp",
        )
    },
}

# Seed sensitivity of the already-bracketed OpenAI typography fixture: the
# boundary was (0.05, 0.06] at seed 0; probe the same rungs at two more seeds.
SEED_PROBE = {
    "source": "data/synthid/originals/ChatGPT Image May 31, 2026, 02_03_55 PM.png",
    "rungs": (0.05, 0.06),
    "seeds": (1, 2),
    "max_side": 1536,
}


def requested_steps(effective_steps: int, strength: float) -> int:
    return max(1, math.ceil(effective_steps / max(float(strength), 1e-6)))


@app.function(image=image, gpu=_GPU, volumes={"/cache": cache}, timeout=3000)
def calibrate(payload: dict[str, bytes]) -> dict:
    """Run the chroma ladders, the qwen floors, and the seed probe."""
    import io
    import time

    import torch
    from PIL import Image

    sys.path.insert(0, "/pkg")
    report: dict = {}

    def load(path_key: str, max_side: int) -> Image.Image:
        source = Image.open(io.BytesIO(payload[path_key])).convert("RGB")
        if max_side and max(source.size) > max_side:
            scale = max_side / max(source.size)
            source = source.resize(
                (round(source.width * scale), round(source.height * scale)),
                Image.Resampling.LANCZOS,
            )
        return source

    def chroma_call(chroma: Any, source: Image.Image, strength: float, seed: int) -> Image.Image:
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
            num_inference_steps=requested_steps(CHROMA_EFFECTIVE_STEPS, strength),
            guidance_scale=5.0,
            generator=generator,
        ).images[0]
        return out if out.size == source.size else out.resize(source.size, Image.Resampling.LANCZOS)

    def png_bytes(image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    from diffusers import ChromaImg2ImgPipeline

    print(f"[chroma] loading {CHROMA_MODEL_ID}...", flush=True)
    chroma = ChromaImg2ImgPipeline.from_pretrained(CHROMA_MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")

    from remove_ai_watermarks._internal.qwen_zimage_pipeline import QwenZImagePipeline

    print("[qwen] loading the production two-stage pipeline...", flush=True)
    qwen = QwenZImagePipeline(
        device="cuda",
        torch_dtype=torch.bfloat16,
        keep_global_models_on_device=True,
        keep_face_models_on_device=True,
    )

    for vendor, fixtures in FIXTURES.items():
        report[vendor] = {}
        for name, spec in fixtures.items():
            print(f"[{vendor}] {name}", flush=True)
            source = load(name, spec["max_side"])
            entry: dict = {"chroma": {}, "qwen_seconds": None}
            for strength in spec["chroma"]:
                started = time.perf_counter()
                out = chroma_call(chroma, source, strength, 0)
                seconds = time.perf_counter() - started
                entry["chroma"][f"{strength:.4f}"] = {
                    "png": png_bytes(out),
                    "seconds": round(seconds, 2),
                }
                print(f"  chroma s={strength:.3f} {seconds:.1f}s", flush=True)
            started = time.perf_counter()
            qwen_out = qwen.run(source, strength=spec["qwen"], seed=0)
            entry["qwen_seconds"] = round(time.perf_counter() - started, 2)
            entry["qwen_png"] = png_bytes(qwen_out)
            print(f"  qwen s={spec['qwen']} {entry['qwen_seconds']}s", flush=True)
            report[vendor][name] = entry

    print("[seeds] openai typography boundary probe...", flush=True)
    seed_source = load(SEED_PROBE["source"], SEED_PROBE["max_side"])
    report["seed_probe"] = {}
    for seed in SEED_PROBE["seeds"]:
        for strength in SEED_PROBE["rungs"]:
            out = chroma_call(chroma, seed_source, strength, seed)
            report["seed_probe"][f"s{strength:.2f}_seed{seed}"] = png_bytes(out)
            print(f"  seed={seed} s={strength:.2f}", flush=True)

    cache.commit()
    return report


@app.local_entrypoint()
def main(out: str = "out/cohort-calibration") -> None:
    payload: dict[str, bytes] = {}
    for vendor, fixtures in FIXTURES.items():
        for name in fixtures:
            if vendor == "meta":
                path = _ROOT / "data" / "contentseal" / "originals" / name
            else:
                path = _ROOT / "data" / "synthid" / "originals" / name
            if not path.exists():
                raise SystemExit(f"fixture not found: {path}")
            payload[name] = path.read_bytes()
    payload[SEED_PROBE["source"]] = (_ROOT / SEED_PROBE["source"]).read_bytes()

    report = calibrate.remote(payload)

    out_root = Path(out)
    summary: dict = {}
    for vendor, fixtures in report.items():
        if vendor == "seed_probe":
            for key, png in fixtures.items():
                destination = out_root / "seed-probe" / f"chroma_{key}.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(png)
            continue
        summary[vendor] = {}
        for name, entry in fixtures.items():
            fixture_dir = out_root / vendor / Path(name).stem
            fixture_dir.mkdir(parents=True, exist_ok=True)
            (fixture_dir / "qwen_floor.png").write_bytes(entry["qwen_png"])
            rungs = {}
            for strength, run in entry["chroma"].items():
                (fixture_dir / f"chroma_s{float(strength):.3f}.png").write_bytes(run["png"])
                rungs[strength] = run["seconds"]
            summary[vendor][name] = {
                "chroma_rungs_seconds": rungs,
                "qwen_seconds": entry["qwen_seconds"],
            }
    (out_root / "timing.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\noutputs in {out_root}. Oracle each fixture ascending, stop at first clean.")
