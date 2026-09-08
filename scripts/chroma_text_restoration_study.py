# /// script
# requires-python = ">=3.11"
# dependencies = ["modal>=1.5"]
# ///
"""Paired offline study of verified-text restoration on Qwen and Chroma.

Each generative profile runs exactly once per fixture. Its already loaded VAE
produces the restoration donor, then the shared production compositor creates
the restored candidate from that one base output. The experiment therefore
measures four outputs (Qwen/Chroma x base/restored) without accidentally timing
or evaluating a second generation for either restoration arm.

Run from the repository root:

    uvx modal run scripts/chroma_text_restoration_study.py

Outputs stay under ``out/text-restoration-engine-study``. Submit exact output
hashes to the matching provider oracle before using fidelity metrics to choose
between engines; fidelity only ranks candidates after the scrub verdict passes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import modal

_ROOT = Path(__file__).resolve().parent.parent
_GPU = "H100"

app = modal.App("chroma-text-restoration-study")

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
        "opencv-python-headless<5",
        "onnxruntime>=1.24.0",
        "huggingface-hub>=0.20.0",
    )
    .env({"HF_HOME": "/cache"})
    .add_local_dir(_ROOT / "src" / "remove_ai_watermarks", remote_path="/pkg/remove_ai_watermarks")
)

cache = modal.Volume.from_name("chroma-hf-cache", create_if_missing=True)


def _png_bytes(image: Any) -> bytes:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _metric_inputs(source: Any, boxes: list[tuple[int, int, int, int]]) -> tuple[Any, Any]:
    import numpy as np

    source_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    mask = np.zeros(source_rgb.shape[:2], dtype=bool)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = True
    return source_rgb, mask


def _metrics(source_rgb: Any, candidate: Any, mask: Any) -> dict[str, float]:
    import math

    import numpy as np

    candidate_rgb = np.asarray(candidate.convert("RGB"), dtype=np.uint8)
    signed_delta = source_rgb.astype(np.float32) - candidate_rgb
    absolute_delta = np.abs(signed_delta)
    mse = float((signed_delta**2).mean())
    return {
        "whole_mae": round(float(absolute_delta.mean()), 6),
        "text_box_mae": round(float(absolute_delta[mask].mean()), 6),
        "outside_box_mae": round(float(absolute_delta[~mask].mean()), 6),
        "psnr_db": round(99.0 if mse <= 1e-12 else 10.0 * math.log10(255.0**2 / mse), 6),
    }


@app.function(image=image, gpu=_GPU, volumes={"/cache": cache}, timeout=3600)
def run(payload: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    import gc
    import io
    import sys
    import time

    import torch
    from PIL import Image

    sys.path.insert(0, "/pkg")

    from remove_ai_watermarks._internal.chroma_zimage_pipeline import ChromaZImagePipeline
    from remove_ai_watermarks._internal.qwen_zimage_pipeline import QwenZImagePipeline
    from remove_ai_watermarks._internal.text_restoration import VerifiedTextLine, restore_verified_text
    from remove_ai_watermarks._internal.two_stage_pipeline import detect_faces
    from remove_ai_watermarks._internal.watermark_profiles import resolve_strength

    sources: dict[str, tuple[Image.Image, str, tuple[VerifiedTextLine, ...]]] = {}
    for name, item in payload.items():
        source = Image.open(io.BytesIO(item["source"])).convert("RGB")
        lines = tuple(
            VerifiedTextLine(
                tuple(line["box"]),
                line.get("text"),
                line.get("script"),
                float(line.get("angle", 0.0)),
            )
            for line in item["lines"]
        )
        sources[name] = source, str(item["vendor"]), lines

    results: dict[str, dict[str, Any]] = {
        name: {"vendor": vendor, "engines": {}} for name, (_, vendor, _) in sources.items()
    }
    metric_inputs = {
        name: _metric_inputs(source, [line.box for line in lines]) for name, (source, _vendor, lines) in sources.items()
    }

    for profile, pipeline_type in (
        ("qwen-zimage", QwenZImagePipeline),
        ("chroma-zimage", ChromaZImagePipeline),
    ):
        print(f"[{profile}] loading...", flush=True)
        pipeline = pipeline_type(
            device="cuda",
            torch_dtype=torch.bfloat16,
            keep_global_models_on_device=True,
            keep_face_models_on_device=True,
            progress_callback=lambda message, p=profile: print(f"[{p}] {message}", flush=True),
        )
        for name, (source, vendor, lines) in sources.items():
            face_count = len(detect_faces(source)) if profile == "chroma-zimage" else 0
            strength = resolve_strength(
                None,
                vendor,
                profile,
                size=source.size,
                face_count=face_count,
            )
            print(f"[{profile}] {name}: strength={strength:.6f}, faces={face_count}", flush=True)

            started = time.perf_counter()
            donor = pipeline._vae_roundtrip(source)
            donor_seconds = time.perf_counter() - started

            started = time.perf_counter()
            base = pipeline.run(source, strength=strength, seed=0)
            generation_seconds = time.perf_counter() - started

            started = time.perf_counter()
            restored = restore_verified_text(source, base, donor, lines)
            restoration_seconds = time.perf_counter() - started

            source_rgb, text_mask = metric_inputs[name]
            base_png = _png_bytes(base)
            restored_png = _png_bytes(restored)
            donor_png = _png_bytes(donor)
            results[name]["engines"][profile] = {
                "strength": strength,
                "face_count": face_count,
                "donor_seconds": round(donor_seconds, 3),
                "generation_seconds": round(generation_seconds, 3),
                "restoration_seconds": round(restoration_seconds, 3),
                "base_sha256": hashlib.sha256(base_png).hexdigest(),
                "restored_sha256": hashlib.sha256(restored_png).hexdigest(),
                "donor_sha256": hashlib.sha256(donor_png).hexdigest(),
                "base_metrics": _metrics(source_rgb, base, text_mask),
                "restored_metrics": _metrics(source_rgb, restored, text_mask),
                "base_png": base_png,
                "restored_png": restored_png,
                "donor_png": donor_png,
            }

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    cache.commit()
    return results


@app.local_entrypoint()
def main(out: str = "out/text-restoration-engine-study") -> None:
    annotations = json.loads((_ROOT / "data/evaluations/fidelity/text-lines.json").read_text(encoding="utf-8"))
    vendors = {
        "ChatGPT Image May 31, 2026, 02_02_23 PM.png": "openai",
        "ChatGPT Image May 31, 2026, 02_03_55 PM.png": "openai",
        "Gemini_Generated_Image_633uuy633uuy633u.png": "google",
    }
    source_root = _ROOT / "data/synthid/originals"
    payload = {
        name: {
            "source": (source_root / name).read_bytes(),
            "vendor": vendor,
            "lines": annotations[name],
        }
        for name, vendor in vendors.items()
    }

    results = run.remote(payload)
    out_root = Path(out)
    summary: dict[str, Any] = {"gpu": _GPU, "fixtures": {}}
    for name, fixture in results.items():
        fixture_dir = out_root / Path(name).stem
        fixture_dir.mkdir(parents=True, exist_ok=True)
        summary["fixtures"][name] = {"vendor": fixture["vendor"], "engines": {}}
        for profile, result in fixture["engines"].items():
            for variant in ("base", "restored", "donor"):
                (fixture_dir / f"{profile}_{variant}.png").write_bytes(result[f"{variant}_png"])
            summary["fixtures"][name]["engines"][profile] = {
                key: value for key, value in result.items() if not key.endswith("_png")
            }

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"outputs in {out_root}")
