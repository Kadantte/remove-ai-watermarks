# /// script
# requires-python = ">=3.11"
# dependencies = ["modal>=1.5"]
# ///
"""Compare one engine at a time on the tracked content-selection matrix.

This is an offline research harness, not a production router. Each Modal call
loads exactly one global engine and runs all 38 fixed inputs at that engine's
shipped provider floor. The shared Z-Image face repair is deliberately omitted:
the model choice changes only the global stage, while the face stage regenerates
the same original crops for both profiles.

Run from the repository root:

    uvx modal run scripts/engine_selection_study.py

The combined metrics land in the gitignored
``out/engine-selection-study/summary.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import modal

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE_ROOT = _ROOT / "data" / "evaluations" / "engine-selection"
_GPU = "H100"

app = modal.App("engine-selection-study")

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
        "scikit-image",
        "lpips",
    )
    .env({"HF_HOME": "/cache", "TORCH_HOME": "/cache/torch"})
    .add_local_dir(
        _ROOT / "src" / "remove_ai_watermarks",
        remote_path="/pkg/remove_ai_watermarks",
    )
    .add_local_dir(_FIXTURE_ROOT, remote_path="/fixtures")
)

cache = modal.Volume.from_name("chroma-hf-cache", create_if_missing=True)

_STRENGTHS = {
    "qwen-zimage": {"openai": 0.07675, "meta": 0.10},
    "chroma-zimage": {"openai": 0.09, "meta": 0.17},
}


@app.function(image=image, gpu=_GPU, volumes={"/cache": cache}, timeout=3600)
def run_engine(engine: str) -> dict[str, Any]:
    import csv
    import hashlib
    import sys
    import time

    import cv2
    import lpips
    import numpy as np
    import torch
    from PIL import Image
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    sys.path.insert(0, "/pkg")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if engine == "qwen-zimage":
        from remove_ai_watermarks._internal.qwen_zimage_pipeline import QwenZImagePipeline

        pipeline_type: Any = QwenZImagePipeline
    elif engine == "chroma-zimage":
        from remove_ai_watermarks._internal.chroma_zimage_pipeline import ChromaZImagePipeline

        pipeline_type = ChromaZImagePipeline
    else:
        raise ValueError(f"unknown engine: {engine}")
    pipeline = pipeline_type(
        device="cuda",
        torch_dtype=torch.bfloat16,
        keep_global_models_on_device=True,
        keep_face_models_on_device=False,
    )

    perceptual = lpips.LPIPS(net="alex", verbose=False).to("cuda").eval()

    def rgb_array(pil_image: Image.Image) -> np.ndarray:
        return np.asarray(pil_image.convert("RGB"), dtype=np.uint8)

    def gray(rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    def lpips_distance(source: np.ndarray, candidate: np.ndarray) -> float:
        height, width = source.shape[:2]
        scale = min(1.0, 512.0 / max(height, width))
        size = (max(1, round(width * scale)), max(1, round(height * scale)))

        def tensor(array: np.ndarray) -> torch.Tensor:
            resized = cv2.resize(array, size, interpolation=cv2.INTER_AREA)
            normalized = resized.astype(np.float32) / 127.5 - 1.0
            return torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to("cuda")

        with torch.inference_mode():
            return float(perceptual(tensor(source), tensor(candidate)).item())

    def edge_f1(source_edges: np.ndarray, candidate_gray: np.ndarray) -> float:
        candidate_edges = cv2.Canny(candidate_gray, 100, 200) > 0
        kernel = np.ones((3, 3), np.uint8)
        source_near = cv2.dilate(source_edges.astype(np.uint8), kernel) > 0
        candidate_near = cv2.dilate(candidate_edges.astype(np.uint8), kernel) > 0
        precision = float((candidate_edges & source_near).sum()) / max(1, int(candidate_edges.sum()))
        recall = float((source_edges & candidate_near).sum()) / max(1, int(source_edges.sum()))
        return 2.0 * precision * recall / max(1e-9, precision + recall)

    def source_features(
        source: np.ndarray,
        source_gray: np.ndarray,
        source_edges: np.ndarray,
        source_lapvar: float,
    ) -> dict[str, float]:
        histogram = np.bincount(source_gray.reshape(-1), minlength=256).astype(np.float64)
        probabilities = histogram[histogram > 0] / histogram.sum()
        hsv = cv2.cvtColor(source, cv2.COLOR_RGB2HSV)
        return {
            "edge_density": round(float(source_edges.mean()), 6),
            "laplacian_variance": round(source_lapvar, 6),
            "entropy_bits": round(float(-(probabilities * np.log2(probabilities)).sum()), 6),
            "mean_saturation": round(float(hsv[..., 1].mean() / 255.0), 6),
        }

    def fidelity(
        source: np.ndarray,
        candidate: np.ndarray,
        source_gray: np.ndarray,
        source_edges: np.ndarray,
        source_lapvar: float,
    ) -> dict[str, float]:
        candidate_gray = gray(candidate)
        candidate_lapvar = float(cv2.Laplacian(candidate_gray, cv2.CV_64F).var())
        return {
            "lpips_512": round(lpips_distance(source, candidate), 6),
            "ssim": round(float(structural_similarity(source_gray, candidate_gray)), 6),
            "psnr_db": round(float(peak_signal_noise_ratio(source, candidate)), 6),
            "mae": round(float(np.abs(source.astype(np.float32) - candidate).mean()), 6),
            "edge_f1": round(edge_f1(source_edges, candidate_gray), 6),
            "laplacian_ratio": round(candidate_lapvar / max(source_lapvar, 1e-9), 6),
        }

    with Path("/fixtures/content-manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        source_path = Path("/fixtures") / row["file"]
        source_image = Image.open(source_path).convert("RGB")
        strength = _STRENGTHS[engine][row["provider"]]
        started = time.perf_counter()
        output = pipeline._run_global(source_image, strength, 0)
        elapsed = time.perf_counter() - started
        source = rgb_array(source_image)
        candidate = rgb_array(output)
        source_gray = gray(source)
        source_edges = cv2.Canny(source_gray, 100, 200) > 0
        source_lapvar = float(cv2.Laplacian(source_gray, cv2.CV_64F).var())
        result = {
            "pair_id": row["pair_id"],
            "provider": row["provider"],
            "content_stratum": row["content_stratum"],
            "strength": strength,
            "seconds": round(elapsed, 3),
            "output_rgb_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
            "source_features": source_features(source, source_gray, source_edges, source_lapvar),
            "fidelity": fidelity(source, candidate, source_gray, source_edges, source_lapvar),
        }
        results.append(result)
        log.info(
            "%s %s/%s (%s/%s)",
            engine,
            index,
            len(rows),
            row["provider"],
            row["content_stratum"],
        )

    cache.commit()
    return {"engine": engine, "gpu": _GPU, "rows": results}


@app.local_entrypoint()
def main(out: str = "out/engine-selection-study") -> None:
    qwen = run_engine.remote("qwen-zimage")
    chroma = run_engine.remote("chroma-zimage")
    report = {
        "protocol": "one engine per sequential H100 invocation; global-stage factorial ablation",
        "qwen-zimage": qwen,
        "chroma-zimage": chroma,
    }
    destination = Path(out) / "summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Wrote %s", destination)
