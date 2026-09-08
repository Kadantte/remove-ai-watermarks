"""The Qwen global stage for the two-stage regeneration recipe.

The shared recipe (the Z-Image face stage, sizing, compositing, the prompt cache,
and the run orchestration) lives in
``remove_ai_watermarks._internal.two_stage_pipeline``; this module adds the
Qwen-Image-2512 global pass with Canny conditioning and its Lightning LoRA
distillation.
"""

# DiffSynth, torch, transformers, and cv2 expose mostly untyped tensor/array APIs.
# Keep the relaxation local to this optional ML boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnnecessaryComparison=false
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, ClassVar

from PIL import Image

from remove_ai_watermarks._internal.two_stage_pipeline import (
    _GLOBAL_NEGATIVE,
    _GLOBAL_PROMPT,
    TwoStageZImagePipeline,
    _cache_static_prompt_embeddings,
    _resize_to_target,
    build_canny_control_image,
    edge_pad_to_grid,
)

log = logging.getLogger(__name__)

QWEN_IMAGE_2512_MODEL_ID = "Qwen/Qwen-Image-2512"
QWEN_CANNY_CONTROLNET_MODEL_ID = "DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny"
QWEN_LIGHTNING_MODEL_ID = "lightx2v/Qwen-Image-2512-Lightning"
QWEN_LIGHTNING_PATTERN = "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"

GLOBAL_STEPS = 4
GLOBAL_CFG = 1.0
# Below this floor the mandatory Qwen stack streams from disk, which is what makes a
# 20B model runnable on a consumer card at all; at or above it, streaming is pure
# waste. Set equal to the face floor rather than lower because that is the
# configuration actually measured with both stacks resident; a tighter gate is
# plausible but unvalidated. Benchmark in docs/module-internals.md, "CPU offload".
RESIDENT_GLOBAL_MODEL_MIN_VRAM_GIB = 64.0

# DiffSynth identifies a pipeline unit by what it produces, so these tuples are how
# the prompt stage is located inside each pipeline and how its cache file is keyed.
_QWEN_PROMPT_OUTPUTS = ("prompt_emb", "prompt_emb_mask")


def resolve_global_model_residency(
    requested: bool | None,
    *,
    total_memory_gib: float,
) -> bool:
    """Keep the mandatory Qwen stack resident when explicitly requested or safely sized."""
    if requested is not None:
        return requested
    return total_memory_gib >= RESIDENT_GLOBAL_MODEL_MIN_VRAM_GIB


def build_global_kwargs(
    image: Image.Image,
    *,
    strength: float,
    seed: int | None,
    controlnet_input: Any,
) -> dict[str, Any]:
    """Build the DiffSynth Qwen call shape without importing the ML runtime."""
    input_image = _resize_to_target(image)
    width, height = input_image.size
    return {
        "prompt": _GLOBAL_PROMPT,
        "negative_prompt": _GLOBAL_NEGATIVE,
        "cfg_scale": GLOBAL_CFG,
        "input_image": input_image,
        "denoising_strength": float(strength),
        "height": height,
        "width": width,
        "seed": seed,
        "rand_device": "cpu",
        "num_inference_steps": GLOBAL_STEPS,
        "exponential_shift_mu": math.log(3.0),
        "blockwise_controlnet_inputs": [controlnet_input],
    }


@dataclass
class QwenZImagePipeline(TwoStageZImagePipeline):
    """Lazy runtime for the two-stage Qwen/Z-Image profile."""

    profile_name: ClassVar[str] = "qwen-zimage"

    def __post_init__(self) -> None:
        super().__post_init__()
        self._qwen_pipe: Any = None

    def _keep_global_models_resident(self) -> bool:
        return resolve_global_model_residency(
            self.keep_global_models_on_device,
            total_memory_gib=self._total_vram_gib(),
        )

    def _qwen_vram_config(self) -> dict[str, Any]:
        import torch

        if self._keep_global_models_resident():
            # Same fp8 storage and bf16 computation as the streaming config, but the
            # weights never leave the GPU. Passing no "disk" anywhere is what makes
            # this work: DiffSynth decides `disk_offload` once from `offload_dtype`,
            # so a card large enough to hold the stack skips both the `to("meta")`
            # drop and the DiskMap re-read entirely.
            return {
                "offload_dtype": torch.float8_e4m3fn,
                "offload_device": "cuda",
                "onload_dtype": torch.float8_e4m3fn,
                "onload_device": "cuda",
                "preparing_dtype": torch.float8_e4m3fn,
                "preparing_device": "cuda",
                "computation_dtype": torch.bfloat16,
                "computation_device": "cuda",
            }
        return {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": torch.float8_e4m3fn,
            "onload_device": "cpu",
            "preparing_dtype": torch.float8_e4m3fn,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }

    def _load_global(self) -> Any:
        if self._qwen_pipe is not None:
            return self._qwen_pipe
        self._require_cuda()
        os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
        if self.hf_token:
            os.environ.setdefault("HF_TOKEN", self.hf_token)
        try:
            from diffsynth.pipelines.qwen_image import ControlNetInput, ModelConfig, QwenImagePipeline
        except ImportError as exc:
            raise ImportError(
                "The qwen-zimage pipeline needs the optional dependency group. "
                "Install: pip install 'remove-ai-watermarks[qwen-zimage]'"
            ) from exc

        self._progress("Loading Qwen-Image-2512, Lightning LoRA, and Canny ControlNet...")
        config = self._qwen_vram_config()
        text_encoder_config = ModelConfig(
            model_id=QWEN_IMAGE_2512_MODEL_ID,
            origin_file_pattern="text_encoder/model*.safetensors",
            **config,
        )
        model_configs = [
            ModelConfig(
                model_id=QWEN_IMAGE_2512_MODEL_ID,
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                **config,
            ),
            text_encoder_config,
            ModelConfig(
                model_id=QWEN_IMAGE_2512_MODEL_ID,
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                **config,
            ),
            ModelConfig(
                model_id=QWEN_CANNY_CONTROLNET_MODEL_ID,
                origin_file_pattern="model.safetensors",
                **config,
            ),
        ]
        prompt_cached = self._prompt_is_cached(QWEN_IMAGE_2512_MODEL_ID, _QWEN_PROMPT_OUTPUTS, _GLOBAL_PROMPT)
        if prompt_cached:
            model_configs.remove(text_encoder_config)
            log.info("Qwen prompt embedding is cached; loading the stack without its text encoder")
        pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=self.torch_dtype,
            device=self.device,
            model_configs=model_configs,
            tokenizer_config=ModelConfig(
                model_id=QWEN_IMAGE_2512_MODEL_ID,
                origin_file_pattern="tokenizer/",
            ),
            vram_limit=self._vram_limit(),
        )
        from diffsynth.diffusion import FlowMatchScheduler

        # Avoid the base Qwen scheduler's terminal rescale for the distilled LoRA.
        # With exponential_shift_mu=log(3), this is the closest DiffSynth equivalent
        # of the source graph's four-step sgm_uniform + AuraFlow shift 3.
        pipe.scheduler = FlowMatchScheduler("Qwen-Image-Lightning")
        lightning = ModelConfig(
            model_id=QWEN_LIGHTNING_MODEL_ID,
            origin_file_pattern=QWEN_LIGHTNING_PATTERN,
        )
        pipe.load_lora(pipe.dit, lightning, alpha=0.8)
        if self.cache_prompt_embeddings:
            _cache_static_prompt_embeddings(
                pipe,
                _QWEN_PROMPT_OUTPUTS,
                model_id=QWEN_IMAGE_2512_MODEL_ID,
                require_cache=prompt_cached,
            )
        self._qwen_pipe = (pipe, ControlNetInput)
        return self._qwen_pipe

    def _run_global(self, image: Image.Image, strength: float, seed: int | None) -> Image.Image:
        pipe, controlnet_input_cls = self._load_global()
        input_image = _resize_to_target(image)
        control = build_canny_control_image(input_image)
        control_input = controlnet_input_cls(
            image=control,
            scale=float(self.controlnet_conditioning_scale),
        )
        self._progress(f"Running Qwen-Image-2512 Canny pass: strength={strength:.4f}, steps={GLOBAL_STEPS}...")
        result = pipe(
            **build_global_kwargs(
                input_image,
                strength=strength,
                seed=seed,
                controlnet_input=control_input,
            )
        )
        if result.size != image.size:
            result = result.resize(image.size, Image.Resampling.LANCZOS)
        return result.convert("RGB")

    def _vae_roundtrip(self, image: Image.Image) -> Image.Image:
        """Reconstruct source pixels through the already loaded Qwen VAE."""
        import torch

        pipe, _controlnet_input_cls = self._load_global()
        source_width, source_height = image.size
        padded = edge_pad_to_grid(image, 8)
        pipe.load_models_to_device(["vae"])
        tensor = pipe.preprocess_image(padded).to(device=self.device, dtype=self.torch_dtype)
        with torch.inference_mode():
            latents = pipe.vae.encode(tensor)
            decoded = pipe.vae.decode(latents)
        return pipe.vae_output_to_image(decoded).crop((0, 0, source_width, source_height)).convert("RGB")
