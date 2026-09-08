"""Unit tests for how far --cpu-offload travels, per profile.

The flag sets two residency fields. The face one is read by the shared base and
reaches every profile; the global one is read by ``QwenZImagePipeline`` alone, so
on ``chroma-zimage`` and ``sdxl-zimage`` it never touches the stack the user is
trying to fit. ``WatermarkRemover`` says so before the model load.

Both halves are exercised with an uninitialized remover, so the core CI matrix
needs no diffusion dependency, model download, or GPU.
"""

from __future__ import annotations

import logging

import pytest

from remove_ai_watermarks._internal.watermark_profiles import (
    AUTO_PROFILE,
    CHROMA_ZIMAGE_PROFILE,
    GLOBAL_OFFLOAD_PROFILES,
    PROFILE_CHOICES,
    QWEN_ZIMAGE_PROFILE,
    SDXL_ZIMAGE_PROFILE,
    global_offload_supported,
)
from remove_ai_watermarks._internal.watermark_remover import WatermarkRemover

OFFLOAD_SUPPORT_CASES = (
    (QWEN_ZIMAGE_PROFILE, True),
    (CHROMA_ZIMAGE_PROFILE, False),
    (SDXL_ZIMAGE_PROFILE, False),
    # Unresolved auto appears only in preload, which builds qwen-zimage.
    (AUTO_PROFILE, True),
)


def _remover(profile: str, cpu_offload: bool) -> WatermarkRemover:
    remover = WatermarkRemover.__new__(WatermarkRemover)
    remover.model_profile = profile
    remover.cpu_offload = cpu_offload
    remover.device = "cuda"
    remover.torch_dtype = None
    remover.hf_token = None
    remover.controlnet_conditioning_scale = 1.0
    remover._progress_callback = None
    remover._qwen_zimage_pipeline = None
    return remover


class TestGlobalOffloadSupport:
    @pytest.mark.parametrize(
        ("profile", "expected"),
        OFFLOAD_SUPPORT_CASES,
    )
    def test_loaded_profile_or_preload_fallback_reports_support(self, profile: str, expected: bool) -> None:
        assert global_offload_supported(profile) is expected

    def test_cases_cover_every_profile(self) -> None:
        assert {profile for profile, _expected in OFFLOAD_SUPPORT_CASES} == set(PROFILE_CHOICES)

    def test_the_underscore_spelling_resolves(self) -> None:
        assert global_offload_supported("qwen_zimage") is True

    def test_the_set_names_a_profile_that_exists(self) -> None:
        # A renamed profile must not leave the flag silently unsupported everywhere.
        assert set(PROFILE_CHOICES) >= GLOBAL_OFFLOAD_PROFILES


class TestUnsupportedProfileWarns:
    @pytest.mark.parametrize("profile", [CHROMA_ZIMAGE_PROFILE, SDXL_ZIMAGE_PROFILE])
    def test_it_names_the_profile_and_the_flag(self, profile: str, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            _remover(profile, cpu_offload=True)._warn_if_global_offload_unsupported()
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "--cpu-offload" in message
        assert profile in message

    def test_qwen_zimage_is_quiet(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            _remover(QWEN_ZIMAGE_PROFILE, cpu_offload=True)._warn_if_global_offload_unsupported()
        assert caplog.records == []

    def test_a_run_that_did_not_ask_for_offload_is_quiet(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            _remover(CHROMA_ZIMAGE_PROFILE, cpu_offload=False)._warn_if_global_offload_unsupported()
        assert caplog.records == []

    def test_preloaded_auto_uses_qwen_offload_without_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from remove_ai_watermarks._internal import qwen_zimage_pipeline

        captured: dict[str, object] = {}

        class Recorder:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

            def preload(self, *, global_only: bool = False) -> None:
                captured["global_only"] = global_only

        monkeypatch.setattr(qwen_zimage_pipeline, "QwenZImagePipeline", Recorder)
        with caplog.at_level(logging.WARNING):
            _remover(AUTO_PROFILE, cpu_offload=True).preload(global_only=True)

        assert caplog.records == []
        assert captured["keep_global_models_on_device"] is False
        assert captured["global_only"] is True

    def test_the_load_path_asks_before_building_the_stack(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The warning has to reach the user ahead of the download, and it has to see
        # the profile auto resolved to, so the seam is the load and not __init__.
        remover = _remover(CHROMA_ZIMAGE_PROFILE, cpu_offload=True)
        order: list[str] = []
        monkeypatch.setattr(
            WatermarkRemover,
            "_warn_if_global_offload_unsupported",
            lambda self: order.append("warned"),
        )

        def _explode(*_args: object, **_kwargs: object) -> object:
            order.append("loaded")
            raise ImportError("no diffusion dependency in this environment")

        monkeypatch.setattr(
            "remove_ai_watermarks._internal.chroma_zimage_pipeline.ChromaZImagePipeline",
            _explode,
        )
        with pytest.raises(ImportError):
            remover._load_qwen_zimage_pipeline()
        assert order == ["warned", "loaded"]
