from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_engine_selection_study.py"
SPEC = importlib.util.spec_from_file_location("analyze_engine_selection_study", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(*, ssim: float, lpips: float) -> dict[str, Any]:
    common = {
        "psnr_db": 30.0,
        "mae": 2.0,
        "edge_f1": 0.8,
        "laplacian_ratio": 0.9,
    }
    return {
        "pair_id": "00",
        "provider": "openai",
        "content_stratum": "test",
        "fidelity": {**common, "ssim": ssim, "lpips_512": lpips},
    }


def test_exact_sign_test_two_sided() -> None:
    assert MODULE._sign_test_two_sided(19, 0) == pytest.approx(2 / 2**19)
    assert MODULE._sign_test_two_sided(12, 7) == pytest.approx(0.359283447265625)
    assert MODULE._sign_test_two_sided(0, 0) == 1.0


def test_analysis_respects_each_metrics_preferred_direction() -> None:
    report = {
        "qwen-zimage": {"rows": [_row(ssim=0.8, lpips=0.1)]},
        "chroma-zimage": {"rows": [_row(ssim=0.9, lpips=0.2)]},
    }

    metrics = MODULE.analyze(report)["providers"]["openai"]["metrics"]

    assert metrics["ssim"]["chroma_wins"] == 1
    assert metrics["ssim"]["qwen_wins"] == 0
    assert metrics["lpips_512"]["chroma_wins"] == 0
    assert metrics["lpips_512"]["qwen_wins"] == 1
