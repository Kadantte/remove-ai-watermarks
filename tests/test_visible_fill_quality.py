"""Cheap constructed-reference regressions for every visible image mark."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from remove_ai_watermarks import watermark_registry as wr
from scripts.fill_quality import SLOT_STAMPABLE, STAMPABLE, psnr, stamp_any
from scripts.invisible_quality_audit import _ssim
from scripts.render_visible_examples import build_pair

_IMAGE_KEYS = tuple(mark.key for mark in wr.known_marks())

# The second case changes both the generated background and the geometry. Samsung's
# faint overlay needs the larger width its calibrated detector covers.
_ALTERNATE_SIZE = {
    "gemini": (1280, 960),
    "doubao": (1280, 960),
    "jimeng": (1280, 960),
    "qwen": (1280, 1280),
    "kling": (1280, 960),
    "yuanbao": (1280, 960),
    "samsung": (2304, 1728),
    "runninghub": (1280, 960),
    "baidu": (1280, 960),
    "liblib": (960, 1280),
    "microsoft": (1280, 960),
    "jimeng_pill": (960, 1280),
}

_MIN_MASK_COVERAGE = 0.94
_MIN_FILLED_PSNR = 25.0
_MIN_FILLED_SSIM = 0.90


def _score_box(clean: np.ndarray, filled: np.ndarray, changed: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(changed)
    assert ys.size > 0
    pad = 8
    y0, y1 = max(0, int(ys.min()) - pad), min(clean.shape[0], int(ys.max()) + 1 + pad)
    x0, x1 = max(0, int(xs.min()) - pad), min(clean.shape[1], int(xs.max()) + 1 + pad)
    truth = clean[y0:y1, x0:x1]
    output = filled[y0:y1, x0:x1]
    return psnr(output, truth), _ssim(
        cv2.cvtColor(output, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(truth, cv2.COLOR_BGR2GRAY),
    )


def test_full_quality_harness_covers_every_registered_image_mark() -> None:
    assert {*STAMPABLE, *SLOT_STAMPABLE} == set(_IMAGE_KEYS)
    for key in _IMAGE_KEYS:
        clean, _marked = build_pair(key)
        assert stamp_any(clean, key) is not None, key


@pytest.mark.parametrize("key", _IMAGE_KEYS)
@pytest.mark.parametrize("case", ["canonical", "alternate"])
def test_cv2_mask_quality_smoke(key: str, case: str) -> None:
    """One fill covers the stamp, clears detection, and preserves local quality."""
    size = None if case == "canonical" else _ALTERNATE_SIZE[key]
    seed = 7 if case == "canonical" else 23
    clean, marked = build_pair(key, size=size, seed=seed)
    changed = np.any(marked != clean, axis=2)
    assert np.any(changed), key

    mark = wr.get_mark(key)
    before = mark.detect(marked, provenance=False)
    assert before.detected, f"{key}/{case}: confidence {before.confidence:.3f}"
    localization = mark.localize(marked, force=False, detection=before)
    assert localization.mask is not None, f"{key}/{case}: no mask"

    coverage = float(np.count_nonzero(localization.mask[changed])) / float(np.count_nonzero(changed))
    assert coverage >= _MIN_MASK_COVERAGE, f"{key}/{case}: mask coverage {coverage:.3f}"

    filled, region = mark.remove(marked, backend="cv2", detection=before)
    assert region is not None, key
    assert np.array_equal(filled[localization.mask == 0], marked[localization.mask == 0]), key

    after = mark.detect(filled, provenance=False)
    assert not after.detected, f"{key}/{case}: residual confidence {after.confidence:.3f}"
    filled_psnr, filled_ssim = _score_box(clean, filled, changed)
    assert filled_psnr >= _MIN_FILLED_PSNR, f"{key}/{case}: PSNR {filled_psnr:.2f} dB"
    assert filled_ssim >= _MIN_FILLED_SSIM, f"{key}/{case}: SSIM {filled_ssim:.4f}"
