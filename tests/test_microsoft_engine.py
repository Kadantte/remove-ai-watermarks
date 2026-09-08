"""Tests for the measured Microsoft top-right AI-badge engine.

The covered variant is a white top-right pill with dark internal shapes. The
2026-08-27 calibration kept visually confirmed carriers, provenance-only files,
and no-signal controls separate. These tests pin the load-bearing constants --
especially the long-side scale basis and the internal holes as the discriminator.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from remove_ai_watermarks import watermark_registry as registry
from remove_ai_watermarks.microsoft_engine import (
    _ALPHA_HEIGHT_FRAC,
    _ALPHA_WIDTH_FRAC,
    MicrosoftEngine,
    _alpha_template,
)

_INSET = 0.010  # measured pill inset from the top/right edges (long-side fraction)


def _pill_geometry(w: int, h: int) -> tuple[int, int, int, int]:
    long_side = max(w, h)
    pw = int(_ALPHA_WIDTH_FRAC * long_side)
    ph = max(4, int(_ALPHA_HEIGHT_FRAC * long_side))
    pad = int(_INSET * long_side)
    return w - pad - pw, pad, pw, ph


def _compose(w: int, h: int, bg: float = 110.0):
    """Composite the synthetic pill at its measured size onto a flat background."""
    img = np.full((h, w, 3), bg, np.uint8)
    at = _alpha_template()
    x0, y0, pw, ph = _pill_geometry(w, h)
    pill = cv2.resize(at, (pw, ph))
    region = img[y0 : y0 + ph, x0 : x0 + pw]
    bright = pill > 0.6
    region[bright] = 245
    # Internal holes are dark ink inside the pill, not background.
    region[~bright] = 45
    return img, (x0, y0, pw, ph)


def _plain_pill(w: int, h: int, text: str | None = None) -> np.ndarray:
    """Return a white rounded pill without the expected holes, or with foreign text."""
    img = np.full((h, w, 3), 110.0, np.uint8)
    x0, y0, pw, ph = _pill_geometry(w, h)
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (245, 245, 245), -1)
    cv2.circle(img, (x0 + ph // 2, y0 + ph // 2), ph // 3, (110, 110, 110), -1)
    if text:
        cv2.putText(img, text, (x0 + ph, y0 + ph // 2 + ph // 6), cv2.FONT_HERSHEY_SIMPLEX, ph / 90.0, (45, 45, 45), 1)
    return img


class TestLocate:
    def test_box_anchored_top_right(self):
        eng = MicrosoftEngine()
        loc = eng.locate(np.zeros((1024, 1024, 3), np.uint8))
        assert loc.x + loc.w == pytest.approx(1024 - int(0.004 * 1024), abs=2)
        assert loc.y == pytest.approx(int(0.003 * 1024), abs=2)

    def test_box_scales_with_long_side_not_width(self):
        # Measured: the pill tracks the render dimension, so a 1024x1536 portrait
        # carries the SAME pill size as 1536x1024. A width basis undersized the
        # template by the aspect ratio and dropped every portrait carrier.
        eng = MicrosoftEngine()
        portrait = eng.locate(np.zeros((1536, 1024, 3), np.uint8))
        landscape = eng.locate(np.zeros((1024, 1536, 3), np.uint8))
        assert portrait.w == landscape.w
        small = eng.locate(np.zeros((720, 480, 3), np.uint8))
        assert small.w < portrait.w


class TestConfig:
    def test_provenance_relaxation_is_the_measured_07(self):
        # The relaxed band was measured on the OCR-censused MS cohort: 257
        # badge-less files max 0.251, so the 0.266 relaxed gate admits the three
        # faint badges in [0.251, 0.38) with zero measured false fills. Do not
        # move the factor without re-censusing the badge-less cohort.
        assert MicrosoftEngine().config.provenance_ncc_factor == 0.7

    def test_long_scale_basis(self):
        assert MicrosoftEngine().config.scale_basis == "long"

    def test_threshold_and_geometry_pins(self):
        from remove_ai_watermarks.microsoft_engine import (
            DETECT_NCC_THRESHOLD,
            MARGIN_RIGHT_FRAC,
            WM_WIDTH_FRAC,
        )

        assert pytest.approx(0.38) == DETECT_NCC_THRESHOLD  # controls max 0.293; carriers max 0.579
        assert pytest.approx(0.170) == WM_WIDTH_FRAC
        assert pytest.approx(0.004) == MARGIN_RIGHT_FRAC

    def test_registry_row(self):
        mark = registry.get_mark("microsoft")
        assert mark.location == "top-right"
        assert mark.label == "Microsoft top-right AI badge"
        assert mark.in_auto
        assert mark.provenance_platform_tokens == ("microsoft",)
        assert mark.label_regime is None  # not a China-TC260 mark


class TestDetect:
    @pytest.mark.parametrize(("w", "h"), [(1024, 1024), (1536, 1024), (1024, 1536), (720, 480), (1206, 1194)])
    def test_composites_detected_across_sizes(self, w, h):
        eng = MicrosoftEngine()
        img, _box = _compose(w, h)
        det = eng.detect(img)
        assert det.detected, f"{w}x{h}: conf={det.confidence:.3f}"
        assert det.confidence >= 0.38

    def test_portrait_composite_region_covers_pill(self):
        eng = MicrosoftEngine()
        img, (x, y, pw, ph) = _compose(1024, 1536)
        det = eng.detect(img)
        assert det.detected
        rx, ry, rw, _rh = det.region
        assert abs((rx + rw) - (x + pw)) < 0.08 * pw
        assert abs(ry - y) < 0.4 * ph

    def test_clean_gradient_not_detected(self):
        eng = MicrosoftEngine()
        ramp = np.tile(np.linspace(0, 255, 1024, dtype=np.uint8), (1024, 1))
        img = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
        assert not eng.detect(img).detected

    def test_plain_white_pill_not_detected(self):
        # The expected internal holes are the discriminator: any other bright rounded
        # element in the corner must not attribute Microsoft.
        eng = MicrosoftEngine()
        assert not eng.detect(_plain_pill(1024, 1024)).detected

    def test_foreign_text_pill_not_detected(self):
        eng = MicrosoftEngine()
        assert not eng.detect(_plain_pill(1024, 1024, text="Sample Text")).detected

    def test_busy_content_corner_not_detected(self):
        # A photo-like textured corner must stay under the gate.
        eng = MicrosoftEngine()
        rng = np.random.default_rng(7)
        img = rng.integers(0, 255, (1024, 1024, 3), dtype=np.uint8)
        img = cv2.GaussianBlur(img, (0, 0), 3)
        assert not eng.detect(img).detected


class TestMask:
    def test_footprint_covers_the_pill(self):
        eng = MicrosoftEngine()
        img, (x, y, pw, ph) = _compose(1536, 1024)
        det = eng.detect(img)
        assert det.detected
        mask = eng.footprint_mask(img, detection=det)
        assert mask.shape[:2] == img.shape[:2]
        ys, xs = np.where(mask > 0)
        assert xs.min() >= x - 0.15 * pw
        assert xs.max() <= x + pw + 0.15 * pw
        assert ys.min() >= y - 0.3 * ph
        assert ys.max() <= y + ph + 0.3 * ph
        # the fill must cover the pill area, not just the text glyphs
        assert float(mask[y : y + ph, x : x + pw].mean()) > 0.4
