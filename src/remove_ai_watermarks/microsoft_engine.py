"""Microsoft top-right AI-badge detector/localizer.

This engine covers one measured Microsoft output variant: a white pill with dark
internal shapes in the top-right corner. The evaluated files used both "Made with
AI" and "AI-Generated" wording. This is narrower than Microsoft's documented
watermark feature, which can use a Copilot icon or text and can place the mark in
other positions. A Microsoft provenance signal therefore does not establish that
this exact visible variant is present.

Detection matches a synthetic pill silhouette (white pill with the sparkle and
text KNOCKED OUT) against the top-hat blob of the located box: the holes are what
discriminate this pill from any other bright rounded element in the corner.
Removal is the shared **localize -> fill**; the glyph-bbox :meth:`footprint_mask`
covers the whole pill including its text.

The tuned numbers below were remeasured on 2026-08-27 with the registered engine
and ``scripts/registered_mark_calibrate.py``. The arms were kept distinct: 17
visually confirmed carriers, 343 Microsoft-provenance files whose visible-mark
status was not adjudicated, and 1200 non-overlapping no-signal controls:

  * Geometry is single-mode and tight: pill 0.152 x 0.040 of the LONG side
    (aspect 3.73-3.89 over 720..1536 px), margins ~0.010/0.007 of the same basis. One size
    mode, so the shared 3-rung ladder is untouched and the locate box simply
    wraps the pill with NCC slack.
  * Provenance relaxation 0.7 (relaxed gate 0.266), enabled 2026-08-28 when the
    cohort the strict-only note was waiting for became available: an OCR badge
    census split the 343 Microsoft-C2PA uploads into 86 badge carriers and 257
    true badge-less files (the watermark is a per-user opt-in, so 75% of MS
    uploads carry none). Badge-less max 0.251 / p99 0.213, so the relaxed band
    [0.251, 0.38) holds three genuine faint badges and zero false fills
    (measured 3/3; doubao ships 0.7 on a 58%-precision band). Strict controls
    max 0.293 / p99 0.200 vs the 0.38 gate.
  * Front-end "binary": the pill is a bold opaque overlay; the tophat blob is
    solid with dark-text holes, exactly the template's shape.
"""

# pyright: reportUnusedFunction=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from remove_ai_watermarks import _text_mark_engine
from remove_ai_watermarks._text_mark_engine import TextMarkConfig, TextMarkEngine

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Locate geometry as a fraction of the image LONG side (measured; scale_basis="long":
# on 1024x1536 portraits the pill tracks the 1536, and a width basis undersized
# the template until the portrait carriers fell to 0.15-0.32 NCC).
# The box wraps the measured pill rect (0.152W x 0.040W) with NCC slack; margins
# sit inside the pill's own ~0.010W-right / ~0.007W-top insets.
WM_WIDTH_FRAC = 0.170
WM_HEIGHT_FRAC = 0.055
MARGIN_RIGHT_FRAC = 0.004
MARGIN_TOP_FRAC = 0.003

# Glyph appearance: a bright near-white pill (luma ~245), gray-scale (sat < 60).
MAX_SATURATION = 60
LOGO_MIN_LUMA = 170
TOPHAT_DELTA = 10

# Calibrated 2026-08-27: non-overlapping no-signal controls (n=1200) max 0.293 /
# p99 0.200; visually confirmed carriers (n=17) p50 0.519 / p90 0.578 / max
# 0.579, with 15/17 above the 0.38 gate. The two misses score 0.249 and 0.315.
DETECT_MIN_COVERAGE = 0.30  # the pill fills most of its box; content corners do not
DETECT_NCC_THRESHOLD = 0.38

# Pill silhouette geometry (fraction of width): 0.152W x 0.040W, aspect ~3.78.
_ALPHA_NATIVE_WIDTH = 335
_ALPHA_WIDTH_FRAC = 0.152
_ALPHA_HEIGHT_FRAC = 0.040

_CONFIG = TextMarkConfig(
    name="Microsoft top-right AI badge",
    asset_name="microsoft_alpha.png",
    corner="tr",
    margin_floor=2,
    width_frac=WM_WIDTH_FRAC,
    height_frac=WM_HEIGHT_FRAC,
    margin_x_frac=MARGIN_RIGHT_FRAC,
    margin_bottom_frac=MARGIN_TOP_FRAC,
    max_saturation=MAX_SATURATION,
    logo_min_luma=LOGO_MIN_LUMA,
    tophat_delta=TOPHAT_DELTA,
    morph_open_size=5,
    detect_min_coverage=DETECT_MIN_COVERAGE,
    detect_ncc_threshold=DETECT_NCC_THRESHOLD,
    alpha_width_frac=_ALPHA_WIDTH_FRAC,
    alpha_height_frac=_ALPHA_HEIGHT_FRAC,
    min_gw=24,
    detect_frontend="binary",
    scale_basis="long",
    provenance_ncc_factor=0.7,
)


def _alpha_template() -> NDArray[Any] | None:
    """The bundled Microsoft pill template (float [0,1]), or None."""
    return _text_mark_engine.load_alpha_template(_CONFIG.asset_name)


class MicrosoftEngine(TextMarkEngine):
    """Detect/localize the measured Microsoft top-right AI badge."""

    def __init__(self) -> None:
        super().__init__(_CONFIG)
