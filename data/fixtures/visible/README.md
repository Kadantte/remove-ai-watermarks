# Visible-mark example gallery

One committed example per registered visible mark, so the repository carries a
working sample of everything it supports. `tests/test_visible_examples.py` holds
both sides to it: a mark registered without an example fails the suite, and so
does an engine that stops detecting its own example.

## What these files are

Every example is SYNTHETIC: `scripts/render_visible_examples.py` composites the
mark's committed alpha or silhouette onto a deterministic generated base photo at
the engine's measured geometry. Controlled provider captures were used to solve
some alpha assets, including Samsung's measured opacity; the remaining shapes are
repository-owned renders rather than copied vendor rasters.

The examples demonstrate DETECTION geometry and house style, not vendor raster
fidelity; real-world variants (fonts, opacities, sizes) are covered by the
engines' recorded calibration evidence.

## Regeneration

    uv run python scripts/render_visible_examples.py

The generator self-verifies: it fails (exit 1) if any registered mark does not
detect on its own example, so regeneration is the fix point for drift.

## Layout

    <mark-key>/example.png   1536x2048..2048x2048 PNG, one per image mark
    <mark-key>/example.mp4   960x540 90-frame clip, one per video mark
                            (kling carries both: it is registered in both registries)

Special cases: `gemini` composites the sparkle alpha map at the provider's
configured position; `jimeng_pill` is the capture-less pill at the measured
3:4 portrait geometry; `microsoft` is the opaque white pill with dark text
holes (the discriminator its detector keys on). Video examples composite the
detector's own synthetic template on every frame; where two marks share a
shape family the example carries the discriminative variant (`veo` the legacy
text form, `kling` the logo-plus-wordmark pair flush to the edge), because the
temporal selection resolves cross-template ties by table order.
