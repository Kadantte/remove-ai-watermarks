# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "huggingface-hub>=0.20.0",
#   "numpy",
#   "onnxruntime>=1.24.0",
#   "opencv-python-headless<5",
#   "paddleocr>=3.3.3",
#   "paddlepaddle",
#   "pillow",
# ]
# ///
"""Infer stable source-text lines without modifying an image.

This evaluation-only dry run proposes line annotations for selective text
restoration. Every proposal still needs human verification: stable OCR can lose
punctuation with high confidence. It separately flags lines whose recognition
changes under crop jitter or whose minimum confidence is below the threshold.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import click
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Dogfoods the packaged draft API (remove_ai_watermarks.text_draft); this script
# keeps only the CLI wrapper so the package stays the one home for the logic.
from remove_ai_watermarks.text_draft import (  # noqa: E402
    _detect_line_boxes,
    _recognize,
    choose_language,
    stable_recognition,
)


@click.command()
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--min-score", default=0.85, show_default=True, type=click.FloatRange(0.0, 1.0))
def main(source: Path, out: Path, min_score: float) -> None:
    """Write draft line text for SOURCE; manually verify every proposal."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    from paddleocr import PaddleOCR, TextRecognition

    source_rgb = np.asarray(Image.open(source).convert("RGB"))
    detector = PaddleOCR(
        lang="ch",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    engines = {
        "en": TextRecognition(model_name="en_PP-OCRv5_mobile_rec"),
        "ru": TextRecognition(model_name="eslav_PP-OCRv5_mobile_rec"),
        "ch": TextRecognition(model_name="PP-OCRv5_server_rec"),
    }
    boxes = _detect_line_boxes(detector, source_rgb)
    accepted = []
    rejected = []
    for box in boxes:
        probes = {}
        for language, engine in engines.items():
            script = "cjk" if language == "ch" else "alphabetic"
            probes[language] = _recognize(engine, source_rgb, box, script, 0.1)
        language = choose_language(probes)
        script = "cjk" if language == "ch" else "alphabetic"
        reads = [_recognize(engines[language], source_rgb, box, script, ratio) for ratio in (0.08, 0.12, 0.2)]
        text = stable_recognition(reads, min_score)
        result = {
            "box": box,
            "script": script,
            "language": language,
            "reads": [{"text": value, "score": score} for value, score in reads],
        }
        if text is None:
            rejected.append(result)
        else:
            accepted.append({"box": box, "text": text, "script": script, "min_score": min(score for _, score in reads)})
    payload = {"source": source.name, "accepted": accepted, "rejected": rejected}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Accepted %s lines and rejected %s uncertain lines", len(accepted), len(rejected))


if __name__ == "__main__":
    main()
