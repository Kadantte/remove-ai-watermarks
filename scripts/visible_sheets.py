"""Build BLIND contact sheets for hand-labeling relaxation additions.

Crops are centered on the DETECTED REGION (not the corner), padded by ~0.9x the mark
size, and resized to 240px with INTER_NEAREST -- a downscaled preview destroys a faint
mark, so nothing here may smooth. The manifest is written to a separate file that must
NOT be read until labeling is finished.

Each sheet mixes three strata in shuffled order:
  add    - the relaxation additions whose precision we are measuring
  pos    - strict-consistent detections (a mark is really there): labeler sensitivity
  clean  - verified-clean negatives (no mark can be there): labeler specificity
The two control strata are what make a low measured precision trustworthy.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from remove_ai_watermarks import watermark_registry as wr
from remove_ai_watermarks.image_io import imread

CELL = 240
COLS, ROWS = 6, 3
PER = COLS * ROWS


def region_for(path: str, key: str) -> tuple[int, int, int, int] | None:
    """Re-detect to recover the mark bbox (Candidate carries no region)."""
    img = imread(path)
    if img is None:
        return None
    mark = next(m for m in wr._REGISTRY if m.key == key)
    d = mark.detect(img, provenance=True)
    return d.region if d.region else None


def crop(path: str, region: tuple[int, int, int, int] | None, pad_factor: float = 0.9) -> NDArray[Any] | None:
    img = imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if region:
        x, y, rw, rh = region
    else:
        return None
    px, py = int(rw * pad_factor), int(rh * pad_factor)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(w, x + rw + px), min(h, y + rh + py)
    c = img[y0:y1, x0:x1]
    if c.size == 0:
        return None
    s = CELL / max(c.shape[0], c.shape[1])
    return cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))), interpolation=cv2.INTER_NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON candidate list")
    parser.add_argument("output", type=Path, help="Directory for blinded contact sheets")
    args = parser.parse_args()
    with args.input.open() as fh:
        items = json.load(fh)  # [{uid,path,key,stratum,conf}]
    outdir = args.output
    outdir.mkdir(parents=True, exist_ok=True)
    random.Random(1234).shuffle(items)  # noqa: S311 -- sheet ordering, not cryptography

    manifest = []
    cells = []
    for it in items:
        reg = region_for(it["path"], it["key"])
        c = crop(it["path"], reg)
        if c is None:
            continue
        cells.append((it, c))

    for si in range(0, len(cells), PER):
        chunk = cells[si : si + PER]
        sheet = np.full((ROWS * (CELL + 26), COLS * (CELL + 8), 3), 40, np.uint8)
        for i, (it, c) in enumerate(chunk):
            r, col = divmod(i, COLS)
            y0 = r * (CELL + 26) + 22
            x0 = col * (CELL + 8) + 4
            sheet[y0 : y0 + c.shape[0], x0 : x0 + c.shape[1]] = c
            label = f"{si + i:04d}"  # index ONLY -- no stratum, no confidence
            cv2.putText(sheet, label, (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            manifest.append({"idx": si + i, **it})
        cv2.imwrite(str(outdir / f"sheet_{si // PER:03d}.png"), sheet)

    with open(outdir / "MANIFEST_DO_NOT_OPEN.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    print(f"sheets={(len(cells) + PER - 1) // PER} cells={len(cells)} -> {outdir}")


if __name__ == "__main__":
    main()
