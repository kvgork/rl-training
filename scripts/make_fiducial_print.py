"""Print file of the H8 table fiducial: ArUco DICT_4X4_50 id 0, 50 mm (plan §10 2026-10-17 "H8 software").

Usage:
    pixi run python scripts/make_fiducial_print.py [--out runs/hw/fiducial/aruco_4x4_50_id0_50mm_A4.pdf]

An A4 page at 24 px/mm (609.6 dpi: each of the 6 x 6 cells is exactly 200 px), the marker centred on the page with a
thin grey cut square 10 mm outside it, crosshair ticks on its centre lines (to align the sides with the base x / y
axes) and a label. Print at 100 % ("actual size", no fit-to-page), check the black square measures 50.0 mm with a
ruler, glue it flat with its centre at base (0.09, 0.05) and its TOP edge (the label side) towards +x, away from the
base (``perception/fiducial.py``). Also writes the marker alone as ``<out stem>.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

PX_PER_MM = 24  # 609.6 dpi
A4_MM = (210, 297)
DEFAULT_OUT = Path("runs/hw/fiducial/aruco_4x4_50_id0_50mm_A4.pdf")


def page(marker_id: int = 0, size_mm: float = 50.0) -> Image.Image:
    from so101_rl.perception.fiducial import DICTIONARY

    w, h = (round(v * PX_PER_MM) for v in A4_MM)
    m_px = round(size_mm * PX_PER_MM)
    if m_px % 6:
        raise ValueError(f"{size_mm} mm is not a whole number of pixels per cell at {PX_PER_MM} px/mm")
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY)), marker_id, m_px)
    img = Image.new("L", (w, h), 255)
    x0, y0 = (w - m_px) // 2, (h - m_px) // 2
    img.paste(Image.fromarray(marker), (x0, y0))
    d = ImageDraw.Draw(img)
    cut = 10 * PX_PER_MM
    d.rectangle([x0 - cut, y0 - cut, x0 + m_px + cut, y0 + m_px + cut], outline=170, width=3)
    cx, cy, tick = w // 2, h // 2, 8 * PX_PER_MM
    for a, b in (((cx, y0 - cut - tick), (cx, y0 - cut)), ((cx, y0 + m_px + cut), (cx, y0 + m_px + cut + tick)),
                 ((x0 - cut - tick, cy), (x0 - cut, cy)), ((x0 + m_px + cut, cy), (x0 + m_px + cut + tick, cy))):  # fmt: skip
        d.line([a, b], fill=0, width=4)
    lines = (f"ArUco {DICTIONARY} id {marker_id}, {size_mm:g} mm black square. Print at 100 % (actual size).",
             "Glue with this TOP edge towards +x (away from the arm base), centre at base (0.09, 0.05).")  # fmt: skip
    font = 3 * PX_PER_MM  # 3 mm text
    for i, text in enumerate(lines):
        d.text((20 * PX_PER_MM, y0 - cut - tick - (len(lines) - i) * 5 * PX_PER_MM), text, fill=0, font_size=font)
    return img


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--marker-id", type=int, default=0)
    ap.add_argument("--size-mm", type=float, default=50.0)
    a = ap.parse_args(argv)
    img = page(a.marker_id, a.size_mm)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(a.out, "PDF", resolution=PX_PER_MM * 25.4)
    m_px = round(a.size_mm * PX_PER_MM)
    marker = np.asarray(img)[(img.height - m_px) // 2 :][:m_px, (img.width - m_px) // 2 :][:, :m_px]
    Image.fromarray(marker).save(a.out.with_suffix(".png"), dpi=(PX_PER_MM * 25.4,) * 2)
    print(f"wrote {a.out} (A4, {PX_PER_MM * 25.4:g} dpi, marker {a.size_mm:g} mm = {m_px} px) and {a.out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
