#!/usr/bin/env python3
"""
Inspect a PDF for available data (vector paths, text, images, layers)
and visualise the vector content.

Usage:
    python inspect_pdf_data.py path/to/your.pdf
    python inspect_pdf_data.py path/to/your.pdf --page 0
"""

import sys
import argparse
from pathlib import Path
from collections import Counter

import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.collections import LineCollection
import numpy as np


def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def inspect_page(page: fitz.Page, page_num: int, out_dir: Path):
    print_header(f"PAGE {page_num}  ({page.rect.width:.1f} × {page.rect.height:.1f} pts)")

    # ------------------------------------------------------------------
    # 1. Vector drawings (the important part for the "vector method")
    # ------------------------------------------------------------------
    drawings = page.get_drawings()
    print(f"\nVector drawings (paths): {len(drawings)}")

    if drawings:
        item_types = Counter()
        has_fill = has_stroke = 0
        colours = set()
        widths = set()
        total_points = 0

        for path in drawings:
            if path.get("fill"):
                has_fill += 1
                fill = path["fill"]
                colours.add(tuple(fill[:3]) if fill else None)

            color = path.get("color")
            width = path.get("width")                
            if color is not None or (width is not None and width > 0):
                has_stroke += 1
                if color:
                    colours.add(tuple(color[:3]))
            if width is not None:
                widths.add(round(width, 3))

            for item in path.get("items", []):
                op = item[0]
                item_types[op] += 1
                # rough point count
                if op == "l":
                    total_points += 2
                elif op == "c":
                    total_points += 4
                elif op == "re":
                    total_points += 4
                elif op in ("qu", "v", "y"):
                    total_points += 3

        print(f"  • Paths with fill      : {has_fill}")
        print(f"  • Paths with stroke    : {has_stroke}")
        print(f"  • Drawing operators    : {dict(item_types)}")
        print(f"  • Approx. total points : {total_points}")
        print(f"  • Distinct colours     : {len(colours)}")
        print(f"  • Line widths          : {sorted(widths)[:10]}{'...' if len(widths)>10 else ''}")

        # Show a few example paths
        print("\n  Example path (first one):")
        p0 = drawings[0]
        print(f"    type/close : {p0.get('type')} / closePath={p0.get('closePath')}")
        print(f"    color/fill : {p0.get('color')} / {p0.get('fill')}")
        print(f"    width      : {p0.get('width')}")
        print(f"    rect       : {p0.get('rect')}")
        print(f"    items (first 5):")
        for item in p0.get("items", [])[:5]:
            print(f"      {item}")
        if len(p0.get("items", [])) > 5:
            print(f"      ... (+{len(p0['items'])-5} more)")

    # ------------------------------------------------------------------
    # 2. Text
    # ------------------------------------------------------------------
    text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    blocks = text_dict.get("blocks", [])
    text_blocks = [b for b in blocks if b.get("type") == 0]
    print(f"\nText blocks: {len(text_blocks)}")
    if text_blocks:
        sample = ""
        for b in text_blocks[:3]:
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    sample += span.get("text", "") + " "
        print(f"  Sample text: {sample[:120].strip()}...")

    # ------------------------------------------------------------------
    # 3. Images
    # ------------------------------------------------------------------
    images = page.get_images(full=True)
    print(f"\nEmbedded images: {len(images)}")
    for i, img in enumerate(images[:5]):
        print(f"  [{i}] xref={img[0]}, size≈{img[2]}×{img[3]}")
    if len(images) > 5:
        print(f"  ... and {len(images)-5} more")

    # ------------------------------------------------------------------
    # 4. Optional Content / Layers (OCGs)
    # ------------------------------------------------------------------
    try:
        ocgs = page.parent.get_ocgs()
        print(f"\nDocument layers (OCGs): {len(ocgs)}")
        for xref, info in list(ocgs.items())[:8]:
            print(f"  • {info.get('name', '?')} (on={info.get('on')})")
        if len(ocgs) > 8:
            print(f"  ... and {len(ocgs)-8} more")
    except Exception:
        print("\nLayers: none or could not read")

    # ------------------------------------------------------------------
    # 5. Visual output
    # ------------------------------------------------------------------
    # A) Rendered page (raster preview)
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    raster_path = out_dir / f"page{page_num:02d}_raster.png"
    pix.save(raster_path)
    print(f"\nSaved raster preview → {raster_path}")

    # B) Pure vector plot with matplotlib
    if drawings:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12 * page.rect.height / page.rect.width))
        ax.set_xlim(page.rect.x0, page.rect.x1)
        ax.set_ylim(page.rect.y1, page.rect.y0)  # PDF y grows upward, matplotlib downward
        ax.set_aspect("equal")
        ax.set_title(f"Vector paths only – page {page_num} ({len(drawings)} paths)")
        ax.axis("off")

        for path in drawings:
            color = path.get("color") or (0, 0, 0)
            fill = path.get("fill")
            lw = path.get("width") or 0.5

            # Collect line segments for this path
            segments = []
            for item in path.get("items", []):
                op = item[0]
                if op == "l":                      # line
                    p1, p2 = item[1], item[2]
                    segments.append([(p1.x, p1.y), (p2.x, p2.y)])
                elif op == "re":                   # rectangle
                    r = item[1]
                    segments.append([(r.x0, r.y0), (r.x1, r.y0)])
                    segments.append([(r.x1, r.y0), (r.x1, r.y1)])
                    segments.append([(r.x1, r.y1), (r.x0, r.y1)])
                    segments.append([(r.x0, r.y1), (r.x0, r.y0)])
                elif op == "c":                    # cubic Bézier – approximate with polyline
                    p0, p1, p2, p3 = item[1], item[2], item[3], item[4]
                    # simple sampling
                    ts = np.linspace(0, 1, 12)
                    pts = []
                    for t in ts:
                        x = (1-t)**3*p0.x + 3*(1-t)**2*t*p1.x + 3*(1-t)*t**2*p2.x + t**3*p3.x
                        y = (1-t)**3*p0.y + 3*(1-t)**2*t*p1.y + 3*(1-t)*t**2*p2.y + t**3*p3.y
                        pts.append((x, y))
                    for a, b in zip(pts[:-1], pts[1:]):
                        segments.append([a, b])
                # (other operators qu/v/y can be added similarly if needed)

            if segments:
                lc = LineCollection(segments, colors=[color], linewidths=lw, alpha=0.85)
                ax.add_collection(lc)

        vector_path = out_dir / f"page{page_num:02d}_vectors.png"
        plt.tight_layout()
        plt.savefig(vector_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved vector plot   → {vector_path}")

    return len(drawings)


def main():
    parser = argparse.ArgumentParser(description="Inspect PDF vector & other data")
    parser.add_argument("pdf", type=Path, help="Input PDF file")
    parser.add_argument("--page", type=int, default=None, help="Only inspect this page (0-based)")
    parser.add_argument("--outdir", type=Path, default=None, help="Where to save images")
    args = parser.parse_args()

    pdf_path = args.pdf
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    out_dir = args.outdir or pdf_path.parent / f"{pdf_path.stem}_inspect"
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    print_header(f"PDF: {pdf_path.name}")
    print(f"Pages: {len(doc)}")
    print(f"Output folder: {out_dir}")

    pages_to_do = [args.page] if args.page is not None else range(len(doc))

    total_paths = 0
    for pno in pages_to_do:
        if pno < 0 or pno >= len(doc):
            print(f"Page {pno} out of range")
            continue
        total_paths += inspect_page(doc[pno], pno, out_dir)

    print_header("SUMMARY")
    print(f"Total vector paths found: {total_paths}")
    if total_paths > 50:
        print("→ Looks like a rich vector PDF – vector method is promising.")
    elif total_paths > 0:
        print("→ Some vector data present; still worth trying the vector route.")
    else:
        print("→ Almost no vector paths. Raster (OpenCV) is probably the better choice.")

    print(f"\nOpen the images in:\n  {out_dir}")
    doc.close()


if __name__ == "__main__":
    main()