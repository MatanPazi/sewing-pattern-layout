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


# ------------------------------------------------------------------
# 5. Visual output + full operator dump (debug version)
# ------------------------------------------------------------------
def dump_and_plot_drawings(page, drawings, page_num, out_dir):
    log_path = out_dir / f"page{page_num:02d}_drawings.txt"
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"=== PAGE {page_num} – {len(drawings)} paths ===\n\n")

        fig, ax = plt.subplots(1, 1, figsize=(14, 14 * page.rect.height / page.rect.width))
        ax.set_xlim(page.rect.x0, page.rect.x1)
        ax.set_ylim(page.rect.y1, page.rect.y0)  # PDF → matplotlib y-flip
        ax.set_aspect("equal")
        ax.set_title(f"Vector paths (numbered) – page {page_num}")
        ax.axis("off")

        global_op_id = 0          # unique ID for every single operator
        path_id = 0

        for path in drawings:
            path_id += 1
            color = path.get("color") or (0, 0, 0)
            fill  = path.get("fill")
            lw    = path.get("width") or 0.8
            close = path.get("closePath", False)
            ptype = path.get("type", "?")

            header = (f"PATH {path_id:03d}  type={ptype}  closePath={close}  "
                      f"color={color}  fill={fill}  width={lw}  rect={path.get('rect')}")
            print(header)
            log.write(header + "\n")

            segments = []
            first_pt = None
            last_pt  = None

            for item in path.get("items", []):
                global_op_id += 1
                op = item[0]
                pts = item[1:]

                # ---- log the operator ----
                desc = f"  OP {global_op_id:04d}  {op}  {pts}"
                print(desc)
                log.write(desc + "\n")

                # ---- turn into line segments ----
                if op == "l":
                    p1, p2 = pts
                    segments.append([(p1.x, p1.y), (p2.x, p2.y)])
                    if first_pt is None:
                        first_pt = p1
                    last_pt = p2
                    # label
                    mid = ((p1.x + p2.x)/2, (p1.y + p2.y)/2)
                    ax.text(mid[0], mid[1], str(global_op_id), fontsize=6,
                            color="red", ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.15", fc="yellow", alpha=0.7))

                elif op == "re":
                    r = pts[0]          # Rect
                    segs = [
                        [(r.x0, r.y0), (r.x1, r.y0)],
                        [(r.x1, r.y0), (r.x1, r.y1)],
                        [(r.x1, r.y1), (r.x0, r.y1)],
                        [(r.x0, r.y1), (r.x0, r.y0)],
                    ]
                    segments.extend(segs)
                    first_pt = fitz.Point(r.x0, r.y0)
                    last_pt  = first_pt
                    cx, cy = r.x0 + r.width/2, r.y0 + r.height/2
                    ax.text(cx, cy, str(global_op_id), fontsize=6, color="red",
                            ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.15", fc="yellow", alpha=0.7))

                elif op == "c":                    # cubic Bézier
                    p0, p1, p2, p3 = pts
                    ts = np.linspace(0, 1, 16)
                    curve = []
                    for t in ts:
                        x = (1-t)**3*p0.x + 3*(1-t)**2*t*p1.x + 3*(1-t)*t**2*p2.x + t**3*p3.x
                        y = (1-t)**3*p0.y + 3*(1-t)**2*t*p1.y + 3*(1-t)*t**2*p2.y + t**3*p3.y
                        curve.append((x, y))
                    for a, b in zip(curve[:-1], curve[1:]):
                        segments.append([a, b])
                    if first_pt is None:
                        first_pt = p0
                    last_pt = p3
                    mid = curve[len(curve)//2]
                    ax.text(mid[0], mid[1], str(global_op_id), fontsize=6, color="red",
                            ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.15", fc="yellow", alpha=0.7))

                elif op == "qu":                   # ★ previously missing
                    q = pts[0]                    # Quad
                    segs = [
                        [(q.ul.x, q.ul.y), (q.ur.x, q.ur.y)],
                        [(q.ur.x, q.ur.y), (q.lr.x, q.lr.y)],
                        [(q.lr.x, q.lr.y), (q.ll.x, q.ll.y)],
                        [(q.ll.x, q.ll.y), (q.ul.x, q.ul.y)],
                    ]
                    segments.extend(segs)
                    first_pt = q.ul
                    last_pt  = q.ul
                    cx = (q.ul.x + q.lr.x)/2
                    cy = (q.ul.y + q.lr.y)/2
                    ax.text(cx, cy, str(global_op_id), fontsize=6, color="red",
                            ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.15", fc="yellow", alpha=0.7))

                else:
                    msg = f"    *** UNHANDLED OPERATOR: {op} ***"
                    print(msg)
                    log.write(msg + "\n")

            # ---- implicit closePath ----
            if close and first_pt is not None and last_pt is not None:
                if abs(first_pt.x - last_pt.x) > 1e-3 or abs(first_pt.y - last_pt.y) > 1e-3:
                    segments.append([(last_pt.x, last_pt.y), (first_pt.x, first_pt.y)])
                    log.write(f"  (implicit closePath line added)\n")

            if segments:
                lc = LineCollection(segments, colors=[color], linewidths=max(lw, 0.4), alpha=0.9)
                ax.add_collection(lc)

            log.write("\n")
            print()

        vector_path = out_dir / f"page{page_num:02d}_vectors_numbered.png"
        plt.tight_layout()
        plt.savefig(vector_path, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"Saved numbered vector plot → {vector_path}")
        print(f"Full operator dump         → {log_path}")

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
        dump_and_plot_drawings(page, drawings, page_num, out_dir)

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
