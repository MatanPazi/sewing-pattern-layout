#!/usr/bin/env python3
"""
outline_patterns_debug.py

Step 1
Read stroked paths from a PDF page and draw them exactly as they are
(no snap, no union, no extension). Saves a PNG + a short text summary
to verify the raw geometry before any later processing.

Step 2
Perform directional extensions to verify overlapping segments to allow unary_union to work well.
Saves a PNG + a short text summary to verify correct extension direction + length.

How to run:
python outline_patterns_debug.py --pdf /path/to/your.pdf --pages 0
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from collections import defaultdict

import fitz
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Helpers
def dist(a, b):
    """Euclidean distance between two (x, y) points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _unit_dir(a, b):
    """Unit vector from a → b. Returns (0,0) if degenerate."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy)
    if L < 1e-12:
        return (0.0, 0.0)
    return (dx / L, dy / L)

def extend_polyline_ends(segments, extension: float):
    """
    Extend only the two ends of a polyline along their local direction.

    segments: ordered list of ((x1,y1),(x2,y2)) forming one path.
    extension: how far to push each free end (PDF units).

    Why only the ends: internal samples of a Bezier are already connected;
    only the path terminals need to bridge dash gaps / corner gaps.
    """
    if not segments or extension <= 0:
        return list(segments)

    out = [list(s) for s in segments]  # mutable copies

    # --- start of path: extend backward from first segment ---
    a, b = out[0]
    ux, uy = _unit_dir(b, a)  # direction pointing outward at start
    if ux != 0.0 or uy != 0.0:
        out[0] = ((a[0] + ux * extension, a[1] + uy * extension), b)

    # --- end of path: extend forward from last segment ---
    a, b = out[-1]
    ux, uy = _unit_dir(a, b)  # direction pointing outward at end
    if ux != 0.0 or uy != 0.0:
        out[-1] = (a, (b[0] + ux * extension, b[1] + uy * extension))

    return [(tuple(p), tuple(q)) for p, q in out]

def path_to_segments(path):
    """
    Convert one PyMuPDF drawing path into straight segments.

    Beziers are sampled so curves become polylines. Rectangles and
    quads are expanded to their four edges. This is the only place
    PDF operators are interpreted.
    """
    segments = []
    for item in path.get("items", []):
        op = item[0]
        if op == "l":
            p1, p2 = item[1], item[2]
            segments.append(((p1.x, p1.y), (p2.x, p2.y)))
        elif op == "re":
            r = item[1]
            segments.extend([
                ((r.x0, r.y0), (r.x1, r.y0)),
                ((r.x1, r.y0), (r.x1, r.y1)),
                ((r.x1, r.y1), (r.x0, r.y1)),
                ((r.x0, r.y1), (r.x0, r.y0)),
            ])
        elif op == "c":
            p0, p1, p2, p3 = item[1], item[2], item[3], item[4]
            ts = np.linspace(0, 1, 8)
            pts = []
            for t in ts:
                x = (
                    (1 - t) ** 3 * p0.x
                    + 3 * (1 - t) ** 2 * t * p1.x
                    + 3 * (1 - t) * t ** 2 * p2.x
                    + t ** 3 * p3.x
                )
                y = (
                    (1 - t) ** 3 * p0.y
                    + 3 * (1 - t) ** 2 * t * p1.y
                    + 3 * (1 - t) * t ** 2 * p2.y
                    + t ** 3 * p3.y
                )
                pts.append((x, y))
            for a, b in zip(pts[:-1], pts[1:]):
                segments.append((a, b))
        elif op == "qu":
            q = item[1]
            segments.extend([
                ((q.ul.x, q.ul.y), (q.ur.x, q.ur.y)),
                ((q.ur.x, q.ur.y), (q.lr.x, q.lr.y)),
                ((q.lr.x, q.lr.y), (q.ll.x, q.ll.y)),
                ((q.ll.x, q.ll.y), (q.ul.x, q.ul.y)),
            ])
    return segments


def color_key(c):
    """Quantize a PDF colour to a hashable RGB tuple (or None)."""
    if not c:
        return None
    return tuple(round(float(x), 3) for x in c[:3])


def segments_bbox(segments):
    """Axis-aligned bounding box of a list of segments."""
    if not segments:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p[0] for s in segments for p in s]
    ys = [p[1] for s in segments for p in s]
    return (min(xs), min(ys), max(xs), max(ys))


def step1_read_and_draw(page, out_dir: Path, stem: str, min_segment_length: float = 0.5):
    """
    Step 1: extract every stroked path, convert to segments, save debug
    PNG + text summary. No geometric processing yet.

    Returns
    -------
    records : list of dicts (one per original path)
    all_segments : flat list of ((x1,y1),(x2,y2))
    """
    drawings = page.get_drawings()
    records = []
    all_segments = []

    for idx, path in enumerate(drawings):
        color = path.get("color")
        width = float(path.get("width") or 0.0)
        if color is None and width <= 0:
            continue

        segs = path_to_segments(path)
        if not segs:
            continue

        length = sum(dist(a, b) for a, b in segs)
        if length < min_segment_length:
            continue

        rec = {
            "path_id": idx,
            "color": color_key(color),
            "width": round(width, 2),
            "segments": segs,
            "length": length,
            "style": (color_key(color),),
        }
        records.append(rec)
        all_segments.extend(segs)

    # --- text summary ---
    summary_path = out_dir / f"{stem}_step1_summary.txt"
    bbox = segments_bbox(all_segments)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("STEP 1 – paths as-is\n")
        f.write(f"  paths kept     : {len(records)}\n")
        f.write(f"  segments total : {len(all_segments)}\n")
        f.write(f"  bbox           : {bbox}\n")
        if records:
            lengths = [r["length"] for r in records]
            f.write(f"  path length    : min={min(lengths):.2f}  max={max(lengths):.2f}\n")
            by_colour = defaultdict(int)
            for r in records:
                by_colour[r["color"]] += 1
            f.write(f"  distinct colours: {len(by_colour)}\n")
            for col, n in sorted(by_colour.items(), key=lambda x: -x[1])[:15]:
                f.write(f"    {col}: {n} paths\n")
        f.write("\nPer-path detail (first 40):\n")
        for r in records[:40]:
            f.write(
                f"  path {r['path_id']:4d}  colour={r['color']}  "
                f"width={r['width']}  segs={len(r['segments']):3d}  "
                f"len={r['length']:.1f}\n"
            )
        if len(records) > 40:
            f.write(f"  … and {len(records) - 40} more paths\n")
    print(f"Step 1 summary → {summary_path}")

    # --- PNG ---
    png_path = out_dir / f"{stem}_step1_paths.png"
    _draw_segments_debug(
        all_segments,
        records,
        page_rect=page.rect,
        out_path=png_path,
        title="STEP 1 – paths as-is (no snap / no union)",
    )
    print(f"Step 1 image   → {png_path}")

    return records, all_segments

def step2_extend_ends(records, out_dir: Path, stem: str, extension: float = 12.0):
    """
    Step 2: extend each path's terminal stubs along their direction.

    Does NOT snap endpoints and does NOT extend internal segments.
    Purpose: close small dash/corner gaps so later unary_union can node
    collinear or meeting strokes without a large snap radius.

    Returns
    -------
    extended_records : same structure as Step 1 records, segments updated
    all_segments     : flat list of extended segments
    """
    extended_records = []
    all_segments = []
    n_extended = 0

    for rec in records:
        segs = rec["segments"]
        if not segs:
            continue
        new_segs = extend_polyline_ends(segs, extension)
        if new_segs != segs:
            n_extended += 1
        new_rec = dict(rec)
        new_rec["segments"] = new_segs
        new_rec["length"] = sum(dist(a, b) for a, b in new_segs)
        extended_records.append(new_rec)
        all_segments.extend(new_segs)

    # --- text summary ---
    summary_path = out_dir / f"{stem}_step2_summary.txt"
    bbox = segments_bbox(all_segments)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("STEP 2 – directional end extension\n")
        f.write(f"  extension length : {extension}\n")
        f.write(f"  paths processed  : {len(extended_records)}\n")
        f.write(f"  paths changed    : {n_extended}\n")
        f.write(f"  segments total   : {len(all_segments)}\n")
        f.write(f"  bbox             : {bbox}\n")
    print(f"Step 2 summary → {summary_path}")

    # --- PNG: show original (thin grey) + extended (coloured) ---
    png_path = out_dir / f"{stem}_step2_extended.png"
    _draw_step2_debug(
        original_records=records,
        extended_records=extended_records,
        page_bbox=bbox,
        out_path=png_path,
        extension=extension,
    )
    print(f"Step 2 image   → {png_path}")

    return extended_records, all_segments


def _draw_segments_debug(all_segments, records, page_rect, out_path: Path, title: str):
    """Draw every path in a cycling colour for visual inspection."""
    x0, y0, x1, y1 = page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1
    w = max(x1 - x0, 1.0)
    h = max(y1 - y0, 1.0)
    fig_w = 14
    fig_h = max(6.0, fig_w * h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(x0 - 10, x1 + 10)
    ax.set_ylim(y1 + 10, y0 - 10)  # PDF y-down
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11)

    try:
        cmap = plt.colormaps["tab20"]
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap("tab20")

    for i, rec in enumerate(records):
        segs = rec["segments"]
        if not segs:
            continue
        lc = LineCollection(segs, colors=[cmap(i % 20)], linewidths=1.2, alpha=0.9)
        ax.add_collection(lc)

    ax.text(
        0.01, 0.99,
        f"segments={len(all_segments)}  paths={len(records)}",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=3),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def _draw_step2_debug(original_records, extended_records, page_bbox, out_path, extension):
    """
    Overlay:
      - original segments in light grey
      - extended segments in tab20 colours
    so you can see exactly what the extension added.
    """
    x0, y0, x1, y1 = page_bbox
    # pad a bit more because ends grew
    pad = max(extension * 2, 20.0)
    w = max(x1 - x0, 1.0)
    h = max(y1 - y0, 1.0)
    fig_w = 14
    fig_h = max(6.0, fig_w * h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y1 + pad, y0 - pad)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"STEP 2 – end extension (len={extension})", fontsize=11)

    # originals in grey
    orig_segs = [s for r in original_records for s in r["segments"]]
    if orig_segs:
        lc = LineCollection(orig_segs, colors=["0.75"], linewidths=0.8, alpha=0.7, zorder=1)
        ax.add_collection(lc)

    try:
        cmap = plt.colormaps["tab20"]
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap("tab20")

    for i, rec in enumerate(extended_records):
        segs = rec["segments"]
        if not segs:
            continue
        lc = LineCollection(segs, colors=[cmap(i % 20)], linewidths=1.3, alpha=0.95, zorder=2)
        ax.add_collection(lc)

    ax.text(
        0.01, 0.99,
        f"grey=original  colour=extended  extension={extension}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=3),
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Step 1: read PDF paths as-is")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--pages", type=str, default="0", help="e.g. 0 or 0-2 or 0,2")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--min-seg-length", type=float, default=0.5)
    args = parser.parse_args()

    if not args.pdf.is_file():
        raise SystemExit(f"PDF not found: {args.pdf}")

    out_dir = args.out_dir or (args.pdf.parent / "debug_outline")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.pdf.stem

    doc = fitz.open(args.pdf)

    def parse_pages(spec, doc):
        pages = []
        for part in spec.split(","):
            part = part.strip()
            if "-" in part:
                a, b = map(int, part.split("-"))
                pages.extend(range(a, b + 1))
            else:
                pages.append(int(part))
        return sorted(set(p for p in pages if 0 <= p < len(doc)))

    page_nums = parse_pages(args.pages, doc)
    if not page_nums:
        raise SystemExit("No valid pages selected")

    print(f"PDF     : {args.pdf}")
    print(f"Pages   : {page_nums}")
    print(f"Out dir : {out_dir}")

    for pno in page_nums:
        page = doc[pno]
        page_stem = f"{stem}_p{pno}"
        print(f"\n=== Page {pno} ===")
        # ----- STEP 1 - Read and draw as is -----        
        records, all_segments = step1_read_and_draw(
            page,
            out_dir=out_dir,
            stem=page_stem,
            min_segment_length=args.min_seg_length,
        )
        print(f"  paths={len(records)}  segments={len(all_segments)}")

        # ----- STEP 2 - Extension -----
        ext_records, ext_segments = step2_extend_ends(
            records,
            out_dir=out_dir,
            stem=page_stem,
            extension=12.0,   # tune from typical dash gap; try 8–15
        )
        print(f"  step2 segments={len(ext_segments)}")        

    doc.close()
    print("\nDone. Check the Step 1 PNG and summary txt.")


if __name__ == "__main__":
    main()
