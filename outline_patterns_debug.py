"""
#!/usr/bin/env python3
outline_patterns_debug.py

How to run:
python outline_patterns_debug.py --pattern-pdf PAT.pdf --lines-pdf LINES.pdf --pattern-pages 0-3 --lines-pages 0-3
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

import matplotlib as mpl
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Point
from shapely.ops import polygonize, unary_union

import json

# ------------------------------------------------------------------
# Multi-page assembly helpers
# ------------------------------------------------------------------

class AssembledPage:
    """Minimal page-like object so the existing detect_* functions keep working."""
    def __init__(self, paths, rect):
        self._paths = paths
        self.rect = rect
        self.mediabox = rect
        self.cropbox = rect
        self.rotation = 0

    def get_drawings(self):
        return self._paths

def _find_large_rectangles(page, min_area_ratio=0.55, min_side_ratio=0.60):
    """
    Return list of candidate content rectangles on a page.
    Prefers a single 're' operator; also accepts 4-line rectangles.
    """
    drawings = page.get_drawings()
    page_area = page.rect.width * page.rect.height
    candidates = []

    for path in drawings:
        items = path.get("items", [])
        if not items:
            continue

        # Case 1: pure rectangle operator
        if len(items) == 1 and items[0][0] == "re":
            r = items[0][1]
            w, h = r.width, r.height
            if (w * h >= min_area_ratio * page_area and
                w >= min_side_ratio * page.rect.width and
                h >= min_side_ratio * page.rect.height):
                candidates.append(fitz.Rect(r))
            continue

        # Case 2: four long axis-aligned lines that form a rectangle
        # (simplified but works for most pattern borders)
        segs = path_to_segments(path)
        if len(segs) < 4:
            continue
        # ... (you can expand this later if needed)

    return candidates


def _detect_content_rect(doc, page_numbers):
    """
    Look for a large rectangle that exists on *every* page of the given list.
    Returns the median rectangle (in page coordinates) or None.
    """
    if not page_numbers:
        return None

    all_rects = []
    for pno in page_numbers:
        page = doc[pno]
        rects = _find_large_rectangles(page)
        if not rects:
            return None          # must exist on every page
        # take the largest one on this page
        largest = max(rects, key=lambda r: r.width * r.height)
        all_rects.append(largest)

    # All pages have a large rect → check they are similar
    widths  = [r.width  for r in all_rects]
    heights = [r.height for r in all_rects]
    if (max(widths)  - min(widths)  > 8.0 or
        max(heights) - min(heights) > 8.0):
        return None

    # Return a representative rectangle (median position + size)
    med_x0 = float(np.median([r.x0 for r in all_rects]))
    med_y0 = float(np.median([r.y0 for r in all_rects]))
    med_w  = float(np.median(widths))
    med_h  = float(np.median(heights))
    return fitz.Rect(med_x0, med_y0, med_x0 + med_w, med_y0 + med_h)


def assemble_pages(pattern_doc, pattern_pages,
                   instr_doc, instr_pages,
                   overlap=0.0):
    """
    Assemble multi-page pattern into a single global coordinate system.
    Returns a dict containing the transformed paths and a correctly sized canvas.
    """
    import math
    import numpy as np

    if not pattern_pages:
        raise ValueError("No pattern pages given")

    # ------------------------------------------------------------------
    # 1. Detect content rectangle from the instruction layer
    # ------------------------------------------------------------------
    content_rect = _detect_content_rect(instr_doc, instr_pages)

    if content_rect is not None:
        tile_w = content_rect.width
        tile_h = content_rect.height
        print(f"Content rectangle detected: {tile_w:.1f} × {tile_h:.1f}")
    else:
        sample = pattern_doc[pattern_pages[0]]
        tile_w = sample.rect.width
        tile_h = sample.rect.height
        content_rect = None
        print("No consistent content rectangle – using full page size")

    # ------------------------------------------------------------------
    # 2. Intelligent layout decision based on boundary overflows
    # ------------------------------------------------------------------
    n = len(pattern_pages)
    horizontal_overflow = False
    vertical_overflow = False

    ref_x0 = content_rect.x0 if content_rect else 0
    ref_y0 = content_rect.y0 if content_rect else 0
    ref_x1 = content_rect.x1 if content_rect else tile_w
    ref_y1 = content_rect.y1 if content_rect else tile_h

    # Scan drawings to see which boundary is crossed
    for pno in pattern_pages:
        page = pattern_doc[pno]
        for path in page.get_drawings():
            for item in path.get("items", []):
                op = item[0]
                pts = []
                if op == "l":
                    pts = [item[1], item[2]]
                elif op == "c":
                    pts = item[1:]
                elif op == "re":
                    r = item[1]
                    pts = [fitz.Point(r.x0, r.y0), fitz.Point(r.x1, r.y1)]
                elif op == "qu":
                    q = item[1]
                    pts = [q.ul, q.ur, q.lr, q.ll]

                for pt in pts:
                    p = fitz.Point(pt)
                    if p.x > ref_x1 + 1.0 or p.x < ref_x0 - 1.0:
                        horizontal_overflow = True
                    if p.y > ref_y1 + 1.0 or p.y < ref_y0 - 1.0:
                        vertical_overflow = True

    # Determine layout orientation based on overflow direction
    if horizontal_overflow and not vertical_overflow:
        cols, rows = n, 1
        print("Detected horizontal extension: arranging pages in a single row")
    elif vertical_overflow and not horizontal_overflow:
        cols, rows = 1, n
        print("Detected vertical extension: arranging pages in a single column")
    else:
        # Fallback based on page orientation if overflow is ambiguous or absent
        sample = pattern_doc[pattern_pages[0]]
        if sample.rect.width >= sample.rect.height:
            cols, rows = n, 1
            print("No clear overflow detected, defaulting to horizontal layout (landscape)")
        else:
            cols, rows = 1, n
            print("No clear overflow detected, defaulting to vertical layout (portrait)")

    print(f"Using grid {rows}×{cols}")

    # ------------------------------------------------------------------
    # 3. Place every path and collect real bounding box
    # ------------------------------------------------------------------
    global_paths = []
    page_transforms = []
    all_x = []
    all_y = []

    for idx, pno in enumerate(pattern_pages):
        page = pattern_doc[pno]
        row = idx // cols
        col = idx % cols

        # Base translation – place tiles next to each other
        tx = col * (tile_w - overlap)
        ty = row * (tile_h - overlap)

        # Optional origin shift only if we have a content rect
        if content_rect is not None:
            tx -= content_rect.x0
            ty -= content_rect.y0

        mat = fitz.Matrix(1, 0, 0, 1, tx, ty)
        page_transforms.append((pno, mat))

        for path in page.get_drawings():
            new_items = []
            xs = []
            ys = []

            for item in path.get("items", []):
                op = item[0]
                if op == "l":
                    p1 = fitz.Point(item[1]) * mat
                    p2 = fitz.Point(item[2]) * mat
                    new_items.append(("l", p1, p2))
                    xs.extend([p1.x, p2.x])
                    ys.extend([p1.y, p2.y])
                elif op == "c":
                    pts = [fitz.Point(p) * mat for p in item[1:]]
                    new_items.append(("c", *pts))
                    for p in pts:
                        xs.append(p.x)
                        ys.append(p.y)
                elif op == "re":
                    r = fitz.Rect(item[1]) * mat
                    new_items.append(("re", r, item[2] if len(item) > 2 else 1))
                    xs.extend([r.x0, r.x1])
                    ys.extend([r.y0, r.y1])
                elif op == "qu":
                    q = item[1]
                    ul = fitz.Point(q.ul) * mat
                    ur = fitz.Point(q.ur) * mat
                    lr = fitz.Point(q.lr) * mat
                    ll = fitz.Point(q.ll) * mat
                    new_items.append(("l", ul, ur))
                    new_items.append(("l", ur, lr))
                    new_items.append(("l", lr, ll))
                    new_items.append(("l", ll, ul))
                    xs.extend([ul.x, ur.x, lr.x, ll.x])
                    ys.extend([ul.y, ur.y, lr.y, ll.y])
                else:
                    new_items.append(item)

            # ---- critical: keep a correct rect for this path ----
            new_path = dict(path)  # preserve width, color, etc.
            new_path["items"] = new_items
            if xs and ys:
                new_path["rect"] = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
            else:
                # fallback – should rarely happen
                new_path["rect"] = fitz.Rect(tx, ty, tx + tile_w, ty + tile_h)

            global_paths.append(new_path)

            # also collect for the overall canvas
            all_x.extend(xs)
            all_y.extend(ys)

    # ------------------------------------------------------------------
    # 4. Real bounding box of everything + padding
    # ------------------------------------------------------------------
    if not all_x:
        global_rect = fitz.Rect(0, 0, tile_w, tile_h)
    else:
        pad = 30.0
        global_rect = fitz.Rect(
            min(all_x) - pad,
            min(all_y) - pad,
            max(all_x) + pad,
            max(all_y) + pad,
        )

    print(f"Final canvas: {global_rect.width:.1f} × {global_rect.height:.1f}")
    return {
        "paths": global_paths,
        "global_rect": global_rect,
        "content_rect": content_rect,
        "tile_w": tile_w,
        "tile_h": tile_h,
        "grid": (cols, rows),
        "page_transforms": page_transforms,
    }

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

def path_to_segments(path, n_bezier=8):
    """
    Convert one PyMuPDF drawing path into straight segments.
    n_bezier: samples per cubic (8 = arrows stay small; 32 = smoother outlines).
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
            ts = np.linspace(0, 1, n_bezier)
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

def step_lines_detect(page, out_dir: Path, stem: str, assembled=None, **kwargs):
    """
    Detect grain/fold lines, then write debug TXT + PNG (same style as steps 1–5).
    """
    lines = detect_special_lines(page, **kwargs)
    n_grain = sum(1 for o in lines if o["type"] == "grain")
    n_fold = sum(1 for o in lines if o["type"] == "fold")

    print(f"  grain={n_grain}  fold={n_fold}  total={len(lines)}")
    for i, obj in enumerate(lines):
        print(
            f"  [{i}] {obj['type']:5s}  score={obj['score']}  "
            f"shaft={obj['shaft']['length']:.1f}  "
            f"segs={obj['shaft']['num_segments']}  "
            f"arrows={len(obj['arrows'])}"
        )

    summary_path = out_dir / f"{stem}_lines_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("LINES – grain / fold\n")
        f.write(f"  grain : {n_grain}\n")
        f.write(f"  fold  : {n_fold}\n")
        f.write(f"  total : {len(lines)}\n")
        for i, obj in enumerate(lines):
            sh = obj["shaft"]
            f.write(
                f"  [{i}] type={obj['type']}  score={obj['score']}  "
                f"shaft_len={sh['length']:.1f}  segs={sh['num_segments']}  "
                f"arrows={len(obj['arrows'])}  path_ids={sh.get('path_ids', [])}\n"
            )
            f.write(f"      endpoints : {sh['endpoints']}\n")
            f.write(f"      shaft segments ({len(sh['segments'])}):\n")
            for s in sh["segments"]:
                f.write(f"        {s[0]} -> {s[1]}\n")
            n_arrow_seg = sum(len(a["segments"]) for a in obj["arrows"])
            f.write(f"      arrow segments ({n_arrow_seg}):\n")
            for a in obj["arrows"]:
                for s in a["segments"]:
                    f.write(f"        {s[0]} -> {s[1]}\n")
            if obj.get("crossbars"):
                f.write(f"      crossbars : {len(obj['crossbars'])}\n")
    print(f"LINES summary → {summary_path}")

    png_path = out_dir / f"{stem}_lines.png"
    _draw_lines_debug(page, lines, png_path, assembled=assembled)
    print(f"LINES image   → {png_path}")
    return lines


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

        segs = path_to_segments(path, n_bezier=32)
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

def step2_extend_to_first_hit(
    records,
    out_dir: Path,
    stem: str,
    max_extension: float = 15.0,
    hit_eps: float = 0.1,
    past_hit: float = 0.05,
    max_rounds: int = 5,
    lateral_tol: float = 1.0,
):
    """
    Step 2 – close small gaps by growing path ends, without shooting past each other.

    Each path end that is still free looks forward along its last segment
    (up to max_extension) and takes the nearest of:
      • a crossing with another path's stroke
      • another path end sitting in a narrow tube ahead (dashed / almost collinear)
      • another free end whose own look-ahead meets this one (a corner)

    Two cases:

    1. Two ends see each other (A finds B and B finds A, they face one
       another, and they are close enough to reach).
       That is one join. Both ends are moved to the midpoint of the gap.
       Neither is allowed to keep growing onto the other's original line

    2. One end hits a stroke that is not looking back (a T).
       Only that end is grown to the hit, slightly past it so union can node.

    Repeat for a few rounds so C can meet A after A has moved.
    Ends that already joined are not grown again.
    """
    from shapely.geometry import LineString, Point

    def ray_ray_intersection(o1, d1, o2, d2):
        det = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(det) < 1e-12:
            return None
        ox, oy = o2[0] - o1[0], o2[1] - o1[1]
        t = (ox * d2[1] - oy * d2[0]) / det
        s = (ox * d1[1] - oy * d1[0]) / det
        if t <= hit_eps or s <= hit_eps:
            return None
        if t > max_extension or s > max_extension:
            return None
        return (t, s, (o1[0] + d1[0] * t, o1[1] + d1[1] * t))

    def tube_hit(origin, direction, point):
        vx = point[0] - origin[0]
        vy = point[1] - origin[1]
        t = vx * direction[0] + vy * direction[1]
        if t <= hit_eps or t > max_extension:
            return None
        lateral = abs(vx * direction[1] - vy * direction[0])
        if lateral > lateral_tol:
            return None
        return t

    def cell_key(x, y, cell):
        return (int(x // cell), int(y // cell))

    def nearby_keys(x, y, cell, radius_cells):
        cx, cy = int(x // cell), int(y // cell)
        return [(cx + dx, cy + dy)
                for dx in range(-radius_cells, radius_cells + 1)
                for dy in range(-radius_cells, radius_cells + 1)]

    def other_paths(cand, self_key):
        """Path indices this candidate connected to, excluding self."""
        if cand[0] in ("seg", "prox"):
            return {cand[4]}
        k1, k2 = cand[2], cand[4]
        return {k1[0], k2[0]} - {self_key[0]}

    work = [[list(s) for s in rec["segments"]] for rec in records]
    applied = set()
    n_seg = n_ray = n_prox = n_mutual = 0
    extension_lengths = []
    round_log = []
    hit_points = []

    cell = max(max_extension, 4.0)
    radius_cells = 1

    def set_terminal(key, pt):
        pi, is_start = key
        if not work[pi]:
            return
        if is_start:
            work[pi][0][0] = pt
        else:
            work[pi][-1][1] = pt

    for rnd in range(max_rounds):
        terminals = {}
        for pi, segs in enumerate(work):
            if not segs:
                continue
            a0, b0 = segs[0]
            d0 = _unit_dir(b0, a0)
            if d0 != (0.0, 0.0) and (pi, True) not in applied:
                terminals[(pi, True)] = (tuple(a0), d0)
            a1, b1 = segs[-1]
            d1 = _unit_dir(a1, b1)
            if d1 != (0.0, 0.0) and (pi, False) not in applied:
                terminals[(pi, False)] = (tuple(b1), d1)

        if not terminals:
            round_log.append(f"  round {rnd+1}: no free terminals")
            break

        seg_grid = defaultdict(list)
        for pi, segs in enumerate(work):
            for a, b in segs:
                if dist(a, b) <= 1e-12:
                    continue
                ls = LineString([a, b])
                x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                x1, y1 = max(a[0], b[0]), max(a[1], b[1])
                for cx in range(int(x0 // cell), int(x1 // cell) + 1):
                    for cy in range(int(y0 // cell), int(y1 // cell) + 1):
                        seg_grid[(cx, cy)].append((pi, ls))

        term_grid = defaultdict(list)
        seen_term_pt = set()
        for pi, segs in enumerate(work):
            if not segs:
                continue
            for pt in (tuple(segs[0][0]), tuple(segs[-1][1])):
                sp = (pi, pt)
                if sp in seen_term_pt:
                    continue
                seen_term_pt.add(sp)
                term_grid[cell_key(pt[0], pt[1], cell)].append(sp)

        candidates = []

        for key, (origin, direction) in terminals.items():
            pi = key[0]
            tip = (
                origin[0] + direction[0] * max_extension,
                origin[1] + direction[1] * max_extension,
            )
            ray = LineString([origin, tip])
            rx0, ry0 = min(origin[0], tip[0]), min(origin[1], tip[1])
            rx1, ry1 = max(origin[0], tip[0]), max(origin[1], tip[1])
            cx0, cy0 = int(rx0 // cell), int(ry0 // cell)
            cx1, cy1 = int(rx1 // cell), int(ry1 // cell)

            best_t, best_pt, best_other = None, None, None
            tested_seg = set()
            for cx in range(cx0 - radius_cells, cx1 + radius_cells + 1):
                for cy in range(cy0 - radius_cells, cy1 + radius_cells + 1):
                    for other_pi, other_ls in seg_grid.get((cx, cy), []):
                        if other_pi == pi:
                            continue
                        sid = (other_pi, id(other_ls))
                        if sid in tested_seg:
                            continue
                        tested_seg.add(sid)
                        try:
                            inter = ray.intersection(other_ls)
                        except Exception:
                            continue
                        if inter.is_empty:
                            continue
                        points = []
                        if inter.geom_type == "Point":
                            points = [inter]
                        elif inter.geom_type == "MultiPoint":
                            points = list(inter.geoms)
                        elif inter.geom_type == "LineString":
                            # collinear overlap: nearest point on the overlap, not the far end
                            c = list(inter.coords)
                            points = [Point(c[0]), Point(c[-1])]
                        elif inter.geom_type == "GeometryCollection":
                            points = [g for g in inter.geoms if g.geom_type == "Point"]
                        for p in points:
                            t = dist(origin, (p.x, p.y))
                            if t <= hit_eps or t > max_extension:
                                continue
                            if (p.x - origin[0]) * direction[0] + (p.y - origin[1]) * direction[1] < 0:
                                continue
                            if best_t is None or t < best_t:
                                best_t = t
                                best_pt = (
                                    origin[0] + direction[0] * t,
                                    origin[1] + direction[1] * t,
                                )
                                best_other = other_pi
            if best_t is not None:
                candidates.append(("seg", best_t, key, best_pt, best_other))

            best_pt_t, best_pt_pt, best_pt_other = None, None, None
            tested_pt = set()
            for ck in nearby_keys(origin[0], origin[1], cell, radius_cells + 1):
                for other_pi, pt in term_grid.get(ck, []):
                    if other_pi == pi:
                        continue
                    if (other_pi, pt) in tested_pt:
                        continue
                    tested_pt.add((other_pi, pt))
                    t = tube_hit(origin, direction, pt)
                    if t is None:
                        continue
                    if best_pt_t is None or t < best_pt_t:
                        best_pt_t = t
                        best_pt_pt = (
                            origin[0] + direction[0] * t,
                            origin[1] + direction[1] * t,
                        )
                        best_pt_other = other_pi
            if best_pt_t is not None:
                candidates.append(("prox", best_pt_t, key, best_pt_pt, best_pt_other))

        keys = list(terminals.keys())
        seen = set()
        for i, k1 in enumerate(keys):
            o1, d1 = terminals[k1]
            near_keys = set(nearby_keys(o1[0], o1[1], cell, radius_cells + 1))
            for k2 in keys[i + 1:]:
                if k2[0] == k1[0]:
                    continue
                o2, d2 = terminals[k2]
                if cell_key(o2[0], o2[1], cell) not in near_keys:
                    if dist(o1, o2) > max_extension * 1.5:
                        continue
                hit = ray_ray_intersection(o1, d1, o2, d2)
                if hit is None:
                    continue
                t1, t2, p = hit
                pair = frozenset([k1, k2])
                if pair in seen:
                    continue
                seen.add(pair)
                candidates.append(("ray", min(t1, t2), k1, t1, k2, t2, p))

        best_for = {}
        has_geom = defaultdict(bool)
        for cand in candidates:
            if cand[0] == "seg":
                _, t, key, pt, _ = cand
                has_geom[key] = True
                if key not in best_for or t < best_for[key][0]:
                    best_for[key] = (t, cand)
            elif cand[0] == "ray":
                _, _, k1, t1, k2, t2, p = cand
                has_geom[k1] = True
                has_geom[k2] = True
                if k1 not in best_for or t1 < best_for[k1][0]:
                    best_for[k1] = (t1, cand)
                if k2 not in best_for or t2 < best_for[k2][0]:
                    best_for[k2] = (t2, cand)
        for cand in candidates:
            if cand[0] != "prox":
                continue
            _, t, key, pt, _ = cand
            if has_geom[key]:
                continue
            if key not in best_for or t < best_for[key][0]:
                best_for[key] = (t, cand)

        def find_partner(key, cand):
            o1, d1 = terminals[key]
            for opi in other_paths(cand, key):
                for st in (True, False):
                    k2 = (opi, st)
                    if k2 not in best_for or k2 == key:
                        continue
                    if key[0] not in other_paths(best_for[k2][1], k2):
                        continue
                    o2, d2 = terminals[k2]
                    if dist(o1, o2) > max_extension:
                        continue
                    # both must be looking toward each other
                    if (o2[0] - o1[0]) * d1[0] + (o2[1] - o1[1]) * d1[1] <= 0:
                        continue
                    if (o1[0] - o2[0]) * d2[0] + (o1[1] - o2[1]) * d2[1] <= 0:
                        continue
                    return k2
            return None

        applied_this_round = 0

        # ---- 1. mutual pairs: one point for both ----
        for key, (t, cand) in sorted(best_for.items(), key=lambda kv: kv[1][0]):
            if key in applied:
                continue
            partner = find_partner(key, cand)
            if partner is None or partner in applied:
                continue
            o1 = terminals[key][0]
            o2 = terminals[partner][0]
            mid = ((o1[0] + o2[0]) * 0.5, (o1[1] + o2[1]) * 0.5)
            set_terminal(key, mid)
            set_terminal(partner, mid)
            hit_points.append({"pt": mid, "path": key[0], "end": "start" if key[1] else "end"})
            hit_points.append({"pt": mid, "path": partner[0], "end": "start" if partner[1] else "end"})
            applied.add(key)
            applied.add(partner)
            applied_this_round += 1
            n_mutual += 1
            extension_lengths.append(dist(o1, mid))
            extension_lengths.append(dist(o2, mid))

        # ---- 2. leftover unpaired (true T onto someone else's stroke) ----
        for key, (t, cand) in sorted(best_for.items(), key=lambda kv: kv[1][0]):
            if key in applied:
                continue
            if cand[0] in ("seg", "prox"):
                _, t, k, pt, _ = cand
                if k in applied:
                    continue
                o, d = terminals[k]
                pt2 = (o[0] + d[0] * (t + past_hit), o[1] + d[1] * (t + past_hit))
                set_terminal(k, pt2)
                hit_points.append({"pt": tuple(pt2), "path": k[0], "end": "start" if k[1] else "end"})
                applied.add(k)
                applied_this_round += 1
                extension_lengths.append(t)
                if cand[0] == "seg":
                    n_seg += 1
                else:
                    n_prox += 1
            else:
                _, _, k1, t1, k2, t2, p = cand
                if k1 in applied or k2 in applied:
                    continue
                if best_for.get(k1, (None,))[1] is not cand:
                    continue
                if best_for.get(k2, (None,))[1] is not cand:
                    continue
                set_terminal(k1, p)
                set_terminal(k2, p)
                hit_points.append({"pt": tuple(p), "path": k1[0], "end": "start" if k1[1] else "end"})
                hit_points.append({"pt": tuple(p), "path": k2[0], "end": "start" if k2[1] else "end"})
                applied.add(k1)
                applied.add(k2)
                applied_this_round += 1
                n_ray += 1
                extension_lengths.append(t1)
                extension_lengths.append(t2)

        round_log.append(
            f"  round {rnd+1}: free={len(terminals)}  cand={len(candidates)}  "
            f"applied={applied_this_round}  mutual={n_mutual}"
        )
        if applied_this_round == 0:
            break

    extended_records = []
    all_segments = []
    for pi, rec in enumerate(records):
        segs = work[pi]
        new_segs = [(tuple(a), tuple(b)) for a, b in segs] if segs else []
        new_rec = dict(rec)
        new_rec["segments"] = new_segs
        new_rec["length"] = sum(dist(a, b) for a, b in new_segs) if new_segs else 0.0
        extended_records.append(new_rec)
        all_segments.extend(new_segs)

    summary_path = out_dir / f"{stem}_step2_summary.txt"
    bbox = segments_bbox(all_segments)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("STEP 2 – nearest hit; mutual pairs meet at midpoint\n")
        f.write(f"  max_extension     : {max_extension}\n")
        f.write(f"  lateral_tol       : {lateral_tol}\n")
        f.write(f"  max_rounds        : {max_rounds}\n")
        f.write(f"  mutual pair joins : {n_mutual}\n")
        f.write(f"  segment-hit joins : {n_seg}\n")
        f.write(f"  ray–ray joins     : {n_ray}\n")
        f.write(f"  proximity joins   : {n_prox}\n")
        f.write(f"  terminals done    : {len(applied)}\n")
        for line in round_log:
            f.write(line + "\n")
        if extension_lengths:
            f.write(
                f"  hit distance      : min={min(extension_lengths):.2f}  "
                f"max={max(extension_lengths):.2f}  "
                f"mean={sum(extension_lengths)/len(extension_lengths):.2f}\n"
            )
        f.write(f"  segments total    : {len(all_segments)}\n")
        f.write(f"  bbox              : {bbox}\n")
    print(f"Step 2 summary → {summary_path}")

    png_path = out_dir / f"{stem}_step2_extended.png"
    _draw_step2_debug(
        original_records=records,
        extended_records=extended_records,
        page_bbox=bbox,
        out_path=png_path,
        extension=max_extension,
    )
    print(f"Step 2 image   → {png_path}")

    return extended_records, all_segments, applied, hit_points


def step3_snap_open_to_hits(
    records,
    applied,
    hit_points,
    out_dir: Path,
    stem: str,
    join_radius: float = 5.0,
    path_snap_radius: float = 1.0,
):
    """
    Step 3: resolve nearby hit points and open path terminals into
    common junctions.

    Hit points produced by Step 2 and currently open path terminals are
    treated as connection candidates. Candidates within `join_radius`
    are grouped together. For each group of two or more candidates:

      1. Compute the group's average position.
      2. Exclude all paths already represented by the group.
      3. If the average position is within `path_snap_radius` of another
         path, use the nearest point on that foreign path as the junction.
         The target may lie anywhere along the path, not necessarily at
         a terminal or vertex.
      4. Otherwise, use the group's average position as the junction.
      5. Move all participating hit/open terminals to the resulting
         junction.
    """
    from shapely.geometry import LineString, Point

    work = [[list(s) for s in rec["segments"]] for rec in records]

    def set_terminal(pi, is_start, pt):
        if not work[pi]:
            return
        if is_start:
            work[pi][0][0] = pt
        else:
            work[pi][-1][1] = pt

    def end_flag(s):
        return s == "start" or s is True

    def nearest_on_foreign_paths(pt, skip):
        """Nearest point on any path whose index is not in skip. Returns (dist, xy) or (None, None)."""
        p = Point(pt)
        best_d, best_xy = None, None
        for pi, segs in enumerate(work):
            if pi in skip:
                continue
            for a, b in segs:
                if dist(a, b) < 1e-12:
                    continue
                ls = LineString([tuple(a), tuple(b)])
                d = ls.distance(p)
                if best_d is not None and d >= best_d:
                    continue
                proj = ls.interpolate(ls.project(p))
                best_d = d
                best_xy = (proj.x, proj.y)
        return best_d, best_xy

    # --- nodes: every hit + every open terminal (no hit_cluster) ---
    nodes = []
    for h in hit_points:
        nodes.append({
            "kind": "hit",
            "pt": h["pt"],
            "members": [h],
        })

    for pi, segs in enumerate(work):
        if not segs:
            continue
        for is_start, pt in ((True, tuple(segs[0][0])), (False, tuple(segs[-1][1]))):
            if (pi, is_start) in applied:
                continue
            nodes.append({
                "kind": "open",
                "pt": pt,
                "pi": pi,
                "is_start": is_start,
            })

    parent = list(range(len(nodes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if dist(nodes[i]["pt"], nodes[j]["pt"]) <= join_radius:
                union(i, j)

    groups = defaultdict(list)
    for i in range(len(nodes)):
        groups[find(i)].append(i)

    events = []
    n_snapped = 0
    n_path_snaps = 0

    for idxs in groups.values():
        if len(idxs) < 2:
            continue

        pts = [nodes[i]["pt"] for i in idxs]
        avg = (
            sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts),
        )

        skip = set()
        for i in idxs:
            nd = nodes[i]
            if nd["kind"] == "open":
                skip.add(nd["pi"])
            else:
                for m in nd["members"]:
                    skip.add(m["path"])

        d_path, on_path = nearest_on_foreign_paths(avg, skip)
        if d_path is not None and d_path <= path_snap_radius:
            target = on_path
            n_path_snaps += 1
            where = f"PATH_SNAP dist={d_path:.3f}"
        else:
            target = avg
            where = "AVG"

        events.append(
            f"GROUP size={len(idxs)}  {where}  "
            f"target=({target[0]:.3f},{target[1]:.3f})"
        )

        for i in idxs:
            nd = nodes[i]
            if nd["kind"] == "open":
                set_terminal(nd["pi"], nd["is_start"], target)
                n_snapped += 1
                events.append(
                    f"  SNAP open  path={nd['pi']}  "
                    f"end={'start' if nd['is_start'] else 'end'}"
                )
            else:
                for m in nd["members"]:
                    set_terminal(m["path"], end_flag(m["end"]), target)
                    n_snapped += 1
                events.append(
                    f"  SNAP hit  paths={[m['path'] for m in nd['members']]}"
                )

    out_records = []
    all_segments = []
    for pi, rec in enumerate(records):
        segs = work[pi]
        new_segs = [(tuple(a), tuple(b)) for a, b in segs] if segs else []
        new_rec = dict(rec)
        new_rec["segments"] = new_segs
        new_rec["length"] = sum(dist(a, b) for a, b in new_segs) if new_segs else 0.0
        out_records.append(new_rec)
        all_segments.extend(new_segs)

    n_open = sum(1 for nd in nodes if nd["kind"] == "open")
    summary_path = out_dir / f"{stem}_step3_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("STEP 3 – resolve nearby hit/open candidates into junctions\n")
        f.write(f"  join_radius      : {join_radius}\n")
        f.write(f"  path_snap_radius : {path_snap_radius}\n")
        f.write(f"  hit_points       : {len(hit_points)}\n")
        f.write(f"  open terminals   : {n_open}\n")
        f.write(f"  groups >=2       : {sum(1 for v in groups.values() if len(v)>=2)}\n")
        f.write(f"  path snaps       : {n_path_snaps}\n")
        f.write(f"  snaps            : {n_snapped}\n")
        f.write("\nEvents:\n")
        if events:
            for line in events:
                f.write(f"  {line}\n")
        else:
            f.write("  (none)\n")
    print(f"Step 3 summary → {summary_path}")

    png_path = out_dir / f"{stem}_step3_junctions.png"
    bbox = segments_bbox(all_segments)
    _draw_step2_debug(
        original_records=records,
        extended_records=out_records,
        page_bbox=bbox,
        out_path=png_path,
        extension=join_radius,
    )
    print(f"Step 3 image   → {png_path}")

    return out_records, all_segments

def step4_snap_and_union(
    records,
    out_dir: Path,
    stem: str,
    snap_tol: float = 0.05,
):
    """
    step 4: tiny endpoint snap (float-noise only) + unary_union noding.

    snap_tol ~ 0.01–0.1 is intentional: we are NOT bridging dash gaps here
    (extension already did that). We only collapse near-identical coordinates
    so GEOS does not leave hairline gaps that break polygonize.

    unary_union builds the planar arrangement: intersections become nodes,
    overlapping collinear pieces merge.

    Returns
    -------
    noded_geom : shapely geometry (often MultiLineString / GeometryCollection)
    noded_segments : list of ((x1,y1),(x2,y2)) extracted for plotting
    """
    from shapely.geometry import LineString, MultiLineString, Point
    from shapely.ops import unary_union

    # --- flat list of segments from records ---
    segs = []
    for rec in records:
        segs.extend(rec["segments"])

    if not segs:
        print("step 4: no segments")
        return None, []

    # --- tiny endpoint snap via coordinate quantisation ---
    # Group endpoints that fall in the same snap cell, replace by mean.
    def snap_key(p):
        return (round(p[0] / snap_tol), round(p[1] / snap_tol))

    buckets = defaultdict(list)
    for a, b in segs:
        buckets[snap_key(a)].append(a)
        buckets[snap_key(b)].append(b)

    rep = {}
    for k, pts in buckets.items():
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        rep[k] = (mx, my)

    snapped = []
    for a, b in segs:
        a2, b2 = rep[snap_key(a)], rep[snap_key(b)]
        if dist(a2, b2) > 1e-9:
            snapped.append((a2, b2))

    # --- unary_union (noding) ---
    lines = [LineString([a, b]) for a, b in snapped]
    noded = unary_union(MultiLineString(lines))

    # extract segments for debug plot
    noded_segments = []
    geoms = []
    if noded.is_empty:
        pass
    elif noded.geom_type == "LineString":
        geoms = [noded]
    elif noded.geom_type == "MultiLineString":
        geoms = list(noded.geoms)
    elif hasattr(noded, "geoms"):
        geoms = [g for g in noded.geoms if g.geom_type in ("LineString", "MultiLineString")]
        flat = []
        for g in geoms:
            if g.geom_type == "LineString":
                flat.append(g)
            else:
                flat.extend(g.geoms)
        geoms = flat

    for g in geoms:
        coords = list(g.coords)
        for i in range(len(coords) - 1):
            noded_segments.append((coords[i], coords[i + 1]))

    # --- summary ---
    summary_path = out_dir / f"{stem}_step4_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("step 4 – tiny snap + unary_union\n")
        f.write(f"  snap_tol           : {snap_tol}\n")
        f.write(f"  input segments     : {len(segs)}\n")
        f.write(f"  after snap         : {len(snapped)}\n")
        f.write(f"  noded geom type    : {noded.geom_type}\n")
        f.write(f"  noded pieces       : {len(geoms)}\n")
        f.write(f"  noded segments out : {len(noded_segments)}\n")
        if not noded.is_empty:
            f.write(f"  noded bounds       : {noded.bounds}\n")
            f.write(f"  noded length       : {noded.length:.1f}\n")
    print(f"step 4 summary → {summary_path}")

    # --- PNG ---
    png_path = out_dir / f"{stem}_step4_noded.png"
    _draw_noded_debug(noded_segments, noded.bounds if not noded.is_empty else (0, 0, 1, 1),
                      png_path, snap_tol)
    print(f"step 4 image   → {png_path}")

    return noded, noded_segments

def step5_polygonize(noded, out_dir: Path, stem: str, min_face_area: float = 400.0):
    """
    Step 5: turn noded linework into closed faces (Shapely polygonize).
    Each face is a dict with 'polygon', 'points', 'area', 'bbox'.
    """
    if noded is None or noded.is_empty:
        print("Step 5: empty noded geometry")
        return []

    raw = list(polygonize(noded))
    faces = []
    rejected = 0
    for poly in raw:
        if poly is None or poly.is_empty:
            rejected += 1
            continue
        if poly.geom_type != "Polygon":
            continue
        if poly.area < min_face_area:
            rejected += 1
            continue
        coords = list(poly.exterior.coords)
        faces.append({
            "polygon": poly,
            "points": coords[:-1],
            "area": float(poly.area),
            "bbox": poly.bounds,
            "perimeter": float(poly.length),
        })
    faces.sort(key=lambda f: f["area"], reverse=True)

    summary_path = out_dir / f"{stem}_step5_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("STEP 5 – polygonize\n")
        f.write(f"  raw faces     : {len(raw)}\n")
        f.write(f"  min_face_area : {min_face_area}\n")
        f.write(f"  kept          : {len(faces)}\n")
        f.write(f"  rejected      : {rejected}\n")
        for i, face in enumerate(faces[:30]):
            f.write(f"  [{i}] area={face['area']:.0f}  peri={face['perimeter']:.1f}  "
                    f"pts={len(face['points'])}  bbox={face['bbox']}\n")
        if len(faces) > 30:
            f.write(f"  … and {len(faces) - 30} more\n")
    print(f"Step 5 summary → {summary_path}")

    png_path = out_dir / f"{stem}_step5_faces.png"
    _draw_faces_debug(faces, png_path)
    print(f"Step 5 image   → {png_path}")
    return faces


def step_write_selected(selected, out_dir: Path, stem: str):
    summary_path = out_dir / f"{stem}_selected_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("SELECTED – combined outlines\n")
        f.write(f"  pieces : {len(selected)}\n")
        for i, p in enumerate(selected):
            f.write(
                f"  [{i}] area={p.get('area', 0):.0f}  peri={p.get('perimeter', 0):.1f}  "
                f"pts={len(p.get('points') or [])}  bbox={p.get('bbox')}\n"
            )
            f.write(f"      segments ({len(p.get('segments') or [])}):\n")
            for s in p.get("segments") or []:
                f.write(f"        {s[0]} -> {s[1]}\n")
    print(f"SELECTED summary → {summary_path}")

    json_path = out_dir / f"{stem}_selected.json"
    payload = []
    for i, p in enumerate(selected):
        payload.append({
            "index": i,
            "source_piece": p.get("source_piece"),
            "area": p.get("area"),
            "perimeter": p.get("perimeter"),
            "bbox": [float(x) for x in (p.get("bbox") or ())],
            "selected_faces": p.get("selected_faces") or [],
            "points": [[float(x), float(y)] for x, y in (p.get("points") or [])],
        })
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"pieces": payload}, f, indent=2)
    print(f"SELECTED json    → {json_path}")

    png_path = out_dir / f"{stem}_selected.png"
    _draw_selected_debug(selected, png_path)
    print(f"SELECTED image   → {png_path}")


def group_faces_into_pieces(faces, min_shared_edge: float = 1.0):
    """
    Faces that share a real edge belong to the same pattern piece.
    Returns faces_per_piece: list of (piece_idx, [face, ...]) for FaceSelector.
    """
    n = len(faces)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            gi, gj = faces[i]["polygon"], faces[j]["polygon"]
            if not gi.touches(gj):
                continue
            try:
                inter = gi.intersection(gj)
            except Exception:
                continue
            if inter.length >= min_shared_edge:
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    faces_per_piece = []
    for pi, idxs in enumerate(sorted(groups.values(),
                                     key=lambda ix: -sum(faces[k]["area"] for k in ix))):
        faces_per_piece.append((pi, [faces[k] for k in idxs]))
    return faces_per_piece

def _draw_lines_debug(page, lines, out_path: Path, assembled=None):
    """Same canvas style as _draw_step2_debug / _draw_faces_debug."""
    if assembled is not None:
        r = assembled["global_rect"]
        x0, y0, x1, y1 = r.x0, r.y0, r.x1, r.y1
        grid = assembled.get("grid")
        tile_w = assembled.get("tile_w")
        tile_h = assembled.get("tile_h")
    else:
        r = page.rect
        x0, y0, x1, y1 = r.x0, r.y0, r.x1, r.y1
        grid = tile_w = tile_h = None

    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    fig, ax = plt.subplots(figsize=(14, max(6.0, 14 * h / w)))
    ax.set_xlim(x0 - 20, x1 + 20)
    ax.set_ylim(y1 + 20, y0 - 20)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("LINES – grain (red) / fold (purple)", fontsize=11)

    if grid is not None and tile_w and tile_h:
        cols, rows = grid
        for rr in range(rows + 1):
            ax.axhline(rr * tile_h, color="0.90", linewidth=0.5, zorder=0)
        for cc in range(cols + 1):
            ax.axvline(cc * tile_w, color="0.90", linewidth=0.5, zorder=0)

    # faint context: every drawing on the page
    ctx = []
    for path in page.get_drawings():
        ctx.extend(path_to_segments(path))
    if ctx:
        ax.add_collection(LineCollection(
            ctx, colors=["0.82"], linewidths=0.5, alpha=0.8, zorder=1
        ))

    for i, obj in enumerate(lines):
        color = "red" if obj["type"] == "grain" else "purple"
        segs = obj.get("all_segments") or obj["shaft"]["segments"]
        if segs:
            ax.add_collection(LineCollection(
                segs, colors=[color], linewidths=2.0, alpha=0.95, zorder=3
            ))
        for ep in obj["shaft"]["endpoints"]:
            ax.plot(ep[0], ep[1], "o", color="orange", markersize=5, zorder=4)
        ax.text(
            obj["shaft"]["endpoints"][0][0],
            obj["shaft"]["endpoints"][0][1],
            f"{i}:{obj['type'][0].upper()}",
            color=color, fontsize=8, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
            zorder=5,
        )

    n_grain = sum(1 for o in lines if o["type"] == "grain")
    n_fold = sum(1 for o in lines if o["type"] == "fold")
    ax.text(
        0.01, 0.99,
        f"grain={n_grain}  fold={n_fold}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=3),
        zorder=10,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()

def _draw_faces_debug(faces, out_path: Path):
    if not faces:
        return
    xs0, ys0, xs1, ys1 = zip(*[f["bbox"] for f in faces])
    x0, y0, x1, y1 = min(xs0), min(ys0), max(xs1), max(ys1)
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    fig, ax = plt.subplots(figsize=(14, max(6.0, 14 * h / w)))
    ax.set_xlim(x0 - 20, x1 + 20)
    ax.set_ylim(y1 + 20, y0 - 20)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("STEP 5 – polygonize faces", fontsize=11)
    try:
        cmap = mpl.colormaps["tab20"]
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap("tab20")
    for i, face in enumerate(faces):
        ax.add_patch(MplPolygon(
            face["points"], closed=True,
            facecolor=cmap(i % 20), edgecolor="0.2",
            linewidth=0.6, alpha=0.55,
        ))
    ax.text(0.01, 0.99, f"faces={len(faces)}", transform=ax.transAxes,
            fontsize=9, va="top",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=3))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

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
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
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
    ax.set_title(f"STEP 2 – extend to first hit (max={extension})", fontsize=11)

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
        f"grey=original  colour=extended  max_extension={extension}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=3),
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def _draw_noded_debug(noded_segments, bounds, out_path, snap_tol):
    """Single-colour plot of the noded linework."""
    x0, y0, x1, y1 = bounds
    pad = 20.0
    w = max(x1 - x0, 1.0)
    h = max(y1 - y0, 1.0)
    fig_w = 14
    fig_h = max(6.0, fig_w * h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y1 + pad, y0 - pad)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"step 4 – snap({snap_tol}) + unary_union", fontsize=11)

    if noded_segments:
        lc = LineCollection(noded_segments, colors=["0.15"], linewidths=1.0, alpha=0.9)
        ax.add_collection(lc)

        # degree of each snapped node
        q = 3  # round to 0.001
        deg = defaultdict(int)
        pos = {}
        for a, b in noded_segments:
            if dist(a, b) < 1e-12:
                continue
            ka = (round(a[0], q), round(a[1], q))
            kb = (round(b[0], q), round(b[1], q))
            deg[ka] += 1
            deg[kb] += 1
            pos[ka] = a
            pos[kb] = b

        d1 = [pos[k] for k, d in deg.items() if d == 1]
        d3 = [pos[k] for k, d in deg.items() if d >= 3]
        # degree-2 vertices omitted on purpose (just samples along a stroke)

        if d1:
            xs, ys = zip(*d1)
            ax.scatter(xs, ys, s=28, c="red", zorder=6, linewidths=0, label="dangle (deg 1)")
        if d3:
            xs, ys = zip(*d3)
            ax.scatter(xs, ys, s=22, c="lime", zorder=6, linewidths=0.3,
                       edgecolors="black", label="junction (deg 3+)")

        node_count = len(deg)
        n_dangle = len(d1)
        n_junc = len(d3)
    else:
        node_count = n_dangle = n_junc = 0

    ax.text(
        0.01, 0.99,
        f"noded segments={len(noded_segments)}  nodes={node_count}  snap_tol={snap_tol}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=3),
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _draw_selected_debug(selected, out_path: Path):
    if not selected:
        return
    xs0, ys0, xs1, ys1 = zip(*[p["bbox"] for p in selected])
    x0, y0, x1, y1 = min(xs0), min(ys0), max(xs1), max(ys1)
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    fig, ax = plt.subplots(figsize=(14, max(6.0, 14 * h / w)))
    ax.set_xlim(x0 - 20, x1 + 20)
    ax.set_ylim(y1 + 20, y0 - 20)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("SELECTED – combined outlines", fontsize=11)
    try:
        cmap = mpl.colormaps["tab10"]
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap("tab10")
    for i, p in enumerate(selected):
        segs = p.get("segments") or []
        if segs:
            ax.add_collection(LineCollection(
                segs, colors=[cmap(i % 10)], linewidths=2.0, zorder=3
            ))
    ax.text(
        0.01, 0.99, f"pieces={len(selected)}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=3),
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()    

# ------------------------------------------------------------------
# INTERACTIVE FACE SELECTOR  (v2 – faster, per-piece colours, start selected)
# ------------------------------------------------------------------

class FaceSelector:
    def __init__(self, page, faces_per_piece, assembled=None,
                 title="Select faces – click to toggle"):
        self.faces_per_piece = faces_per_piece
        self.done = False
        self.cancelled = False

        # Flatten
        self.flat_faces = []          # (piece_idx, face_idx, face_dict)
        for pi, faces in faces_per_piece:
            for fi, face in enumerate(faces):
                self.flat_faces.append((pi, fi, face))

        # Start with EVERYTHING selected
        self.selected = {(pi, fi) for pi, fi, _ in self.flat_faces}

        # One stable colour per pattern piece
        n_pieces = max((pi for pi, _, _ in self.flat_faces), default=0) + 1
        try:
            # Matplotlib ≥ 3.7
            cmap = mpl.colormaps["tab20"]
        except (AttributeError, KeyError):
            # Fallback for older versions
            cmap = plt.cm.get_cmap("tab20")

        self.piece_colors = {pi: cmap(pi % 20) for pi in range(max(n_pieces, 1))}
        # ---- figure ----
        if assembled is not None:
            global_rect = assembled["global_rect"]
            fig_w = 16
            fig_h = max(8.0, fig_w * global_rect.height / max(global_rect.width, 1.0))
            self.fig, self.ax = plt.subplots(figsize=(fig_w, fig_h))
            self.ax.set_xlim(global_rect.x0 - 10, global_rect.x1 + 10)
            self.ax.set_ylim(global_rect.y1 + 10, global_rect.y0 - 10)
        else:
            page_rect = page.rect
            self.fig, self.ax = plt.subplots(
                figsize=(12, 12 * page_rect.height / page_rect.width)
            )
            self.ax.set_xlim(page_rect.x0, page_rect.x1)
            self.ax.set_ylim(page_rect.y1, page_rect.y0)

            pix = page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            self.ax.imshow(
                img,
                extent=[page_rect.x0, page_rect.x1, page_rect.y1, page_rect.y0],
                alpha=0.35,
                zorder=0,
            )

        self.ax.set_aspect("equal")
        self.ax.axis("off")

        # Title higher up and smaller so it doesn’t cover content
        self.ax.set_title(title, pad=2, fontsize=11)

        # Create patches ONCE
        self.patches = {}          # (pi, fi) → MplPolygon
        self.labels  = {}          # (pi, fi) → text artist

        # artists that show the combined outline of each piece
        self.combined_outlines = {}   # piece_idx → LineCollection or list of plots    

        for pi, fi, face in self.flat_faces:
            color = self.piece_colors[pi]

            poly = MplPolygon(
                face["points"],
                closed=True,
                facecolor=color,
                edgecolor="0.45",      # same as _update_face_style
                linewidth=0.7,         # same as selected state
                alpha=0.62,            # same as selected state
                zorder=3,
                picker=False,
            )
            self.ax.add_patch(poly)
            self.patches[(pi, fi)] = poly

            # # label
            # cx = sum(p[0] for p in face["points"]) / len(face["points"])
            # cy = sum(p[1] for p in face["points"]) / len(face["points"])
            # txt = self.ax.text(
            #     cx, cy, f"{pi}.{fi}",
            #     color="black", fontsize=8, fontweight="bold",
            #     ha="center", va="center",
            #     bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1),
            #     zorder=5,
            # )
            # self.labels[(pi, fi)] = txt

        # Make sure every face starts with the correct visual style
        for pi, fi, _ in self.flat_faces:
            self._update_face_style(pi, fi)

        # Then draw the combined outlines
        self._update_all_combined_outlines()

        # Status bar – placed very high and compact
        self.status = self.ax.text(
            0.01, 0.995,
            self._status_text(),
            transform=self.ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(facecolor="white", alpha=0.88, edgecolor="none", pad=3),
            zorder=10,
        )

        # events
        self.cid_click = self.fig.canvas.mpl_connect(
            "button_press_event", self._on_click
        )
        self.cid_key = self.fig.canvas.mpl_connect(
            "key_press_event", self._on_key
        )

        # Make layout tighter
        self.fig.tight_layout(pad=0.4)
        self.fig.subplots_adjust(top=0.96)   # push title up

    def _update_all_combined_outlines(self):
        piece_ids = {pi for pi, _, _ in self.flat_faces}
        for pi in piece_ids:
            self._update_combined_outline(pi)

    def _compute_combined_polygon(self, piece_idx):
        """Return the unary union of all currently selected faces of this piece."""
        from shapely.ops import unary_union
        polys = []
        for pi, fi, face in self.flat_faces:
            if pi == piece_idx and (pi, fi) in self.selected:
                polys.append(face["polygon"])
        if not polys:
            return None
        try:
            u = unary_union(polys)
            if u.is_empty:
                return None
            if u.geom_type == "MultiPolygon":
                # take the largest part (usually there is only one)
                u = max(u.geoms, key=lambda g: g.area)
            return u if u.geom_type == "Polygon" else None
        except Exception:
            return None
        
    def _update_combined_outline(self, piece_idx):
        """Strong, unmistakable main outline."""
        # remove previous
        if piece_idx in self.combined_outlines:
            arts = self.combined_outlines[piece_idx]
            if not isinstance(arts, (list, tuple)):
                arts = [arts]
            for a in arts:
                a.remove()
            del self.combined_outlines[piece_idx]

        poly = self._compute_combined_polygon(piece_idx)
        if poly is None or poly.exterior is None:
            return

        coords = list(poly.exterior.coords)
        xs, ys = zip(*coords)

        # soft glow underneath
        glow, = self.ax.plot(
            xs, ys,
            color="white",
            linewidth=7.0,
            solid_capstyle="round",
            solid_joinstyle="round",
            alpha=0.45,
            zorder=6,
        )
        # crisp dark core on top
        core, = self.ax.plot(
            xs, ys,
            color="black",
            linewidth=3.0,
            solid_capstyle="round",
            solid_joinstyle="round",
            alpha=0.95,
            zorder=7,
        )
        self.combined_outlines[piece_idx] = [glow, core]


    def _status_text(self):
        n = len(self.selected)
        total = len(self.flat_faces)
        return (f"Selected: {n}/{total}   "
                f"click=toggle   Enter/d=done   a=all   c=clear   Esc=cancel")

    def _update_face_style(self, pi, fi):
        """Face edges stay subtle."""
        poly = self.patches[(pi, fi)]
        is_sel = (pi, fi) in self.selected
        color = self.piece_colors[pi]

        poly.set_facecolor(color)
        poly.set_edgecolor("0.45")          # medium-light grey
        poly.set_linewidth(0.7 if is_sel else 0.5)
        poly.set_alpha(0.62 if is_sel else 0.18)
        poly.set_zorder(3 if is_sel else 2)

    def _on_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        pt = Point(x, y)
        for pi, fi, face in reversed(self.flat_faces):
            if face["polygon"].contains(pt) or face["polygon"].touches(pt):
                key = (pi, fi)
                if key in self.selected:
                    self.selected.remove(key)
                else:
                    self.selected.add(key)

                # update only the face that changed + its piece outline
                self._update_face_style(pi, fi)
                self._update_combined_outline(pi)

                self.status.set_text(self._status_text())
                self.fig.canvas.draw_idle()
                return

    def _on_key(self, event):
        if event.key in ("enter", "d"):
            self.done = True
            plt.close(self.fig)
        elif event.key == "escape":
            self.cancelled = True
            self.selected.clear()
            plt.close(self.fig)
        elif event.key == "a":
            self.selected = {(pi, fi) for pi, fi, _ in self.flat_faces}
            for pi, fi, _ in self.flat_faces:
                self._update_face_style(pi, fi)
            self._update_all_combined_outlines()
            self.status.set_text(self._status_text())
            self.fig.canvas.draw_idle()
        elif event.key == "c":
            self.selected.clear()
            for pi, fi, _ in self.flat_faces:
                self._update_face_style(pi, fi)
            self._update_all_combined_outlines()
            self.status.set_text(self._status_text())
            self.fig.canvas.draw_idle()

    def run(self):
        plt.show(block=True)

        if self.cancelled:
            return []

        # ---- build the final combined outlines (what you actually want) ----
        from collections import defaultdict
        from shapely.ops import unary_union

        by_piece = defaultdict(list)
        for pi, fi, face in self.flat_faces:
            if (pi, fi) in self.selected:
                by_piece[pi].append(face["polygon"])

        result = []
        for pi, polys in sorted(by_piece.items()):
            try:
                u = unary_union(polys)
                if u.is_empty:
                    continue
                if u.geom_type == "MultiPolygon":
                    u = max(u.geoms, key=lambda g: g.area)
                if u.geom_type != "Polygon" or u.exterior is None:
                    continue

                coords = list(u.exterior.coords)
                segs = [(coords[i], coords[i+1]) for i in range(len(coords)-1)]
                result.append({
                    "segments": segs,
                    "points": coords[:-1],
                    "area": float(u.area),
                    "bbox": u.bounds,
                    "source_piece": pi,
                    "closed": True,
                    # keep empty for compatibility with write_patterns_txt / render
                    "size_variants": [],
                    "path_ids": [],          # no longer meaningful
                    "perimeter": float(u.length),
                })
            except Exception:
                continue

        return result


def select_faces_pyside6(faces_per_piece, lines=None, assembled=None):
    """
    Desktop picker. Hover highlights a whole piece. First click selects it.
    Later clicks toggle inner faces. Alt-click drops the piece.
    Returns combined outlines, same shape as FaceSelector.run().
    """
    import sys
    from PySide6.QtCore import Qt, QPointF, QTimer
    from PySide6.QtGui import (
        QBrush, QColor, QPainter, QPainterPath, QPen, QPolygonF,
    )
    from PySide6.QtWidgets import (
        QApplication, QDialog, QFrame, QGraphicsPolygonItem, QGraphicsPathItem,
        QGraphicsLineItem, QGraphicsScene, QGraphicsView, QHBoxLayout,
        QLabel, QPushButton, QVBoxLayout,
    )
    from shapely.ops import unary_union

    PALETTE = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
        "#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363",
    ]
    lines = lines or []
    pieces = {pi: fl for pi, fl in faces_per_piece}

    if assembled is not None:
        r = assembled["global_rect"]
        x0, y0, x1, y1 = r.x0 - 20, r.y0 - 20, r.x1 + 20, r.y1 + 20
    else:
        boxes = [f["bbox"] for fl in pieces.values() for f in fl]
        if boxes:
            x0 = min(b[0] for b in boxes) - 20
            y0 = min(b[1] for b in boxes) - 20
            x1 = max(b[2] for b in boxes) + 20
            y1 = max(b[3] for b in boxes) + 20
        else:
            x0 = y0 = 0.0
            x1 = y1 = 1.0

    class FaceItem(QGraphicsPolygonItem):
        def __init__(self, pi, fi, pts):
            super().__init__(QPolygonF([QPointF(p[0], p[1]) for p in pts]))
            self.pi = pi
            self.fi = fi
            self.setAcceptHoverEvents(True)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setZValue(1)

        def hoverEnterEvent(self, event):
            self.scene().picker.hover_piece(self.pi)
            super().hoverEnterEvent(event)

        def hoverLeaveEvent(self, event):
            self.scene().picker.schedule_unhover(self.pi)
            super().hoverLeaveEvent(event)

        def mousePressEvent(self, event):
            if event.button() == Qt.MouseButton.LeftButton:
                alt = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
                self.scene().picker.click_face(self.pi, self.fi, alt)
                event.accept()
                return
            super().mousePressEvent(event)

    class PatternView(QGraphicsView):
        def __init__(self, scene):
            super().__init__(scene)
            self.setRenderHint(QPainter.RenderHint.Antialiasing)
            self.setFrameShape(QFrame.Shape.NoFrame)
            self.setBackgroundBrush(QColor("#f3f3f3"))
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
            self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
            self.setMouseTracking(True)
            self._panning = False
            self._pan_pos = None

        def wheelEvent(self, event):
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)

        def mousePressEvent(self, event):
            hit = None
            for it in self.items(event.pos()):
                if isinstance(it, FaceItem):
                    hit = it
                    break
            if event.button() == Qt.MouseButton.LeftButton and hit is None:
                self._panning = True
                self._pan_pos = event.pos()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
            super().mousePressEvent(event)

        def mouseMoveEvent(self, event):
            if self._panning and self._pan_pos is not None:
                d = event.pos() - self._pan_pos
                self._pan_pos = event.pos()
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - d.x())
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - d.y())
                return
            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event):
            if self._panning:
                self._panning = False
                self._pan_pos = None
                self.setCursor(Qt.CursorShape.ArrowCursor)
                event.accept()
                return
            super().mouseReleaseEvent(event)

    class Picker(QDialog):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Select pattern outlines")
            self.resize(1400, 900)
            self.active = set()
            self.off = set()
            self.hover_pi = None
            self.output = []
            self.face_items = []
            self.outline_items = []

            self.scene = QGraphicsScene(self)
            self.scene.picker = self
            self.scene.setSceneRect(x0, y0, x1 - x0, y1 - y0)
            self.view = PatternView(self.scene)

            for pi, fl in faces_per_piece:
                for fi, face in enumerate(fl):
                    item = FaceItem(pi, fi, face["points"])
                    self.scene.addItem(item)
                    self.face_items.append(item)

            for obj in lines:
                color = QColor("#e31a1c" if obj["type"] == "grain" else "#6a3d9a")
                pen = QPen(color)
                pen.setCosmetic(True)
                pen.setWidthF(2.0)
                for a, b in obj.get("all_segments") or []:
                    ln = QGraphicsLineItem(a[0], a[1], b[0], b[1])
                    ln.setPen(pen)
                    ln.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                    ln.setZValue(8)
                    self.scene.addItem(ln)

            hint = QLabel(
                "Red = grain, purple = fold. Hover highlights a piece. "
                "Click selects it. Click an inner face to toggle it. "
                "Alt-click drops the piece. Drag the background to pan, wheel to zoom."
            )
            hint.setWordWrap(True)
            self.status = QLabel()

            all_btn = QPushButton("All")
            clear_btn = QPushButton("Clear")
            approve = QPushButton("Approve")
            cancel = QPushButton("Cancel")
            approve.setObjectName("approve")
            all_btn.clicked.connect(self.select_all)
            clear_btn.clicked.connect(self.clear)
            approve.clicked.connect(self.approve)
            cancel.clicked.connect(self.reject)

            row = QHBoxLayout()
            row.addWidget(all_btn)
            row.addWidget(clear_btn)
            row.addStretch(1)
            row.addWidget(cancel)
            row.addWidget(approve)

            lay = QVBoxLayout(self)
            lay.setContentsMargins(10, 10, 10, 10)
            lay.addWidget(hint)
            lay.addWidget(self.view, 1)
            lay.addLayout(row)
            lay.addWidget(self.status)
            self.restyle_all()
            self.update_status()

        def showEvent(self, event):
            super().showEvent(event)
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        def color_for(self, pi, alpha):
            c = QColor(PALETTE[pi % len(PALETTE)])
            c.setAlpha(alpha)
            return c

        def restyle_all(self):
            for it in self.face_items:
                self.apply_style(it)
            self.refresh_outlines()
            self.update_status()

        def restyle_piece(self, pi):
            for it in self.face_items:
                if it.pi == pi:
                    self.apply_style(it)

        def apply_style(self, it):
            on_piece = it.pi in self.active
            face_on = on_piece and (it.pi, it.fi) not in self.off
            hovered = it.pi == self.hover_pi
            if on_piece and face_on:
                alpha = 210 if hovered else 175
            elif hovered:
                alpha = 120
            else:
                alpha = 40
            it.setBrush(QBrush(self.color_for(it.pi, alpha)))
            pen = QPen(QColor(40, 40, 40, 200 if face_on else 90))
            pen.setCosmetic(True)
            pen.setWidthF(1.6 if face_on else 0.9)
            it.setPen(pen)

        def hover_piece(self, pi):
            if self.hover_pi == pi:
                return
            old = self.hover_pi
            self.hover_pi = pi
            if old is not None:
                self.restyle_piece(old)
            self.restyle_piece(pi)

        def schedule_unhover(self, pi):
            def check():
                if self.hover_pi != pi:
                    return
                pos = self.view.mapFromGlobal(self.cursor().pos())
                still = any(
                    isinstance(it, FaceItem) and it.pi == pi
                    for it in self.view.items(pos)
                )
                if still:
                    return
                self.hover_pi = None
                self.restyle_piece(pi)
            QTimer.singleShot(0, check)

        def click_face(self, pi, fi, alt):
            if alt:
                self.active.discard(pi)
                self.off = {k for k in self.off if k[0] != pi}
            elif pi not in self.active:
                self.active.add(pi)
            else:
                key = (pi, fi)
                if key in self.off:
                    self.off.remove(key)
                else:
                    self.off.add(key)
                fl = pieces[pi]
                if all((pi, i) in self.off for i in range(len(fl))):
                    self.active.discard(pi)
                    self.off = {k for k in self.off if k[0] != pi}
            self.restyle_all()

        def select_all(self):
            self.off.clear()
            self.active = {pi for pi, _fl in faces_per_piece}
            self.restyle_all()

        def clear(self):
            self.active.clear()
            self.off.clear()
            self.restyle_all()

        def refresh_outlines(self):
            for it in self.outline_items:
                self.scene.removeItem(it)
            self.outline_items.clear()
            for pi in self.active:
                fl = pieces[pi]
                polys = [fl[fi]["polygon"] for fi in range(len(fl)) if (pi, fi) not in self.off]
                if not polys:
                    continue
                try:
                    u = unary_union(polys)
                except Exception:
                    continue
                if u.geom_type == "MultiPolygon":
                    u = max(u.geoms, key=lambda g: g.area)
                if u.geom_type != "Polygon":
                    continue
                path = QPainterPath()
                coords = list(u.exterior.coords)
                path.moveTo(coords[0][0], coords[0][1])
                for x, y in coords[1:]:
                    path.lineTo(x, y)
                path.closeSubpath()
                glow = QGraphicsPathItem(path)
                gpen = QPen(QColor(255, 255, 255, 180))
                gpen.setCosmetic(True)
                gpen.setWidthF(6)
                glow.setPen(gpen)
                glow.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                glow.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                glow.setZValue(5)
                core = QGraphicsPathItem(path)
                cpen = QPen(QColor(0, 0, 0))
                cpen.setCosmetic(True)
                cpen.setWidthF(2.4)
                core.setPen(cpen)
                core.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                core.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                core.setZValue(6)
                self.scene.addItem(glow)
                self.scene.addItem(core)
                self.outline_items.extend((glow, core))

        def update_status(self):
            total = sum(len(fl) for fl in pieces.values())
            n = 0
            for pi, fl in pieces.items():
                if pi in self.active:
                    n += sum(1 for fi in range(len(fl)) if (pi, fi) not in self.off)
            self.status.setText(
                f"Pieces {len(self.active)}/{len(pieces)}    faces {n}/{total}"
            )

        def build_result(self):
            result = []
            for pi, fl in faces_per_piece:
                if pi not in self.active:
                    continue
                chosen = [fi for fi in range(len(fl)) if (pi, fi) not in self.off]
                polys = [fl[fi]["polygon"] for fi in chosen]
                if not polys:
                    continue
                try:
                    u = unary_union(polys)
                    if u.is_empty:
                        continue
                    if u.geom_type == "MultiPolygon":
                        u = max(u.geoms, key=lambda g: g.area)
                    if u.geom_type != "Polygon":
                        continue
                    coords = list(u.exterior.coords)
                    result.append({
                        "segments": [(coords[i], coords[i + 1]) for i in range(len(coords) - 1)],
                        "points": coords[:-1],
                        "area": float(u.area),
                        "bbox": u.bounds,
                        "source_piece": pi,
                        "selected_faces": chosen,
                        "closed": True,
                        "size_variants": [],
                        "path_ids": [],
                        "perimeter": float(u.length),
                    })
                except Exception:
                    continue
            return result

        def approve(self):
            self.output = self.build_result()
            self.accept()

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QDialog { background: #fafafa; }
        QLabel { color: #222; font-size: 13px; }
        QPushButton {
            background: white; border: 1px solid #ccc; border-radius: 6px;
            padding: 6px 14px; font-size: 13px;
        }
        QPushButton:hover { background: #f0f0f0; }
        QPushButton#approve { background: #1f6feb; color: white; border: none; }
        QPushButton#approve:hover { background: #1858c4; }
    """)
    dlg = Picker()
    dlg.exec()
    return dlg.output


def select_faces_interactive(page, faces_per_piece, assembled=None):
    selector = FaceSelector(page, faces_per_piece, assembled=assembled)
    return selector.run()
    

def detect_special_lines(page,
                         min_shaft_length=20.0,
                         arrow_search_radius=12.0,
                         max_arrow_size=35.0,
                         max_arrow_segments=30,
                         min_arrow_width=0.7,
                         point_tol=1.0,
                         min_crossbar_length=4.0,
                         max_crossbar_length=90.0,
                         join_gap=8.0,
                         collinear_dot_min=0.98,
                         perp_dot_max=0.35,
                         min_fold_ends=1):
    """
    Unified grain/fold detection from straight runs + arrow sites.

    Grain: long run with arrow(s) near its own endpoints.
    Fold:  long run with short perpendicular run(s) at end(s)
           and arrow(s) near the outer end of those short runs.

    Shaft-first: all straight geometry is collected first; arrows are
    resolved only relative to concrete shafts. Short fold brackets are
    never excluded early.
    """
    from collections import defaultdict
    import math

    drawings = page.get_drawings()

    def quant(p):
        return (round(p[0] / point_tol), round(p[1] / point_tol))

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1])

    def nrm(v):
        l = math.hypot(v[0], v[1])
        if l < 1e-12:
            return (0.0, 0.0)
        return (v[0] / l, v[1] / l)

    def dot(u, v):
        return u[0] * v[0] + u[1] * v[1]

    def seg_dir_from_item(item):
        """Approximate unit direction from a path item ('l' or 'c')."""
        op = item[0]
        try:
            if op == "l":
                p1, p2 = item[1], item[2]
                return nrm(sub((p2.x, p2.y), (p1.x, p1.y)))
            if op == "c":
                p0, p3 = item[1], item[4]
                return nrm(sub((p3.x, p3.y), (p0.x, p0.y)))
        except Exception:
            return (0.0, 0.0)
        return (0.0, 0.0)

    def angle_deg_undirected(u, v):
        """Angle between directions in [0, 90]."""
        a = abs(dot(u, v))
        a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
        return math.degrees(math.acos(a))

    def arrow_has_angled_segment(segments, shaft_dir, min_deg=10.0, max_deg=80.0):
        """True if at least one segment is angled relative to shaft_dir."""
        if shaft_dir == (0.0, 0.0):
            return True
        for a, b in segments:
            d = nrm(sub(b, a))
            if d == (0.0, 0.0):
                continue
            ang = angle_deg_undirected(shaft_dir, d)
            if min_deg <= ang <= max_deg:
                return True
        return False

    # ------------------------------------------------------------------
    # 1) Collect ALL straight "l" segments + arrow candidates
    #    (no path is excluded from geometry)
    # ------------------------------------------------------------------
    segments = []
    arrow_paths = []

    for path_idx, path in enumerate(drawings):
        width = float(path.get("width") or 0.0)
        color = path.get("color")
        items = path.get("items", [])

        r = path.get("rect")
        segs_all = path_to_segments(path)

        # Record arrow candidates (geometry is still extracted below)
        if r is not None and segs_all:
            size = max(r.width, r.height)
            nseg = len(segs_all)
            thin = 0 < width < min_arrow_width
            if size <= max_arrow_size and nseg <= max_arrow_segments and not thin:
                arrow_paths.append({
                    "path": path,
                    "segments": segs_all,
                    "center": ((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2),
                    "path_id": path_idx,
                })

        # Always collect straight segments
        for item in items:
            if item[0] != "l":
                continue
            p1, p2 = item[1], item[2]
            a = (p1.x, p1.y)
            b = (p2.x, p2.y)
            length = dist(a, b)
            if length < 1.0:
                continue
            d = nrm(sub(b, a))
            segments.append({
                "a": a,
                "b": b,
                "length": length,
                "dir": d,
                "width": round(width, 2),
                "color": tuple(round(float(c), 3) for c in color[:3]) if color else None,
                "path": path,
                "path_id": path_idx,
            })

    if not segments:
        return []

    # ------------------------------------------------------------------
    # 2) Merge collinear endpoint-linked segments into runs
    # ------------------------------------------------------------------
    parent = list(range(len(segments)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    cell = max(join_gap * 1.5, 4.0)
    grid = defaultdict(list)

    def cell_of(p):
        return (int(p[0] // cell), int(p[1] // cell))

    for i, s in enumerate(segments):
        grid[cell_of(s["a"])].append((i, "a"))
        grid[cell_of(s["b"])].append((i, "b"))

    neigh = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]

    for i, s in enumerate(segments):
        for ep_name, ep in (("a", s["a"]), ("b", s["b"])):
            cx, cy = cell_of(ep)
            for dx, dy in neigh:
                for j, other_ep_name in grid.get((cx + dx, cy + dy), []):
                    if j <= i:
                        continue
                    t = segments[j]
                    if s["width"] != t["width"] or s["color"] != t["color"]:
                        continue
                    q = t[other_ep_name]
                    if dist(ep, q) > join_gap:
                        continue
                    if abs(dot(s["dir"], t["dir"])) < collinear_dot_min:
                        continue
                    union(i, j)

    groups = defaultdict(list)
    for i in range(len(segments)):
        groups[find(i)].append(i)

    runs = []
    for idxs in groups.values():
        segs = [segments[i] for i in idxs]
        length = sum(s["length"] for s in segs)
        longest = max(segs, key=lambda s: s["length"])
        direction = longest["dir"]

        counts = defaultdict(int)
        sample = {}
        for s in segs:
            for p in (s["a"], s["b"]):
                qp = quant(p)
                counts[qp] += 1
                sample[qp] = p
        terminals = [sample[q] for q, c in counts.items() if c == 1]

        if len(terminals) < 2:
            pts = [p for s in segs for p in (s["a"], s["b"])]
            best_d = -1.0
            t0 = t1 = pts[0]
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    d = dist(pts[i], pts[j])
                    if d > best_d:
                        best_d = d
                        t0, t1 = pts[i], pts[j]
            terminals = [t0, t1]
        elif len(terminals) > 2:
            best_d = -1.0
            t0, t1 = terminals[0], terminals[1]
            for i in range(len(terminals)):
                for j in range(i + 1, len(terminals)):
                    d = dist(terminals[i], terminals[j])
                    if d > best_d:
                        best_d = d
                        t0, t1 = terminals[i], terminals[j]
            terminals = [t0, t1]

        draw_segs = [(s["a"], s["b"]) for s in segs]
        path_ids = sorted({s["path_id"] for s in segs})

        runs.append({
            "seg_idxs": list(idxs),
            "segments": draw_segs,
            "max_l": max(s["length"] for s in segs),
            "length": length,
            "direction": direction,
            "terminals": (terminals[0], terminals[1]),
            "width": longest["width"],
            "color": longest["color"],
            "path_ids": path_ids,
        })

    long_runs = [r for r in runs if r["max_l"] >= min_shaft_length]
    long_runs.sort(key=lambda r: r["max_l"], reverse=True)

    short_runs = [
        r for r in runs
        if min_crossbar_length <= r["length"] <= max_crossbar_length
    ]

    def arrow_hits_point(ap, point, radius):
        if dist(ap["center"], point) <= radius:
            return True
        for a, b in ap["segments"]:
            if dist(a, point) <= radius or dist(b, point) <= radius:
                return True
        return False

    def arrow_near(point, radius, shaft_dir=None):
        for ap in arrow_paths:
            if not arrow_hits_point(ap, point, radius):
                continue
            if shaft_dir is not None and not arrow_has_angled_segment(ap["segments"], shaft_dir):
                continue
            return True
        return False

    def arrows_for_render(point, radius, shaft_dir=None):
        out = []
        for ap in arrow_paths:
            if not arrow_hits_point(ap, point, radius):
                continue
            if shaft_dir is not None and not arrow_has_angled_segment(ap["segments"], shaft_dir):
                continue
            out.append(ap)
        return out

    # ------------------------------------------------------------------
    # 3) Classify each long run (longest first; consume fold crossbars)
    # ------------------------------------------------------------------
    results = []
    used_seg_idxs = set()

    for L in long_runs:
        if set(L["seg_idxs"]) & used_seg_idxs:
            continue

        ep1, ep2 = L["terminals"]
        ldir = L["direction"]

        # Grain ends: arrow near the shaft terminal itself
        grain_ends = []
        grain_arrows = []
        for ep in (ep1, ep2):
            if arrow_near(ep, arrow_search_radius, shaft_dir=ldir):
                grain_ends.append(ep)
                grain_arrows.extend(arrows_for_render(ep, arrow_search_radius, shaft_dir=ldir))

        # Fold ends: external short perpendicular runs
        fold_ends = 0
        fold_bars = []
        fold_arrows = []

        for ep in (ep1, ep2):
            best = None  # (inner_dist, short_run, outer, arrows)
            for S in short_runs:
                if set(S["seg_idxs"]) & set(L["seg_idxs"]):
                    continue
                if set(S["seg_idxs"]) & used_seg_idxs:
                    continue
                # real fold brackets match shaft stroke; arrow edges do not
                if S["width"] != L["width"]:
                    continue
                if abs(dot(ldir, S["direction"])) > perp_dot_max:
                    continue

                s0, s1 = S["terminals"]
                d0 = dist(s0, ep)
                d1 = dist(s1, ep)
                if min(d0, d1) > join_gap:
                    continue

                if d0 <= d1:
                    inner, outer, inner_d = s0, s1, d0
                else:
                    inner, outer, inner_d = s1, s0, d1

                if not arrow_near(outer, arrow_search_radius, shaft_dir=ldir):
                    continue

                arr = arrows_for_render(outer, arrow_search_radius, shaft_dir=ldir)
                if best is None or inner_d < best[0]:
                    best = (inner_d, S, outer, arr)

            if best is not None:
                fold_ends += 1
                fold_bars.append(best[1])
                fold_arrows.extend(best[3])

        is_fold = fold_ends >= min_fold_ends
        is_grain = len(grain_ends) >= 1

        # Single-path U: short perp arms live inside the same long run
        if not is_fold and is_grain:
            own_short_perp = 0
            own_arrows = []
            for ep in (ep1, ep2):
                for si in L["seg_idxs"]:
                    s = segments[si]
                    if s["length"] < min_crossbar_length or s["length"] > max_crossbar_length:
                        continue
                    if s["width"] != L["width"]:
                        continue
                    if abs(dot(ldir, s["dir"])) > perp_dot_max:
                        continue
                    d0 = dist(s["a"], ep)
                    d1 = dist(s["b"], ep)
                    if min(d0, d1) > join_gap:
                        continue
                    outer = s["b"] if d0 <= d1 else s["a"]
                    if dist(outer, ep) < min_crossbar_length * 0.5:
                        continue
                    if arrow_near(outer, arrow_search_radius, shaft_dir=ldir):
                        own_short_perp += 1
                        own_arrows.extend(
                            arrows_for_render(outer, arrow_search_radius, shaft_dir=ldir)
                        )
            if own_short_perp >= min_fold_ends:
                is_fold = True
                fold_arrows = own_arrows

        if not is_fold and not is_grain:
            continue

        # Prefer fold when both could match (brackets + arrows near shaft ends)
        if is_fold:
            line_type = "fold"
            arrows = fold_arrows
            score = 100 + fold_ends * 10 + len(arrows)
            used_seg_idxs.update(L["seg_idxs"])
            for b in fold_bars:
                used_seg_idxs.update(b["seg_idxs"])
        else:
            line_type = "grain"
            arrows = grain_arrows
            score = len(grain_ends) * 10 + len(arrows)
            used_seg_idxs.update(L["seg_idxs"])

        # Consume only the geometry that belongs to the arrows we just used.
        # This prevents arrow edges from becoming false short shafts later,
        # while leaving nearby fold brackets / other shafts completely free.
        used_arrow_path_ids = {ap["path_id"] for ap in arrows}
        for i, s in enumerate(segments):
            if i in used_seg_idxs:
                continue
            if s["path_id"] in used_arrow_path_ids:
                used_seg_idxs.add(i)

        # de-dup arrows
        uniq = {}
        for a in arrows:
            uniq[id(a["path"])] = a
        arrows = list(uniq.values())

        extra = []
        if is_fold:
            for b in fold_bars:
                extra.extend(b["segments"])
        for a in arrows:
            extra.extend(a["segments"])

        results.append({
            "type": line_type,
            "shaft": {
                "path": None,
                "segments": L["segments"],
                "length": L["length"],
                "endpoints": L["terminals"],
                "rect": None,
                "num_segments": len(L["segments"]),
                "path_ids": L["path_ids"],
            },
            "arrows": arrows,
            "crossbars": fold_bars if is_fold else [],
            "score": score,
            "segments": L["segments"],
            "all_segments": L["segments"] + extra,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def parse_page_spec(spec, doc):
    """None → all pages. '0-2,5' → those indices."""
    if spec is None:
        return list(range(len(doc)))
    pages = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = map(int, part.split("-"))
            pages.extend(range(a, b + 1))
        else:
            pages.append(int(part))
    return sorted(set(p for p in pages if 0 <= p < len(doc)))


def main():
    parser = argparse.ArgumentParser(
        description="Assemble multi-page pattern, then debug steps 1–5 + UI"
    )
    parser.add_argument("--pattern-pdf", type=Path, required=True)
    parser.add_argument("--lines-pdf", type=Path, required=True)
    parser.add_argument("--pattern-pages", type=str, default=None)
    parser.add_argument("--lines-pages", type=str, default=None)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--min-seg-length", type=float, default=2.0)
    parser.add_argument("--max-extension", type=float, default=15.0)
    parser.add_argument("--join-radius", type=float, default=10.0)
    parser.add_argument("--path-snap-radius", type=float, default=2.0)
    parser.add_argument("--snap-tol", type=float, default=0.1)
    parser.add_argument("--min-face-area", type=float, default=400.0)
    args = parser.parse_args()

    if not args.pattern_pdf.is_file():
        raise SystemExit(f"Pattern PDF not found: {args.pattern_pdf}")
    if not args.lines_pdf.is_file():
        raise SystemExit(f"Lines PDF not found: {args.lines_pdf}")

    pattern_doc = fitz.open(args.pattern_pdf)
    instr_doc = fitz.open(args.lines_pdf)

    pat_pages = parse_page_spec(args.pattern_pages, pattern_doc)
    line_pages = parse_page_spec(args.lines_pages, instr_doc)
    if not pat_pages:
        raise SystemExit("No valid pattern pages")
    if not line_pages:
        raise SystemExit("No valid lines pages")

    out_dir = args.out_dir or (args.pattern_pdf.parent / "debug_outline")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.pattern_pdf.stem}_assembled"

    print(f"Pattern PDF   : {args.pattern_pdf}")
    print(f"Lines PDF     : {args.lines_pdf}")
    print(f"Pattern pages : {pat_pages}")
    print(f"Lines pages   : {line_pages}")
    print(f"Out dir       : {out_dir}")

    # ----- assemble both PDFs into one canvas -----
    print("\n=== Assembling pattern pages ===")
    assembled_pat = assemble_pages(
        pattern_doc, pat_pages,
        instr_doc, line_pages,
        overlap=args.overlap,
    )
    page = AssembledPage(assembled_pat["paths"], assembled_pat["global_rect"])

    print("\n=== Assembling lines / instructions pages ===")
    assembled_lines = assemble_pages(
        instr_doc, line_pages,
        instr_doc, line_pages,
        overlap=args.overlap,
    )
    page_lines = AssembledPage(
        assembled_lines["paths"], assembled_lines["global_rect"]
    )

    print("\n=== LINES – grain / fold ===")
    lines = step_lines_detect(
        page_lines, out_dir, stem, assembled=assembled_lines,
    )

    # ----- STEP 1 -----
    print("\n=== STEP 1 – read paths ===")
    records, all_segments = step1_read_and_draw(
        page,
        out_dir=out_dir,
        stem=stem,
        min_segment_length=args.min_seg_length,
    )
    print(f"  paths={len(records)}  segments={len(all_segments)}")

    # ----- STEP 2 -----
    print("\n=== STEP 2 – extend ===")
    ext_records, ext_segments, applied, hit_points = step2_extend_to_first_hit(
        records,
        out_dir=out_dir,
        stem=stem,
        max_extension=args.max_extension,
    )
    print(f"  extended_ends={len(applied)}  hit_points={len(hit_points)}")

    # ----- STEP 3 -----
    print("\n=== STEP 3 – junctions ===")
    junc_records, junc_segments = step3_snap_open_to_hits(
        ext_records,
        applied,
        hit_points,
        out_dir=out_dir,
        stem=stem,
        join_radius=args.join_radius,
        path_snap_radius=args.path_snap_radius,
    )
    print(f"  segments={len(junc_segments)}")

    # ----- STEP 4 -----
    print("\n=== STEP 4 – snap + union ===")
    noded, noded_segments = step4_snap_and_union(
        junc_records,
        out_dir=out_dir,
        stem=stem,
        snap_tol=args.snap_tol,
    )
    print(f"  noded segments={len(noded_segments)}")

    # ----- STEP 5 -----
    print("\n=== STEP 5 – polygonize ===")
    faces = step5_polygonize(
        noded, out_dir, stem, min_face_area=args.min_face_area,
    )
    print(f"  faces={len(faces)}")

    faces_per_piece = group_faces_into_pieces(faces)
    print(f"  pieces={len(faces_per_piece)}")
    for pi, fl in faces_per_piece:
        print(f"    piece {pi}: {len(fl)} face(s)")

    print("\n=== Interactive face selection ===")
    print("Hover highlights a piece. Click selects it.")
    print("Click again toggles an inner face. Alt-click drops the piece.")
    print("Drag empty background to pan. Wheel to zoom.")
    selected = select_faces_pyside6(
        faces_per_piece,
        lines,
        assembled=assembled_pat,
    )

    print(f"Selected outlines: {len(selected)}")
    for i, p in enumerate(selected):
        print(f"  [{i}] area={p['area']:.0f}  peri={p['perimeter']:.0f}")

    step_write_selected(selected, out_dir, stem)
    
    pattern_doc.close()
    instr_doc.close()
    print("\nDone.")


if __name__ == "__main__":
    main()


# TODO
