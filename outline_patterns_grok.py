#!/usr/bin/env python3
"""
Detect sewing pattern pieces and special lines (grain / fold).

Architecture:
  extract_paths()          → single source of truth
  detect_patterns_from_paths() → main outlines
  detect_faces_for_piece() → internal length / lining faces
  analyze_patterns()       → single entry point
"""

import argparse
from pathlib import Path
from collections import defaultdict
import fitz
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon as MplPolygon

from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import polygonize, unary_union, snap
from shapely.validation import make_valid
import math
from itertools import combinations


class AssembledPage:
    def __init__(self, paths, rect):
        self._paths = paths
        self.rect = rect
        self.mediabox = rect
        self.cropbox = rect
        self.rotation = 0

    def get_drawings(self):
        return self._paths


# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def path_to_segments(path):
    """High-density sampling of cubic Béziers (128 samples)."""
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
            # Dense sampling – critical for accurate length-option curves
            n = 128
            ts = np.linspace(0.0, 1.0, n)
            pts = []
            for t in ts:
                mt = 1.0 - t
                x = (mt**3)*p0.x + 3*(mt**2)*t*p1.x + 3*mt*(t**2)*p2.x + (t**3)*p3.x
                y = (mt**3)*p0.y + 3*(mt**2)*t*p1.y + 3*mt*(t**2)*p2.y + (t**3)*p3.y
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


def _ordered_points_from_segments(segs, dist_fn):
    if not segs:
        return []
    pts = [segs[0][0], segs[0][1]]
    for a, b in segs[1:]:
        if dist_fn(pts[-1], a) <= dist_fn(pts[-1], b):
            pts.append(b)
        else:
            pts.append(a)
    return pts


def _dedupe_cycles(cycles):
    out = []
    seen = []
    for c in cycles:
        key = frozenset(c["path_ids"])
        if not key:
            continue
        dup = False
        for s in seen:
            inter = len(key & s)
            if inter and inter >= 0.8 * min(len(key), len(s)):
                dup = True
                break
        if dup:
            for i, s in enumerate(seen):
                inter = len(key & s)
                if inter and inter >= 0.8 * min(len(key), len(s)):
                    if c["area"] > out[i]["area"]:
                        out[i] = c
                        seen[i] = key
                    break
            continue
        seen.append(key)
        out.append(c)
    return out


# ------------------------------------------------------------------
# Multi-page assembly (unchanged)
# ------------------------------------------------------------------
def _find_large_rectangles(page, min_area_ratio=0.55, min_side_ratio=0.60):
    drawings = page.get_drawings()
    page_area = page.rect.width * page.rect.height
    candidates = []
    for path in drawings:
        items = path.get("items", [])
        if not items:
            continue
        if len(items) == 1 and items[0][0] == "re":
            r = items[0][1]
            w, h = r.width, r.height
            if (w * h >= min_area_ratio * page_area and
                w >= min_side_ratio * page.rect.width and
                h >= min_side_ratio * page.rect.height):
                candidates.append(fitz.Rect(r))
    return candidates


def _detect_content_rect(doc, page_numbers):
    if not page_numbers:
        return None
    all_rects = []
    for pno in page_numbers:
        page = doc[pno]
        rects = _find_large_rectangles(page)
        if not rects:
            return None
        largest = max(rects, key=lambda r: r.width * r.height)
        all_rects.append(largest)
    widths  = [r.width  for r in all_rects]
    heights = [r.height for r in all_rects]
    if (max(widths) - min(widths) > 8.0 or max(heights) - min(heights) > 8.0):
        return None
    med_x0 = float(np.median([r.x0 for r in all_rects]))
    med_y0 = float(np.median([r.y0 for r in all_rects]))
    med_w  = float(np.median(widths))
    med_h  = float(np.median(heights))
    return fitz.Rect(med_x0, med_y0, med_x0 + med_w, med_y0 + med_h)


def assemble_pages(pattern_doc, pattern_pages, instr_doc, instr_pages, overlap=0.0):
    if not pattern_pages:
        raise ValueError("No pattern pages given")

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

    n = len(pattern_pages)
    horizontal_overflow = vertical_overflow = False
    ref_x0 = content_rect.x0 if content_rect else 0
    ref_y0 = content_rect.y0 if content_rect else 0
    ref_x1 = content_rect.x1 if content_rect else tile_w
    ref_y1 = content_rect.y1 if content_rect else tile_h

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

    if horizontal_overflow and not vertical_overflow:
        cols, rows = n, 1
        print("Detected horizontal extension: arranging pages in a single row")
    elif vertical_overflow and not horizontal_overflow:
        cols, rows = 1, n
        print("Detected vertical extension: arranging pages in a single column")
    else:
        sample = pattern_doc[pattern_pages[0]]
        if sample.rect.width >= sample.rect.height:
            cols, rows = n, 1
            print("No clear overflow detected, defaulting to horizontal layout")
        else:
            cols, rows = 1, n
            print("No clear overflow detected, defaulting to vertical layout")

    print(f"Using grid {rows}×{cols}")

    global_paths = []
    all_x, all_y = [], []

    for idx, pno in enumerate(pattern_pages):
        page = pattern_doc[pno]
        row = idx // cols
        col = idx % cols
        tx = col * (tile_w - overlap)
        ty = row * (tile_h - overlap)
        if content_rect is not None:
            tx -= content_rect.x0
            ty -= content_rect.y0
        mat = fitz.Matrix(1, 0, 0, 1, tx, ty)

        for path in page.get_drawings():
            new_items = []
            xs, ys = [], []
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
                    new_items.extend([("l", ul, ur), ("l", ur, lr), ("l", lr, ll), ("l", ll, ul)])
                    xs.extend([ul.x, ur.x, lr.x, ll.x])
                    ys.extend([ul.y, ur.y, lr.y, ll.y])
                else:
                    new_items.append(item)

            new_path = dict(path)
            new_path["items"] = new_items
            if xs and ys:
                new_path["rect"] = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
            else:
                new_path["rect"] = fitz.Rect(tx, ty, tx + tile_w, ty + tile_h)
            global_paths.append(new_path)
            all_x.extend(xs)
            all_y.extend(ys)

    if not all_x:
        global_rect = fitz.Rect(0, 0, tile_w, tile_h)
    else:
        pad = 30.0
        global_rect = fitz.Rect(min(all_x)-pad, min(all_y)-pad, max(all_x)+pad, max(all_y)+pad)

    print(f"Final canvas: {global_rect.width:.1f} × {global_rect.height:.1f}")
    return {
        "paths": global_paths,
        "global_rect": global_rect,
        "content_rect": content_rect,
        "tile_w": tile_w,
        "tile_h": tile_h,
        "grid": (cols, rows),
    }


# ------------------------------------------------------------------
# Shared path extraction – SINGLE SOURCE OF TRUTH
# ------------------------------------------------------------------
def extract_paths(page):
    paths = []
    for idx, path in enumerate(page.get_drawings()):
        color = path.get("color")
        width = float(path.get("width") or 0.0)
        if color is None and width <= 0:
            continue
        segs = path_to_segments(path)
        if not segs:
            continue
        length = sum(dist(a, b) for a, b in segs)
        if length < 1.0:
            continue

        def color_key(c):
            if not c:
                return None
            return tuple(round(float(x), 3) for x in c[:3])

        paths.append({
            "id": idx,
            "color": color_key(color),
            "width": round(width, 2),
            "segments": segs,
            "length": length,
            "style": (color_key(color),),
            "raw": path,
        })
    return paths


def _filter_nested_pieces(pieces, containment_ratio=0.85):
    """
    Keep only the outermost pieces.
    A piece is discarded if its centroid lies inside a significantly larger piece.
    This automatically demotes length-option outlines that sit inside a main outline.
    """
    if not pieces:
        return []

    # largest first
    pieces = sorted(pieces, key=lambda p: p["area"], reverse=True)
    kept = []

    for cand in pieces:
        pts = cand.get("points") or []
        if len(pts) < 3:
            continue
        try:
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = make_valid(poly)
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda g: g.area)
            cen = poly.centroid
        except Exception:
            kept.append(cand)
            continue

        is_nested = False
        for larger in kept:
            try:
                lpts = larger.get("points") or []
                if len(lpts) < 3:
                    continue
                lpoly = Polygon(lpts)
                if not lpoly.is_valid:
                    lpoly = make_valid(lpoly)
                if lpoly.geom_type == "MultiPolygon":
                    lpoly = max(lpoly.geoms, key=lambda g: g.area)

                if lpoly.contains(cen) or lpoly.intersects(cen):
                    # also require that the candidate is substantially smaller
                    if cand["area"] < larger["area"] * containment_ratio:
                        is_nested = True
                        break
            except Exception:
                continue

        if not is_nested:
            kept.append(cand)

    return kept


# ------------------------------------------------------------------
# Main outline detection (graph-based, on shared paths)
# ------------------------------------------------------------------
def detect_patterns_from_paths(paths,
                               gap_threshold=25.0,
                               point_tol=1.0,
                               min_perimeter=250.0,
                               min_polygon_area=8000.0):
    from collections import defaultdict

    def poly_area(pts):
        if len(pts) < 3:
            return 0.0
        a = 0.0
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            a += x1 * y2 - x2 * y1
        return abs(a) * 0.5

    def bbox_of_segs(segs):
        xs = [p[0] for s in segs for p in s]
        ys = [p[1] for s in segs for p in s]
        if not xs:
            return (0.0, 0.0, 0.0, 0.0)
        return (min(xs), min(ys), max(xs), max(ys))

    def path_chain_terminals(p):
        items = p["raw"].get("items", [])
        if not items:
            return None, None
        t_start = t_end = None
        it0, itn = items[0], items[-1]
        try:
            if it0[0] == "l":
                t_start = (it0[1].x, it0[1].y)
            elif it0[0] == "c":
                t_start = (it0[1].x, it0[1].y)
            elif it0[0] == "re":
                r = it0[1]
                t_start = (r.x0, r.y0)
            elif it0[0] == "qu":
                t_start = (it0[1].ul.x, it0[1].ul.y)
        except Exception:
            pass
        try:
            if itn[0] == "l":
                t_end = (itn[2].x, itn[2].y)
            elif itn[0] == "c":
                t_end = (itn[4].x, itn[4].y)
            elif itn[0] == "re":
                r = itn[1]
                t_end = (r.x0, r.y0)
            elif itn[0] == "qu":
                t_end = (itn[1].ul.x, itn[1].ul.y)
        except Exception:
            pass
        return t_start, t_end

    for p in paths:
        t0, t1 = path_chain_terminals(p)
        if t0 is None or t1 is None:
            t0, t1 = p["segments"][0][0], p["segments"][-1][1]
        p["t0"], p["t1"] = t0, t1

    print(f"Pattern paths used: {len(paths)}")
    if not paths:
        return []

    terms = []
    for i, p in enumerate(paths):
        terms.append((i, 0, p["t0"], p["style"]))
        terms.append((i, 1, p["t1"], p["style"]))

    parent = list(range(len(terms)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    cell = max(gap_threshold * 1.5, 4.0)
    grid = defaultdict(list)
    for ti, (pi, ei, xy, style) in enumerate(terms):
        grid[(style, int(xy[0]//cell), int(xy[1]//cell))].append(ti)

    neigh = [(dx, dy) for dx in (-1,0,1) for dy in (-1,0,1)]
    for ti, (pi, ei, xy, style) in enumerate(terms):
        cx, cy = int(xy[0]//cell), int(xy[1]//cell)
        for dx, dy in neigh:
            for tj in grid.get((style, cx+dx, cy+dy), []):
                if tj <= ti:
                    continue
                if dist(xy, terms[tj][2]) <= gap_threshold:
                    union(ti, tj)

    def node_id(i, end):
        return find(i*2 + end)

    edges = []
    for i, p in enumerate(paths):
        edges.append({
            "n0": node_id(i, 0),
            "n1": node_id(i, 1),
            "style": p["style"],
            "length": p["length"],
            "path_id": p["id"],
            "segments": p["segments"],
        })

    adj = defaultdict(list)
    for ei, e in enumerate(edges):
        adj[e["n0"]].append(ei)
        adj[e["n1"]].append(ei)

    def other_node(ei, node):
        e = edges[ei]
        return e["n1"] if e["n0"] == node else e["n0"]

    used_edges = set()

    def walk_chain(start_ei, start_node):
        chain = [start_ei]
        used_edges.add(start_ei)
        cur = other_node(start_ei, start_node)
        while True:
            cands = [ei for ei in adj[cur] if ei not in used_edges]
            if len(cands) != 1:
                break
            nxt = cands[0]
            if edges[nxt]["style"] != edges[chain[0]]["style"]:
                break
            used_edges.add(nxt)
            chain.append(nxt)
            cur = other_node(nxt, cur)
            if cur == start_node:
                break
        return chain, cur

    chains = []
    for ei in range(len(edges)):
        if ei in used_edges:
            continue
        e = edges[ei]
        chain_edges, _ = walk_chain(ei, e["n0"])
        nodes = [e["n0"]]
        cur = e["n0"]
        segs, pids, length = [], [], 0.0
        for ce in chain_edges:
            ee = edges[ce]
            segs.extend(ee["segments"])
            pids.append(ee["path_id"])
            length += ee["length"]
            cur = other_node(ce, cur)
            nodes.append(cur)
        closed = (nodes[0] == nodes[-1] and len(chain_edges) >= 1)
        chains.append({
            "nodes": nodes, "closed": closed, "style": e["style"],
            "length": length, "segments": segs, "path_ids": pids,
            "start_node": nodes[0], "end_node": nodes[-1],
        })

    # Already-closed chains
    cycle_closed = []
    for ch in chains:
        if not ch["closed"]:
            continue
        pts = _ordered_points_from_segments(ch["segments"], dist)
        area = poly_area(pts) if len(pts) >= 3 else 0.0
        cycle_closed.append({
            "path_ids": list(ch["path_ids"]),
            "segments": list(ch["segments"]),
            "length": ch["length"],
            "area": area,
            "bbox": bbox_of_segs(ch["segments"]),
            "style": ch["style"],
            "closed": True,
            "points": pts,
        })

    # Multi-chain cycles (DFS)
    chain_adj = defaultdict(list)
    for ci, ch in enumerate(chains):
        if ch["closed"]:
            continue
        chain_adj[ch["start_node"]].append(ci)
        chain_adj[ch["end_node"]].append(ci)

    def chain_other(ci, node):
        ch = chains[ci]
        return ch["end_node"] if ch["start_node"] == node else ch["start_node"]

    used_chain = set()
    for start_ci, ch0 in enumerate(chains):
        if ch0["closed"] or start_ci in used_chain:
            continue
        start = ch0["start_node"]
        stack = [(ch0["end_node"], [start_ci], {start_ci})]
        found = None
        while stack:
            node, path_c, vis = stack.pop()
            if node == start and len(path_c) >= 2:
                found = path_c
                break
            if len(path_c) > 80:
                continue
            for nci in chain_adj.get(node, []):
                if nci in vis or chains[nci]["style"] != ch0["style"]:
                    continue
                stack.append((chain_other(nci, node), path_c+[nci], vis|{nci}))
        if found:
            segs, pids, length = [], [], 0.0
            for ci in found:
                ch = chains[ci]
                segs.extend(ch["segments"])
                pids.extend(ch["path_ids"])
                length += ch["length"]
            pts = _ordered_points_from_segments(segs, dist)
            area = poly_area(pts) if len(pts) >= 3 else 0.0
            if length >= min_perimeter * 0.5:
                cycle_closed.append({
                    "path_ids": pids, "segments": segs, "length": length,
                    "area": area, "bbox": bbox_of_segs(segs),
                    "style": ch0["style"], "closed": True, "points": pts,
                })
                used_chain.update(found)

    # Near-cycles + last-chance merge
    node_xy = defaultdict(list)
    for ti, (_, _, xy, _) in enumerate(terms):
        node_xy[find(ti)].append(xy)

    def node_point(nid):
        arr = node_xy.get(nid) or [(0.,0.)]
        return (sum(p[0] for p in arr)/len(arr), sum(p[1] for p in arr)/len(arr))

    for ci, ch in enumerate(chains):
        if ch["closed"] or ci in used_chain:
            continue
        if dist(node_point(ch["start_node"]), node_point(ch["end_node"])) <= gap_threshold * 4:
            pts = _ordered_points_from_segments(ch["segments"], dist)
            area = poly_area(pts) if len(pts) >= 3 else 0.0
            cycle_closed.append({
                "path_ids": list(ch["path_ids"]), "segments": list(ch["segments"]),
                "length": ch["length"], "area": area,
                "bbox": bbox_of_segs(ch["segments"]),
                "style": ch["style"], "closed": True, "points": pts,
            })
            used_chain.add(ci)

    open_chains = [(ci, ch) for ci, ch in enumerate(chains)
                   if not ch["closed"] and ci not in used_chain]
    for (ci1, ch1), (ci2, ch2) in combinations(open_chains, 2):
        if ch1["style"] != ch2["style"]:
            continue
        ends1 = [node_point(ch1["start_node"]), node_point(ch1["end_node"])]
        ends2 = [node_point(ch2["start_node"]), node_point(ch2["end_node"])]
        if min(dist(a,b) for a in ends1 for b in ends2) <= gap_threshold * 5:
            segs = ch1["segments"] + ch2["segments"]
            pids = ch1["path_ids"] + ch2["path_ids"]
            pts = _ordered_points_from_segments(segs, dist)
            area = poly_area(pts) if len(pts) >= 3 else 0.0
            length = ch1["length"] + ch2["length"]
            if length >= min_perimeter * 0.5:
                cycle_closed.append({
                    "path_ids": pids, "segments": segs, "length": length,
                    "area": area, "bbox": bbox_of_segs(segs),
                    "style": ch1["style"], "closed": True, "points": pts,
                })
                used_chain.add(ci1)
                used_chain.add(ci2)

    mains = [c for c in cycle_closed
             if c["length"] >= min_perimeter and c["area"] >= min_polygon_area]
    mains = _dedupe_cycles(mains)
    mains.sort(key=lambda c: c["area"], reverse=True)

    pieces = []
    for m in mains:
        pieces.append({
            "segments": m["segments"],
            "points": m.get("points") or _ordered_points_from_segments(m["segments"], dist),
            "bbox": m["bbox"],
            "area": m["area"],
            "perimeter": m["length"],
            "path_ids": list(dict.fromkeys(m["path_ids"])),
            "closed": True,
            "style": m.get("style"),
            "size_variants": [],
        })

    print(f"Pattern pieces kept: {len(pieces)}")
    for i, p in enumerate(pieces):
        print(f"  [{i}] paths={len(p['path_ids'])} area={p['area']:.0f}")

    # Remove nested length-option outlines that were promoted to main pieces
    pieces = _filter_nested_pieces(pieces)
    print(f"After nested-filter: {len(pieces)} pieces kept")
    for i, p in enumerate(pieces):
        print(f"  [{i}] paths={len(p['path_ids'])} area={p['area']:.0f}")
    return pieces


# ------------------------------------------------------------------
# Adaptive end-only extension
# ------------------------------------------------------------------
def _extend_to_boundary(ls, outer, max_extend=25.0, step=1.0):
    coords = list(ls.coords)
    if len(coords) < 2:
        return ls

    def extend_end(pt, direction):
        p = np.asarray(pt, dtype=float)
        d = np.asarray(direction, dtype=float)
        n = np.linalg.norm(d)
        if n < 1e-9:
            return tuple(p)
        d /= n
        if not outer.contains(Point(p)) and outer.distance(Point(p)) > 0.8:
            return tuple(p)
        for dist_ in np.arange(step, max_extend + step, step):
            cand = p + d * dist_
            if not outer.contains(Point(cand)):
                return tuple(p + d * (dist_ + 0.5))
        return tuple(p + d * max_extend)

    dir0 = np.asarray(coords[0]) - np.asarray(coords[1])
    dir1 = np.asarray(coords[-1]) - np.asarray(coords[-2])
    new_start = extend_end(coords[0], dir0)
    new_end   = extend_end(coords[-1], dir1)
    return LineString([new_start] + coords[1:-1] + [new_end])


# ------------------------------------------------------------------
# Face detection – works on whole paths from the shared database
# ------------------------------------------------------------------
def detect_faces_for_piece(main_piece, paths,
                           min_face_area=400.0,
                           snap_tol=2.0,
                           inside_tol=3.0):
    pts = main_piece.get("points")
    if not pts or len(pts) < 3:
        pts = _ordered_points_from_segments(main_piece["segments"], dist)
    if len(pts) < 3:
        return []

    try:
        outer = Polygon(pts)
        if not outer.is_valid:
            outer = make_valid(outer)
        if outer.geom_type == "MultiPolygon":
            outer = max(outer.geoms, key=lambda g: g.area)
        if outer.is_empty or outer.area < 1.0:
            return []
    except Exception:
        return []

    main_path_ids = set(main_piece.get("path_ids", []))
    main_style = main_piece.get("style")

    candidates = []
    for p in paths:
        # Never use paths that belong to the main outline
        if p["id"] in main_path_ids:
            continue
        # Never use any path of the same style/colour
        if main_style is not None and p.get("style") == main_style:
            continue

        segs = p["segments"]
        if not segs:
            continue

        # Build one continuous polyline
        coords = [segs[0][0]]
        for a, b in segs:
            if dist(coords[-1], a) <= dist(coords[-1], b):
                coords.append(b)
            else:
                coords.append(a)
        clean = [coords[0]]
        for pt in coords[1:]:
            if dist(clean[-1], pt) > 1e-6:
                clean.append(pt)
        if len(clean) < 2:
            continue

        ls = LineString(clean)
        if ls.length < 1.0:
            continue

        if (outer.intersects(ls) or
            outer.distance(ls) <= inside_tol or
            outer.buffer(inside_tol).intersects(ls)):
            candidates.append(ls)

    print(f"  → {len(candidates)} candidate internal polylines")

    if not candidates:
        coords = list(outer.exterior.coords)
        return [{
            "polygon": outer,
            "segments": [(coords[i], coords[i+1]) for i in range(len(coords)-1)],
            "points": coords[:-1],
            "area": float(outer.area),
            "bbox": outer.bounds,
        }]

    boundary = LineString(list(outer.exterior.coords))
    cutter_lines = []
    for ls in candidates:
        coords = list(ls.coords)
        is_closed = len(coords) >= 3 and dist(coords[0], coords[-1]) <= snap_tol
        if is_closed:
            if dist(coords[0], coords[-1]) > 1e-6:
                coords.append(coords[0])
            cutter_lines.append(LineString(coords))
        else:
            cutter_lines.append(_extend_to_boundary(ls, outer, max_extend=10.0, step=1.0))

    network = MultiLineString([boundary] + cutter_lines)
    try:
        network = snap(network, network, snap_tol)
    except Exception:
        pass

    noded = unary_union(network)
    raw_faces = list(polygonize(noded))

    result = []
    for face in raw_faces:
        if face.is_empty or face.area < min_face_area:
            continue
        try:
            cen = face.centroid
            if not (outer.contains(cen) or outer.touches(cen)):
                continue
            inter = face.intersection(outer)
            if inter.is_empty:
                continue
            if inter.geom_type == "MultiPolygon":
                inter = max(inter.geoms, key=lambda g: g.area)
            if inter.geom_type != "Polygon" or inter.area < min_face_area:
                continue
            face = inter
        except Exception:
            continue

        coords = list(face.exterior.coords)
        if len(coords) < 3:
            continue
        result.append({
            "polygon": face,
            "segments": [(coords[i], coords[i+1]) for i in range(len(coords)-1)],
            "points": coords[:-1],
            "area": float(face.area),
            "bbox": face.bounds,
        })

    result.sort(key=lambda f: f["area"], reverse=True)
    return result


# ------------------------------------------------------------------
# Single entry point
# ------------------------------------------------------------------
def analyze_patterns(page,
                     gap_threshold=30.0,
                     min_perimeter=250.0,
                     min_polygon_area=8000.0,
                     min_face_area=400.0,
                     snap_tol=2.0,
                     inside_tol=3.0):
    paths = extract_paths(page)
    pieces = detect_patterns_from_paths(
        paths,
        gap_threshold=gap_threshold,
        min_perimeter=min_perimeter,
        min_polygon_area=min_polygon_area,
    )
    for i, piece in enumerate(pieces):
        print(f"Detecting faces for piece {i} ...")
        piece["faces"] = detect_faces_for_piece(
            piece, paths,
            min_face_area=min_face_area,
            snap_tol=snap_tol,
            inside_tol=inside_tol,
        )
        print(f"  → {len(piece['faces'])} face(s)")
    return paths, pieces


# ------------------------------------------------------------------
# Special lines (grain/fold) – unchanged from previous working version
# ------------------------------------------------------------------
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
    from collections import defaultdict
    drawings = page.get_drawings()

    def quant(p): return (round(p[0]/point_tol), round(p[1]/point_tol))
    def sub(a,b): return (a[0]-b[0], a[1]-b[1])
    def nrm(v):
        l = math.hypot(v[0], v[1])
        return (0.,0.) if l < 1e-12 else (v[0]/l, v[1]/l)
    def dot(u,v): return u[0]*v[0] + u[1]*v[1]

    segments = []
    arrow_paths = []
    for path_idx, path in enumerate(drawings):
        width = float(path.get("width") or 0.0)
        color = path.get("color")
        items = path.get("items", [])
        r = path.get("rect")
        segs_all = path_to_segments(path)
        if r is not None and segs_all:
            size = max(r.width, r.height)
            if size <= max_arrow_size and len(segs_all) <= max_arrow_segments and not (0 < width < min_arrow_width):
                arrow_paths.append({
                    "path": path, "segments": segs_all,
                    "center": ((r.x0+r.x1)/2, (r.y0+r.y1)/2), "path_id": path_idx,
                })
        for item in items:
            if item[0] != "l": continue
            p1, p2 = item[1], item[2]
            a, b = (p1.x, p1.y), (p2.x, p2.y)
            length = dist(a, b)
            if length < 1.0: continue
            segments.append({
                "a": a, "b": b, "length": length, "dir": nrm(sub(b,a)),
                "width": round(width,2),
                "color": tuple(round(float(c),3) for c in color[:3]) if color else None,
                "path_id": path_idx,
            })

    if not segments:
        return []

    parent = list(range(len(segments)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i,j):
        ri, rj = find(i), find(j)
        if ri != rj: parent[rj] = ri

    cell = max(join_gap*1.5, 4.0)
    grid = defaultdict(list)
    def cell_of(p): return (int(p[0]//cell), int(p[1]//cell))
    for i,s in enumerate(segments):
        grid[cell_of(s["a"])].append((i,"a"))
        grid[cell_of(s["b"])].append((i,"b"))

    for i,s in enumerate(segments):
        for ep_name, ep in (("a",s["a"]),("b",s["b"])):
            cx,cy = cell_of(ep)
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    for j, other in grid.get((cx+dx,cy+dy),[]):
                        if j <= i: continue
                        t = segments[j]
                        if s["width"] != t["width"] or s["color"] != t["color"]: continue
                        q = t[other]
                        if dist(ep,q) > join_gap: continue
                        if abs(dot(s["dir"], t["dir"])) < collinear_dot_min: continue
                        union(i,j)

    groups = defaultdict(list)
    for i in range(len(segments)):
        groups[find(i)].append(i)

    runs = []
    for idxs in groups.values():
        segs = [segments[i] for i in idxs]
        length = sum(s["length"] for s in segs)
        longest = max(segs, key=lambda s: s["length"])
        counts = defaultdict(int)
        sample = {}
        for s in segs:
            for p in (s["a"], s["b"]):
                qp = quant(p)
                counts[qp] += 1
                sample[qp] = p
        terminals = [sample[q] for q,c in counts.items() if c == 1]
        if len(terminals) < 2:
            pts = [p for s in segs for p in (s["a"],s["b"])]
            best = -1
            t0 = t1 = pts[0]
            for i in range(len(pts)):
                for j in range(i+1,len(pts)):
                    d = dist(pts[i],pts[j])
                    if d > best:
                        best = d
                        t0,t1 = pts[i],pts[j]
            terminals = [t0,t1]
        elif len(terminals) > 2:
            best = -1
            t0,t1 = terminals[0],terminals[1]
            for i in range(len(terminals)):
                for j in range(i+1,len(terminals)):
                    d = dist(terminals[i],terminals[j])
                    if d > best:
                        best = d
                        t0,t1 = terminals[i],terminals[j]
            terminals = [t0,t1]
        runs.append({
            "seg_idxs": list(idxs),
            "segments": [(s["a"],s["b"]) for s in segs],
            "max_l": max(s["length"] for s in segs),
            "length": length,
            "direction": longest["dir"],
            "terminals": (terminals[0], terminals[1]),
            "width": longest["width"],
            "color": longest["color"],
            "path_ids": sorted({s["path_id"] for s in segs}),
        })

    long_runs = sorted([r for r in runs if r["max_l"] >= min_shaft_length],
                       key=lambda r: r["max_l"], reverse=True)
    short_runs = [r for r in runs if min_crossbar_length <= r["length"] <= max_crossbar_length]

    def arrow_hits(ap, point, radius):
        if dist(ap["center"], point) <= radius: return True
        for a,b in ap["segments"]:
            if dist(a,point) <= radius or dist(b,point) <= radius: return True
        return False

    def arrow_near(point, radius, shaft_dir=None):
        for ap in arrow_paths:
            if not arrow_hits(ap, point, radius): continue
            if shaft_dir is not None:
                # simple angled check
                has_angle = False
                for a,b in ap["segments"]:
                    d = nrm(sub(b,a))
                    if d == (0.,0.): continue
                    ang = abs(dot(shaft_dir, d))
                    ang = max(0., min(1., ang))
                    deg = math.degrees(math.acos(ang))
                    if 10 <= deg <= 80:
                        has_angle = True
                        break
                if not has_angle: continue
            return True
        return False

    def arrows_for(point, radius, shaft_dir=None):
        out = []
        for ap in arrow_paths:
            if not arrow_hits(ap, point, radius): continue
            if shaft_dir is not None:
                has_angle = False
                for a,b in ap["segments"]:
                    d = nrm(sub(b,a))
                    if d == (0.,0.): continue
                    ang = abs(dot(shaft_dir, d))
                    ang = max(0., min(1., ang))
                    deg = math.degrees(math.acos(ang))
                    if 10 <= deg <= 80:
                        has_angle = True
                        break
                if not has_angle: continue
            out.append(ap)
        return out

    results = []
    used = set()
    for L in long_runs:
        if set(L["seg_idxs"]) & used: continue
        ep1, ep2 = L["terminals"]
        ldir = L["direction"]

        grain_ends, grain_arrows = [], []
        for ep in (ep1, ep2):
            if arrow_near(ep, arrow_search_radius, ldir):
                grain_ends.append(ep)
                grain_arrows.extend(arrows_for(ep, arrow_search_radius, ldir))

        fold_ends, fold_bars, fold_arrows = 0, [], []
        for ep in (ep1, ep2):
            best = None
            for S in short_runs:
                if set(S["seg_idxs"]) & set(L["seg_idxs"]) or set(S["seg_idxs"]) & used: continue
                if S["width"] != L["width"]: continue
                if abs(dot(ldir, S["direction"])) > perp_dot_max: continue
                s0,s1 = S["terminals"]
                d0,d1 = dist(s0,ep), dist(s1,ep)
                if min(d0,d1) > join_gap: continue
                inner, outer, idist = (s0,s1,d0) if d0<=d1 else (s1,s0,d1)
                if not arrow_near(outer, arrow_search_radius, ldir): continue
                arr = arrows_for(outer, arrow_search_radius, ldir)
                if best is None or idist < best[0]:
                    best = (idist, S, outer, arr)
            if best:
                fold_ends += 1
                fold_bars.append(best[1])
                fold_arrows.extend(best[3])

        is_fold = fold_ends >= min_fold_ends
        is_grain = len(grain_ends) >= 1
        if not is_fold and not is_grain: continue

        if is_fold:
            line_type, arrows, score = "fold", fold_arrows, 100 + fold_ends*10 + len(fold_arrows)
            used.update(L["seg_idxs"])
            for b in fold_bars: used.update(b["seg_idxs"])
        else:
            line_type, arrows, score = "grain", grain_arrows, len(grain_ends)*10 + len(grain_arrows)
            used.update(L["seg_idxs"])

        uniq = {id(a["path"]): a for a in arrows}
        arrows = list(uniq.values())
        extra = []
        if is_fold:
            for b in fold_bars: extra.extend(b["segments"])
        for a in arrows: extra.extend(a["segments"])

        results.append({
            "type": line_type,
            "shaft": {"segments": L["segments"], "length": L["length"],
                      "endpoints": L["terminals"], "num_segments": len(L["segments"]),
                      "path_ids": L["path_ids"]},
            "arrows": arrows,
            "crossbars": fold_bars if is_fold else [],
            "score": score,
            "segments": L["segments"],
            "all_segments": L["segments"] + extra,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ------------------------------------------------------------------
# Output helpers
# ------------------------------------------------------------------
def write_patterns_txt(pieces, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Pattern pieces: {len(pieces)}\n\n")
        for i, p in enumerate(pieces):
            f.write(f"PIECE {i}\n")
            f.write(f"  area      : {p.get('area',0):.1f}\n")
            f.write(f"  perimeter : {p.get('perimeter',0):.1f}\n")
            f.write(f"  bbox      : {p.get('bbox')}\n")
            f.write(f"  path_ids  : {p.get('path_ids',[])}\n")
            f.write(f"  segments  : {len(p.get('segments',[]))}\n")
            f.write("\n")
    print(f"Wrote patterns → {out_path}")


def write_lines_txt(lines, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Special lines: {len(lines)}\n\n")
        for i, obj in enumerate(lines):
            f.write(f"LINE {i}  type={obj['type']}\n")
            f.write(f"  score        : {obj['score']}\n")
            f.write(f"  shaft_length : {obj['shaft']['length']:.1f}\n")
            f.write("\n")
    print(f"Wrote lines → {out_path}")


# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------
def render(page, objects, mode, out_path: Path, assembled=None):
    if assembled is not None:
        gr = assembled["global_rect"]
        fig_w = 16
        fig_h = max(8.0, fig_w * gr.height / max(gr.width, 1.0))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_xlim(gr.x0-10, gr.x1+10)
        ax.set_ylim(gr.y1+10, gr.y0-10)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{mode.upper()} – assembled ({assembled['grid'][0]}×{assembled['grid'][1]})")
    else:
        pr = page.rect
        fig, ax = plt.subplots(figsize=(12, 12*pr.height/pr.width))
        ax.set_xlim(pr.x0, pr.x1)
        ax.set_ylim(pr.y1, pr.y0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{mode.upper()} detection")
        pix = page.get_pixmap(matrix=fitz.Matrix(1.4,1.4), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        ax.imshow(img, extent=[pr.x0,pr.x1,pr.y1,pr.y0], alpha=0.40, zorder=0)

    colors = plt.cm.tab10.colors
    if mode == "patterns":
        for i, piece in enumerate(objects):
            c = colors[i % len(colors)]
            if piece.get("segments"):
                ax.add_collection(LineCollection(piece["segments"], colors=[c], linewidths=2.0, alpha=0.95, zorder=3))
            x0,y0,x1,y1 = piece["bbox"]
            ax.add_patch(Rectangle((x0,y0), x1-x0, y1-y0, fill=False, edgecolor=c, linestyle="--", linewidth=1, alpha=0.7, zorder=1))
            ax.text(x0+4, y1-4, f"{i}", color=c, fontsize=11, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1), zorder=6)
    else:
        for i, obj in enumerate(objects):
            c = "red" if obj["type"]=="grain" else "purple"
            ax.add_collection(LineCollection(obj["all_segments"], colors=c, linewidths=2.2, alpha=0.9, zorder=3))
            for ep in obj["shaft"]["endpoints"]:
                ax.plot(ep[0], ep[1], "o", color="orange", markersize=6, zorder=4)
            ax.text(obj["shaft"]["endpoints"][0][0], obj["shaft"]["endpoints"][0][1],
                    f"{i}:{obj['type'][0].upper()}", color=c, fontsize=9, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"), zorder=5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved PNG → {out_path}")


def render_faces(page, faces_per_piece, out_path: Path, assembled=None):
    if assembled is not None:
        gr = assembled["global_rect"]
        fig_w = 16
        fig_h = max(8.0, fig_w * gr.height / max(gr.width, 1.0))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_xlim(gr.x0-10, gr.x1+10)
        ax.set_ylim(gr.y1+10, gr.y0-10)
    else:
        pr = page.rect
        fig, ax = plt.subplots(figsize=(12, 12*pr.height/pr.width))
        ax.set_xlim(pr.x0, pr.x1)
        ax.set_ylim(pr.y1, pr.y0)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.2,1.2), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        ax.imshow(img, extent=[pr.x0,pr.x1,pr.y1,pr.y0], alpha=0.35, zorder=0)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Atomic faces")

    colors = plt.cm.tab20.colors
    fid = 0
    for _, faces in faces_per_piece:
        for face in faces:
            c = colors[fid % len(colors)]
            fid += 1
            ax.add_patch(MplPolygon(face["points"], closed=True, facecolor=c,
                                    edgecolor="black", linewidth=1.2, alpha=0.45, zorder=3))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved faces PNG → {out_path}")


# ------------------------------------------------------------------
# Interactive selector
# ------------------------------------------------------------------
class FaceSelector:
    def __init__(self, page, faces_per_piece, assembled=None, title="Select faces – click to toggle"):
        self.faces_per_piece = faces_per_piece
        self.done = self.cancelled = False
        self.flat_faces = [(pi, fi, face) for pi, faces in faces_per_piece for fi, face in enumerate(faces)]
        self.selected = {(pi, fi) for pi, fi, _ in self.flat_faces}

        n_pieces = max((pi for pi,_,_ in self.flat_faces), default=0) + 1
        try:
            cmap = mpl.colormaps["tab20"]
        except Exception:
            cmap = plt.cm.get_cmap("tab20")
        self.piece_colors = {pi: cmap(pi % 20) for pi in range(max(n_pieces,1))}

        if assembled is not None:
            gr = assembled["global_rect"]
            fig_w = 16
            fig_h = max(8., fig_w * gr.height / max(gr.width,1.))
            self.fig, self.ax = plt.subplots(figsize=(fig_w, fig_h))
            self.ax.set_xlim(gr.x0-10, gr.x1+10)
            self.ax.set_ylim(gr.y1+10, gr.y0-10)
        else:
            pr = page.rect
            self.fig, self.ax = plt.subplots(figsize=(12, 12*pr.height/pr.width))
            self.ax.set_xlim(pr.x0, pr.x1)
            self.ax.set_ylim(pr.y1, pr.y0)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.2,1.2), alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            self.ax.imshow(img, extent=[pr.x0,pr.x1,pr.y1,pr.y0], alpha=0.35, zorder=0)

        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.ax.set_title(title, pad=2, fontsize=11)

        self.patches = {}
        self.combined_outlines = {}
        for pi, fi, face in self.flat_faces:
            poly = MplPolygon(face["points"], closed=True, facecolor=self.piece_colors[pi],
                              edgecolor="0.45", linewidth=0.7, alpha=0.62, zorder=3)
            self.ax.add_patch(poly)
            self.patches[(pi,fi)] = poly

        for pi,fi,_ in self.flat_faces:
            self._update_face_style(pi, fi)
        self._update_all_combined_outlines()

        self.status = self.ax.text(0.01, 0.995, self._status_text(), transform=self.ax.transAxes,
                                   fontsize=9, va="top", ha="left",
                                   bbox=dict(facecolor="white", alpha=0.88, edgecolor="none", pad=3), zorder=10)

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.tight_layout(pad=0.4)
        self.fig.subplots_adjust(top=0.96)

    def _status_text(self):
        return f"Selected: {len(self.selected)}/{len(self.flat_faces)}   click=toggle   Enter/d=done   a=all   c=clear   Esc=cancel"

    def _update_face_style(self, pi, fi):
        poly = self.patches[(pi,fi)]
        sel = (pi,fi) in self.selected
        poly.set_facecolor(self.piece_colors[pi])
        poly.set_edgecolor("0.45")
        poly.set_linewidth(0.7 if sel else 0.5)
        poly.set_alpha(0.62 if sel else 0.18)
        poly.set_zorder(3 if sel else 2)

    def _compute_combined_polygon(self, piece_idx):
        polys = [face["polygon"] for pi,fi,face in self.flat_faces
                 if pi == piece_idx and (pi,fi) in self.selected]
        if not polys:
            return None
        try:
            u = unary_union(polys)
            if u.is_empty: return None
            if u.geom_type == "MultiPolygon":
                u = max(u.geoms, key=lambda g: g.area)
            return u if u.geom_type == "Polygon" else None
        except Exception:
            return None

    def _update_combined_outline(self, piece_idx):
        if piece_idx in self.combined_outlines:
            for a in self.combined_outlines[piece_idx]:
                a.remove()
            del self.combined_outlines[piece_idx]
        poly = self._compute_combined_polygon(piece_idx)
        if poly is None or poly.exterior is None:
            return
        xs, ys = zip(*poly.exterior.coords)
        glow, = self.ax.plot(xs, ys, color="white", linewidth=7, solid_capstyle="round",
                             solid_joinstyle="round", alpha=0.45, zorder=6)
        core, = self.ax.plot(xs, ys, color="black", linewidth=3, solid_capstyle="round",
                             solid_joinstyle="round", alpha=0.95, zorder=7)
        self.combined_outlines[piece_idx] = [glow, core]

    def _update_all_combined_outlines(self):
        for pi in {pi for pi,_,_ in self.flat_faces}:
            self._update_combined_outline(pi)

    def _on_click(self, event):
        if event.inaxes != self.ax or event.button != 1: return
        x, y = event.xdata, event.ydata
        if x is None: return
        pt = Point(x, y)
        for pi, fi, face in reversed(self.flat_faces):
            if face["polygon"].contains(pt) or face["polygon"].touches(pt):
                key = (pi, fi)
                if key in self.selected:
                    self.selected.remove(key)
                else:
                    self.selected.add(key)
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
            self.selected = {(pi,fi) for pi,fi,_ in self.flat_faces}
            for pi,fi,_ in self.flat_faces: self._update_face_style(pi,fi)
            self._update_all_combined_outlines()
            self.status.set_text(self._status_text())
            self.fig.canvas.draw_idle()
        elif event.key == "c":
            self.selected.clear()
            for pi,fi,_ in self.flat_faces: self._update_face_style(pi,fi)
            self._update_all_combined_outlines()
            self.status.set_text(self._status_text())
            self.fig.canvas.draw_idle()

    def run(self):
        plt.show(block=True)
        if self.cancelled:
            return []
        from collections import defaultdict
        by_piece = defaultdict(list)
        for pi, fi, face in self.flat_faces:
            if (pi, fi) in self.selected:
                by_piece[pi].append(face["polygon"])
        result = []
        for pi, polys in sorted(by_piece.items()):
            try:
                u = unary_union(polys)
                if u.is_empty: continue
                if u.geom_type == "MultiPolygon":
                    u = max(u.geoms, key=lambda g: g.area)
                if u.geom_type != "Polygon" or u.exterior is None: continue
                coords = list(u.exterior.coords)
                result.append({
                    "segments": [(coords[i], coords[i+1]) for i in range(len(coords)-1)],
                    "points": coords[:-1],
                    "area": float(u.area),
                    "bbox": u.bounds,
                    "source_piece": pi,
                    "closed": True,
                    "size_variants": [],
                    "path_ids": [],
                    "perimeter": float(u.length),
                })
            except Exception:
                continue
        return result


def select_faces_interactive(page, faces_per_piece, assembled=None):
    return FaceSelector(page, faces_per_piece, assembled=assembled).run()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Sewing pattern detector")
    parser.add_argument("--pattern-pdf", type=Path, required=True)
    parser.add_argument("--lines-pdf", type=Path, required=True)
    parser.add_argument("--pattern-pages", type=str, default=None)
    parser.add_argument("--lines-pages", type=str, default=None)
    parser.add_argument("--overlap", type=float, default=0.0)
    args = parser.parse_args()

    pattern_doc = fitz.open(args.pattern_pdf)
    instr_doc   = fitz.open(args.lines_pdf)

    def parse_page_spec(spec, doc):
        if spec is None:
            return list(range(len(doc)))
        pages = []
        for part in spec.split(","):
            part = part.strip()
            if "-" in part:
                a, b = map(int, part.split("-"))
                pages.extend(range(a, b+1))
            else:
                pages.append(int(part))
        return sorted(set(p for p in pages if 0 <= p < len(doc)))

    pat_pages  = parse_page_spec(args.pattern_pages, pattern_doc)
    line_pages = parse_page_spec(args.lines_pages, instr_doc)

    if not pat_pages or not line_pages:
        raise SystemExit("No valid pages selected")

    print(f"Pattern pages : {pat_pages}")
    print(f"Lines pages   : {line_pages}")

    print("=== PATTERN PDF ===")
    assembled_pat = assemble_pages(pattern_doc, pat_pages, instr_doc, line_pages, overlap=args.overlap)

    print("\n=== LINES / INSTRUCTIONS PDF ===")
    assembled_lines = assemble_pages(instr_doc, line_pages, instr_doc, line_pages, overlap=args.overlap)

    assembled_page_pat   = AssembledPage(assembled_pat["paths"], assembled_pat["global_rect"])
    assembled_page_lines = AssembledPage(assembled_lines["paths"], assembled_lines["global_rect"])

    out_dir = args.pattern_pdf.parent
    stem = f"{args.pattern_pdf.stem}_assembled"

    print("\n=== Analysing patterns ===")
    paths, pieces = analyze_patterns(
        assembled_page_pat,
        gap_threshold=20.0,
        min_perimeter=250.0,
        min_polygon_area=8000.0,
        min_face_area=400.0,
        snap_tol=2.0,
        inside_tol=2.0,
    )

    print(f"Found {len(pieces)} pattern piece(s)")
    write_patterns_txt(pieces, out_dir / f"{stem}_patterns.txt")
    render(assembled_page_pat, pieces, "patterns",
           out_dir / f"{stem}_patterns.png", assembled=assembled_pat)

    faces_per_piece = [(i, p.get("faces", [])) for i, p in enumerate(pieces)]
    render_faces(assembled_page_pat, faces_per_piece,
                 out_dir / f"{stem}_faces.png", assembled=assembled_pat)

    print("\n=== Detecting special lines ===")
    lines = detect_special_lines(assembled_page_lines)
    print(f"Found {len(lines)} special line(s)")
    write_lines_txt(lines, out_dir / f"{stem}_lines.txt")
    render(assembled_page_lines, lines, "lines",
           out_dir / f"{stem}_lines.png", assembled=assembled_lines)

    print("\n=== Interactive face selection ===")
    print("Click faces to toggle.  Enter/d = done, a = all, c = clear, Esc = cancel")
    selected = select_faces_interactive(assembled_page_pat, faces_per_piece, assembled=assembled_pat)

    print(f"User selected {len(selected)} face(s)")
    if selected:
        write_patterns_txt(selected, out_dir / f"{stem}_selected_faces.txt")

    pattern_doc.close()
    instr_doc.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
