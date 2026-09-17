#!/usr/bin/env python3
"""
Detect sewing pattern pieces and special lines (grain / fold).

Outputs:
  - PNG visualisation
  - *_patterns.txt
  - *_lines.txt

Usage:
    # Simplest – uses all pages from both PDFs
    python outline_patterns.py \
        --pattern-pdf pieces.pdf \
        --lines-pdf instructions.pdf

    # With explicit page ranges
    python outline_patterns.py \
        --pattern-pdf pieces.pdf \
        --lines-pdf instructions.pdf \
        --pattern-pages 0-23 \
        --lines-pages 0-2        
"""

import argparse
from pathlib import Path
from collections import defaultdict, Counter
import fitz
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import polygonize, unary_union, split
from shapely.validation import make_valid
import math

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

# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _ordered_points_from_segments(segs, dist_fn=None):
    """
    Turn an unordered list of segments into an ordered point list
    that walks the chain as well as possible.
    """
    if not segs:
        return []

    if dist_fn is None:
        def dist_fn(a, b):
            return math.hypot(a[0]-b[0], a[1]-b[1])

    from collections import defaultdict

    def key(p):
        return (round(p[0], 3), round(p[1], 3))

    adj = defaultdict(list)
    pts = {}
    for a, b in segs:
        ka, kb = key(a), key(b)
        pts[ka] = a
        pts[kb] = b
        adj[ka].append(kb)
        adj[kb].append(ka)

    # Start from a degree-1 node if possible, otherwise any node
    start = None
    for k, nbrs in adj.items():
        if len(set(nbrs)) == 1:
            start = k
            break
    if start is None:
        start = next(iter(adj))

    ordered = []
    visited = set()
    cur = start
    prev = None

    while cur is not None and cur not in visited:
        visited.add(cur)
        ordered.append(pts[cur])
        candidates = [n for n in adj[cur] if n != prev]
        # Prefer unvisited
        candidates = [n for n in candidates if n not in visited] or candidates
        if not candidates:
            break
        # Take the first (or the one that continues the direction – simple version)
        prev, cur = cur, candidates[0]

    return ordered

def path_to_segments(path):
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
                x = (1-t)**3*p0.x + 3*(1-t)**2*t*p1.x + 3*(1-t)*t**2*p2.x + t**3*p3.x
                y = (1-t)**3*p0.y + 3*(1-t)**2*t*p1.y + 3*(1-t)*t**2*p2.y + t**3*p3.y
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


def segment_length(seg):
    (x1, y1), (x2, y2) = seg
    return np.hypot(x2 - x1, y2 - y1)


def point_distance(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])


def bbox_distance(b1, b2):
    dx = max(0.0, max(b1[0] - b2[2], b2[0] - b1[2]))
    dy = max(0.0, max(b1[1] - b2[3], b2[1] - b1[3]))
    return (dx*dx + dy*dy) ** 0.5


def segments_are_close(seg1, seg2, threshold):
    b1 = (min(seg1[0][0], seg1[1][0]), min(seg1[0][1], seg1[1][1]),
          max(seg1[0][0], seg1[1][0]), max(seg1[0][1], seg1[1][1]))
    b2 = (min(seg2[0][0], seg2[1][0]), min(seg2[0][1], seg2[1][1]),
          max(seg2[0][0], seg2[1][0]), max(seg2[0][1], seg2[1][1]))
    if bbox_distance(b1, b2) > threshold:
        return False
    for p in seg1:
        for q in seg2:
            if point_distance(p, q) <= threshold:
                return True
    return False


# ------------------------------------------------------------------
# Union-Find
# ------------------------------------------------------------------
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# ------------------------------------------------------------------
# Multi-page assembly helpers
# ------------------------------------------------------------------

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

# ------------------------------------------------------------------
# PATTERN detection
# ------------------------------------------------------------------
def detect_patterns(page,
                    gap_threshold=20.0,
                    point_tol=1.0,
                    min_perimeter=250.0,
                    min_polygon_area=8000.0):
    """
    Detect main closed (or near-closed) pattern outlines via geometric graph cycles.

    - Each stroked path is an edge between two terminals
    - Same-style terminals snap within gap_threshold
    - Closed outlines = cycles of any number of paths
    - Near-cycles allowed when chain ends are within gap
    """
    from collections import defaultdict
    import math

    drawings = page.get_drawings()

    def quant(p):
        return (round(p[0] / point_tol), round(p[1] / point_tol))

    def color_key(c):
        if not c:
            return None
        return tuple(round(float(x), 3) for x in c[:3])

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

    def path_chain_terminals(path):
        """Stable terminals: first operator start, last operator end."""
        items = path.get("items", [])
        if not items:
            return None, None
        t_start = t_end = None
        it0, itn = items[0], items[-1]
        op0, opn = it0[0], itn[0]
        try:
            if op0 == "l":
                t_start = (it0[1].x, it0[1].y)
            elif op0 == "c":
                t_start = (it0[1].x, it0[1].y)
            elif op0 == "re":
                r = it0[1]
                t_start = (r.x0, r.y0)
            elif op0 == "qu":
                q = it0[1]
                t_start = (q.ul.x, q.ul.y)
        except Exception:
            pass
        try:
            if opn == "l":
                t_end = (itn[2].x, itn[2].y)
            elif opn == "c":
                t_end = (itn[4].x, itn[4].y)
            elif opn == "re":
                r = itn[1]
                t_end = (r.x0, r.y0)
            elif opn == "qu":
                q = itn[1]
                t_end = (q.ul.x, q.ul.y)
        except Exception:
            pass
        return t_start, t_end

    # ------------------------------------------------------------------
    # 1) Collect stroked paths as candidate edges
    # ------------------------------------------------------------------
    paths = []
    for idx, path in enumerate(drawings):
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

        t0, t1 = path_chain_terminals(path)
        if t0 is None or t1 is None:
            t0, t1 = segs[0][0], segs[-1][1]

        paths.append({
            "id": idx,
            "color": color_key(color),
            "width": round(width, 2),
            "segments": segs,
            "t0": t0,
            "t1": t1,
            "length": length,
            "style": (color_key(color),),
        })

    n_paths = len(paths)
    print(f"Pattern paths used: {n_paths}")
    if not paths:
        return []

    # ------------------------------------------------------------------
    # 2) Same-style terminal snapping → graph nodes
    # ------------------------------------------------------------------
    terms = []
    for i, p in enumerate(paths):
        style = p["style"]
        terms.append((i, 0, p["t0"], style))
        terms.append((i, 1, p["t1"], style))

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
        grid[(style, int(xy[0] // cell), int(xy[1] // cell))].append(ti)

    neigh = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    for ti, (pi, ei, xy, style) in enumerate(terms):
        cx, cy = int(xy[0] // cell), int(xy[1] // cell)
        for dx, dy in neigh:
            for tj in grid.get((style, cx + dx, cy + dy), []):
                if tj <= ti:
                    continue
                if dist(xy, terms[tj][2]) <= gap_threshold:
                    union(ti, tj)

    def node_id(path_local, end_idx):
        return find(path_local * 2 + end_idx)

    edges = []
    for i, p in enumerate(paths):
        n0 = node_id(i, 0)
        n1 = node_id(i, 1)
        edges.append({
            "edge_idx": i,
            "path_local": i,
            "n0": n0,
            "n1": n1,
            "style": p["style"],
            "length": p["length"],
            "path_id": p["id"],
            "segments": p["segments"],
        })

    adj = defaultdict(list)
    for ei, e in enumerate(edges):
        adj[e["n0"]].append(ei)
        adj[e["n1"]].append(ei)

    # ------------------------------------------------------------------
    # 3) Contract degree-2 chains, then find cycles
    # ------------------------------------------------------------------
    def other_node(edge_idx, node):
        e = edges[edge_idx]
        return e["n1"] if e["n0"] == node else e["n0"]

    used_edges = set()

    def walk_chain(start_edge, start_node):
        """Follow unique degree-2 continuation; return edge list and end node."""
        chain = [start_edge]
        used_edges.add(start_edge)
        cur = other_node(start_edge, start_node)

        while True:
            cands = [ei for ei in adj[cur] if ei not in used_edges]
            if len(cands) != 1:
                break
            deg = len(adj[cur])
            if deg != 2 and len(cands) != 1:
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
        chain_edges, end_node = walk_chain(ei, e["n0"])
        start_node = e["n0"]

        nodes = [start_node]
        cur = start_node
        segs = []
        path_ids = []
        length = 0.0
        for ce in chain_edges:
            ee = edges[ce]
            segs.extend(ee["segments"])
            path_ids.append(ee["path_id"])
            length += ee["length"]
            cur = other_node(ce, cur)
            nodes.append(cur)

        closed = (nodes[0] == nodes[-1] and len(chain_edges) >= 1)

        chains.append({
            "edge_indices": chain_edges,
            "nodes": nodes,
            "closed": closed,
            "style": e["style"],
            "length": length,
            "segments": segs,
            "path_ids": path_ids,
            "start_node": nodes[0],
            "end_node": nodes[-1],
        })

    # Chain graph for multi-chain cycles
    chain_adj = defaultdict(list)
    for ci, ch in enumerate(chains):
        if ch["closed"]:
            continue
        chain_adj[ch["start_node"]].append(ci)
        chain_adj[ch["end_node"]].append(ci)

    def chain_other(ci, node):
        ch = chains[ci]
        return ch["end_node"] if ch["start_node"] == node else ch["start_node"]

    cycle_closed = []

    # Already-closed contracted chains
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

    def points_for_chain_sequence(chain_idxs):
        segs = []
        pids = []
        length = 0.0
        style = chains[chain_idxs[0]]["style"]
        for ci in chain_idxs:
            ch = chains[ci]
            segs.extend(ch["segments"])
            pids.extend(ch["path_ids"])
            length += ch["length"]
        pts = _ordered_points_from_segments(segs, dist)
        area = poly_area(pts) if len(pts) >= 3 else 0.0
        return segs, pids, length, area, style, pts

    used_chain_in_cycle = set()

    # DFS cycles among open chains
    for start_ci, ch0 in enumerate(chains):
        if ch0["closed"]:
            continue
        if start_ci in used_chain_in_cycle:
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
                if nci in vis:
                    continue
                if chains[nci]["style"] != ch0["style"]:
                    continue
                nnode = chain_other(nci, node)
                stack.append((nnode, path_c + [nci], vis | {nci}))
        if not found:
            continue

        segs, pids, length, area, style, pts = points_for_chain_sequence(found)
        if length >= min_perimeter * 0.5:
            cycle_closed.append({
                "path_ids": pids,
                "segments": segs,
                "length": length,
                "area": area,
                "bbox": bbox_of_segs(segs),
                "style": style,
                "closed": True,
                "points": pts,
            })
            used_chain_in_cycle.update(found)

    # Near-cycles: open chain whose endpoints are close
    node_xy = defaultdict(list)
    for ti, (pi, ei, xy, style) in enumerate(terms):
        node_xy[find(ti)].append(xy)

    def node_point(nid):
        arr = node_xy.get(nid) or [(0.0, 0.0)]
        return (sum(p[0] for p in arr) / len(arr),
                sum(p[1] for p in arr) / len(arr))

    for ci, ch in enumerate(chains):
        if ch["closed"] or ci in used_chain_in_cycle:
            continue
        p0 = node_point(ch["start_node"])
        p1 = node_point(ch["end_node"])
        if dist(p0, p1) <= gap_threshold * 2:
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
            used_chain_in_cycle.add(ci)

    # ------------------------------------------------------------------
    # 4) Select main pieces from cycles
    # ------------------------------------------------------------------
    mains = [
        c for c in cycle_closed
        if c["length"] >= min_perimeter and c["area"] >= min_polygon_area
    ]
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
            # kept empty for backward compatibility with any code that still looks for it
            "size_variants": [],
        })

    print(f"Pattern pieces kept: {len(pieces)}")
    for i, p in enumerate(pieces):
        print(f"  [{i}] paths={len(p['path_ids'])} area={p['area']:.0f}")
    return pieces

def _dedupe_cycles(cycles):
    out = []
    seen = []
    for c in cycles:
        key = frozenset(c["path_ids"])
        if not key:
            continue
        dup = False
        for s in seen:
            # if heavily overlapping path sets, keep larger area only
            inter = len(key & s)
            if inter and inter >= 0.8 * min(len(key), len(s)):
                dup = True
                break
        if dup:
            # replace if bigger area
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


# ------------------------------------------------------------------
# Write text output
# ------------------------------------------------------------------
def write_patterns_txt(pieces, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Pattern pieces: {len(pieces)}\n")
        f.write("# Main outline only (user selects sections interactively)\n\n")

        for i, p in enumerate(pieces):
            f.write(f"PIECE {i}\n")
            f.write(f"  area      : {p.get('area', 0.0):.1f}\n")
            f.write(f"  perimeter : {p.get('perimeter', 0.0):.1f}\n")
            f.write(f"  bbox      : {p.get('bbox')}\n")
            f.write(f"  closed    : {p.get('closed', True)}\n")
            f.write(f"  path_ids  : {p.get('path_ids', [])}\n")
            f.write(f"  segments  : {len(p.get('segments', []))}\n")
            for s in p.get("segments", []):
                f.write(f"    {s[0]} -> {s[1]}\n")

            # ordered points (useful for interactive selection later)
            pts = p.get("points") or []
            if pts:
                f.write(f"  points    : {len(pts)}\n")
                for pt in pts:
                    f.write(f"    {pt}\n")

            f.write("\n")

    print(f"Wrote patterns → {out_path}")


def write_lines_txt(lines, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Special lines: {len(lines)}\n\n")
        for i, obj in enumerate(lines):
            f.write(f"LINE {i}  type={obj['type']}\n")
            f.write(f"  score        : {obj['score']}\n")
            f.write(f"  shaft_length : {obj['shaft']['length']:.1f}\n")
            f.write(f"  shaft_segs   : {obj['shaft']['num_segments']}\n")
            f.write(f"  endpoints    : {obj['shaft']['endpoints']}\n")
            f.write(f"  arrows       : {len(obj['arrows'])}\n")
            f.write("  shaft segments:\n")
            for s in obj["shaft"]["segments"]:
                f.write(f"    {s[0]} -> {s[1]}\n")
            f.write("  arrow segments:\n")
            for a in obj["arrows"]:
                for s in a["segments"]:
                    f.write(f"    {s[0]} -> {s[1]}\n")
            f.write("\n")
    print(f"Wrote lines → {out_path}")


# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------
def render(page, objects, mode, out_path: Path, assembled=None):
    """
    Render detection results.
    
    - assembled=None  → classic single-page render (with PDF background)
    - assembled=dict  → single large assembled canvas (no background image)
    """
    if assembled is not None:
        # ---------- multi-page assembled view ----------
        global_rect = assembled["global_rect"]
        fig_w = 16
        fig_h = max(8.0, fig_w * global_rect.height / max(global_rect.width, 1.0))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_xlim(global_rect.x0 - 10, global_rect.x1 + 10)
        ax.set_ylim(global_rect.y1 + 10, global_rect.y0 - 10)  # PDF y-down
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            f"{mode.upper()} – assembled view  "
            f"({assembled['grid'][0]}×{assembled['grid'][1]} tiles)"
        )

        # Light tile grid
        tile_w = assembled["tile_w"]
        tile_h = assembled["tile_h"]
        cols, rows = assembled["grid"]
        for r in range(rows + 1):
            y = r * tile_h
            ax.axhline(y, color="0.85", linewidth=0.6, zorder=0)
        for c in range(cols + 1):
            x = c * tile_w
            ax.axvline(x, color="0.85", linewidth=0.6, zorder=0)

    else:
        # ---------- classic single-page render ----------
        page_rect = page.rect
        fig, ax = plt.subplots(figsize=(12, 12 * page_rect.height / page_rect.width))
        ax.set_xlim(page_rect.x0, page_rect.x1)
        ax.set_ylim(page_rect.y1, page_rect.y0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{mode.upper()} detection")

        # PDF background
        pix = page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        ax.imshow(
            img,
            extent=[page_rect.x0, page_rect.x1, page_rect.y1, page_rect.y0],
            alpha=0.40,
            zorder=0,
        )

    colors = plt.cm.tab10.colors

    if mode == "patterns":
        for i, piece in enumerate(objects):
            main_color = colors[i % len(colors)]

            # Main outline
            if piece.get("segments"):
                lc = LineCollection(
                    piece["segments"],
                    colors=[main_color],
                    linewidths=2.0,
                    alpha=0.95,
                    zorder=3,
                )
                ax.add_collection(lc)

            # Bounding box + index
            x0, y0, x1, y1 = piece["bbox"]
            rect = Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor=main_color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                zorder=1,
            )
            ax.add_patch(rect)
            ax.text(
                x0 + 4,
                y1 - 4,
                f"{i}",
                color=main_color,
                fontsize=11,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1),
                zorder=6,
            )

        # Legend (simplified)
        ax.text(
            0.02, 0.02,
            "Solid coloured = main outline",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=3),
            zorder=7,
        )

    else:  # lines mode
        for i, obj in enumerate(objects):
            color = "red" if obj["type"] == "grain" else "purple"
            lc = LineCollection(
                obj["all_segments"],
                colors=color,
                linewidths=2.2,
                alpha=0.9,
                zorder=3,
            )
            ax.add_collection(lc)
            for ep in obj["shaft"]["endpoints"]:
                ax.plot(ep[0], ep[1], "o", color="orange", markersize=6, zorder=4)
            ax.text(
                obj["shaft"]["endpoints"][0][0],
                obj["shaft"]["endpoints"][0][1],
                f"{i}:{obj['type'][0].upper()}",
                color=color,
                fontsize=9,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
                zorder=5,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved PNG → {out_path}")


def render_faces(page, faces_per_piece, out_path: Path, assembled=None):
    """
    Draw every atomic face in a different colour on top of the page
    (or on the assembled canvas).
    """
    if assembled is not None:
        global_rect = assembled["global_rect"]
        fig_w = 16
        fig_h = max(8.0, fig_w * global_rect.height / max(global_rect.width, 1.0))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_xlim(global_rect.x0 - 10, global_rect.x1 + 10)
        ax.set_ylim(global_rect.y1 + 10, global_rect.y0 - 10)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Atomic faces (clickable regions)")
    else:
        page_rect = page.rect
        fig, ax = plt.subplots(figsize=(12, 12 * page_rect.height / page_rect.width))
        ax.set_xlim(page_rect.x0, page_rect.x1)
        ax.set_ylim(page_rect.y1, page_rect.y0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Atomic faces")

        # faint PDF background
        pix = page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        ax.imshow(
            img,
            extent=[page_rect.x0, page_rect.x1, page_rect.y1, page_rect.y0],
            alpha=0.35,
            zorder=0,
        )

    colors = plt.cm.tab20.colors
    face_id = 0

    for piece_idx, faces in faces_per_piece:
        for fi, face in enumerate(faces):
            color = colors[face_id % len(colors)]
            face_id += 1

            # filled region
            from matplotlib.patches import Polygon as MplPolygon
            poly = MplPolygon(
                face["points"],
                closed=True,
                facecolor=color,
                edgecolor="black",
                linewidth=1.2,
                alpha=0.45,
                zorder=3,
            )
            ax.add_patch(poly)

            # label
            cx = sum(p[0] for p in face["points"]) / len(face["points"])
            cy = sum(p[1] for p in face["points"]) / len(face["points"])
            ax.text(
                cx, cy,
                f"{piece_idx}.{fi}",
                color="black",
                fontsize=9,
                fontweight="bold",
                ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
                zorder=5,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved faces PNG → {out_path}")

# ------------------------------------------------------------------
# REGION / FACE DETECTION (Shapely)
# ------------------------------------------------------------------

def _extend_line(ls, factor=1.6):
    """Extend a LineString beyond its endpoints so it is guaranteed to cross a surrounding boundary."""
    coords = list(ls.coords)
    if len(coords) < 2:
        return ls
    (x0, y0), (x1, y1) = coords[0], coords[-1]
    dx, dy = x1 - x0, y1 - y0
    # push both ends outward
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = factor / 2
    new0 = (cx - dx * half, cy - dy * half)
    new1 = (cx + dx * half, cy + dy * half)
    return LineString([new0, new1])


def detect_faces_for_piece(main_piece,
                           all_segs_with_id,
                           min_face_area=500.0,
                           snap_tol=2.5,
                           inside_tol=6.0):
    # ---- outer boundary ----
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

    # ---- candidate internal segments ----
    candidates = []
    for path_id, (a, b) in all_segs_with_id:
        if path_id in main_path_ids:
            continue
        if dist(a, b) < 1.0:
            continue
        ls = LineString([a, b])
        if outer.intersects(ls) or outer.distance(ls) <= inside_tol:
            candidates.append(ls)

    print(f"  → {len(candidates)} candidate internal segments for this piece")

    if not candidates:
        # nothing to cut
        coords = list(outer.exterior.coords)
        return [{
            "polygon": outer,
            "segments": [(coords[i], coords[i+1]) for i in range(len(coords)-1)],
            "points": coords[:-1],
            "area": float(outer.area),
            "bbox": outer.bounds,
        }]

    # ---- robust arrangement ----
    # 1. closed outer boundary as LineString
    boundary = LineString(list(outer.exterior.coords))

    # 2. extend every cutter so it is guaranteed to cross the boundary
    extended = [_extend_line(c, factor=1.8) for c in candidates]

    # 3. node everything
    all_lines = [boundary] + extended
    noded = unary_union(MultiLineString(all_lines))

    # 4. polygonize
    raw_faces = list(polygonize(noded))

    # 5. keep only faces whose centroid is inside the original outer
    result = []
    for face in raw_faces:
        if face.is_empty or face.area < min_face_area:
            continue
        try:
            cen = face.centroid
            if not outer.contains(cen) and not outer.touches(cen):
                continue
            # final clip for safety
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
        segs = [(coords[i], coords[i+1]) for i in range(len(coords)-1)]
        result.append({
            "polygon": face,
            "segments": segs,
            "points": coords[:-1],
            "area": float(face.area),
            "bbox": face.bounds,
        })

    result.sort(key=lambda f: f["area"], reverse=True)
    return result

def detect_all_faces(pieces, page, **kwargs):
    """
    Collect every stroked segment together with its original path index,
    then run face detection for each main piece.
    """
    all_segs_with_id = []          # list of (path_id, segment)
    for path_id, path in enumerate(page.get_drawings()):
        segs = path_to_segments(path)
        for s in segs:
            all_segs_with_id.append((path_id, s))

    print(f"Total segments on page: {len(all_segs_with_id)}")

    out = []
    for i, piece in enumerate(pieces):
        faces = detect_faces_for_piece(piece, all_segs_with_id, **kwargs)
        out.append((i, faces))
        print(f"Piece {i}: {len(faces)} atomic face(s)  "
              f"(main area={piece.get('area', 0):.0f})")
    return out

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Debug multi-page assembly")
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
    line_pages = parse_page_spec(args.lines_pages,   instr_doc)

    if not pat_pages:
        raise SystemExit("No valid pattern pages selected")
    if not line_pages:
        raise SystemExit("No valid lines/instructions pages selected")

    print(f"Pattern pages : {pat_pages}")
    print(f"Lines pages   : {line_pages}")    

    print("=== PATTERN PDF ===")
    assembled_pat = assemble_pages(pattern_doc, pat_pages,
                                   instr_doc, line_pages,
                                   overlap=args.overlap)

    print("\n=== LINES / INSTRUCTIONS PDF ===")
    assembled_lines = assemble_pages(instr_doc, line_pages,
                                     instr_doc, line_pages,
                                     overlap=args.overlap)

    # Page-like object so the existing detectors work unchanged
    assembled_page_pat = AssembledPage(assembled_pat["paths"], assembled_pat["global_rect"])    
    assembled_page_lines = AssembledPage(assembled_lines["paths"], assembled_lines["global_rect"])    

    # ------------------------------------------------------------------
    # Output location = same folder as the pattern PDF
    # ------------------------------------------------------------------
    out_dir = args.pattern_pdf.parent
    stem = f"{args.pattern_pdf.stem}_assembled"

    # ------------------------------------------------------------------
    # Detect pattern pieces
    # ------------------------------------------------------------------
    print("\n=== Detecting pattern pieces ===")
    pieces = detect_patterns(assembled_page_pat)
    print(f"Found {len(pieces)} pattern piece(s)")
    write_patterns_txt(pieces, out_dir / f"{stem}_patterns.txt")
    render(
        assembled_page_pat,
        pieces,
        "patterns",
        out_dir / f"{stem}_patterns.png",
        assembled=assembled_pat,
    )

    # ------------------------------------------------------------------
    # Detect special lines (grain / fold)
    # ------------------------------------------------------------------
    print("\n=== Detecting special lines ===")
    lines = detect_special_lines(assembled_page_lines)
    print(f"Found {len(lines)} special line(s)")
    for i, obj in enumerate(lines):
        print(f"  [{i}] {obj['type']:5s}  score={obj['score']}  "
              f"shaft_len={obj['shaft']['length']:.1f}  "
              f"segs={obj['shaft']['num_segments']}")
    write_lines_txt(lines, out_dir / f"{stem}_lines.txt")
    render(
        assembled_page_lines,
        lines,
        "lines",
        out_dir / f"{stem}_lines.png",
        assembled=assembled_lines,
    )

    # Collect every segment that was drawn on the page
    all_segments = []
    for path in assembled_page_pat.get_drawings():
        segs = path_to_segments(path)
        all_segments.extend(segs)

    # ------------------------------------------------------------------
    # Detect faces inside each pattern piece
    # ------------------------------------------------------------------
    print("\n=== Detecting atomic faces ===")
    faces_per_piece = detect_all_faces(
        pieces,
        assembled_page_pat,
        min_face_area=400.0,
        snap_tol=3.0,
        inside_tol=8.0,
    )

    for piece_idx, faces in faces_per_piece:
        print(f"  Piece {piece_idx}: {len(faces)} face(s)")

    # visualise
    render_faces(
        assembled_page_pat,
        faces_per_piece,
        out_dir / f"{stem}_faces.png",
        assembled=assembled_pat,
    )


    pattern_doc.close()
    instr_doc.close()
    print("\nDone.")
    
if __name__ == "__main__":
    main()


# TODO:
# Choosing length/lining options inside the main pattern outline is too complicated...
# Let user choose area of pattern outline.
# Assume main outline was chosen, and user can choose areas to mark/unmark.
