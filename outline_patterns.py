#!/usr/bin/env python3
"""
Detect sewing pattern pieces and special lines (grain / fold).

Outputs:
  - PNG visualisation
  - *_patterns.txt
  - *_lines.txt

Usage:
    python outline_patterns.py input.pdf --mode patterns --page 0
    python outline_patterns.py input.pdf --mode lines    --page 0
    python outline_patterns.py input.pdf --mode both     --page 0
"""

import argparse
from pathlib import Path
from collections import defaultdict, Counter
import fitz
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle


# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------
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
# PATTERN detection
# ------------------------------------------------------------------
def detect_patterns(page,
                    gap_threshold=10.0,
                    point_tol=1.0,
                    min_perimeter=250.0,
                    min_polygon_area=8000.0,
                    min_option_length=60.0,
                    max_dart_length=220.0,
                    attach_tol=12.0):
    """
    Goal 2 pattern detection via generic geometric graph cycles.

    - Each stroked path is an edge between two terminals
    - Same-style terminals snap within gap_threshold
    - Closed outlines = cycles of any number of paths
    - Near-cycles allowed when chain ends are within gap
    - Darts = short same-style leftovers
    - Length options = one joined chain per alternate style attached to a main cycle
    """
    from collections import defaultdict, Counter
    import math

    drawings = page.get_drawings()

    def quant(p):
        return (round(p[0] / point_tol), round(p[1] / point_tol))

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

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
        })

    n_paths = len(paths)
    print(f"Pattern paths used: {n_paths}")
    if not paths:
        return []

    # ------------------------------------------------------------------
    # 2) Same-style terminal snapping → graph nodes
    # ------------------------------------------------------------------
    # terminal records: (path_local, end_idx 0/1, xy, style)
    terms = []
    for i, p in enumerate(paths):
        style = (p["color"], p["width"])
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

    # path edge endpoints in snapped node space
    edges = []  # dicts
    for i, p in enumerate(paths):
        n0 = node_id(i, 0)
        n1 = node_id(i, 1)
        edges.append({
            "edge_idx": i,
            "path_local": i,
            "n0": n0,
            "n1": n1,
            "style": (p["color"], p["width"]),
            "length": p["length"],
            "path_id": p["id"],
            "segments": p["segments"],
        })

    # adjacency: node -> list of edge indices
    adj = defaultdict(list)
    for ei, e in enumerate(edges):
        adj[e["n0"]].append(ei)
        adj[e["n1"]].append(ei)

    # ------------------------------------------------------------------
    # 3) Contract degree-2 chains, then find cycles generically
    # ------------------------------------------------------------------
    def other_node(edge_idx, node):
        e = edges[edge_idx]
        return e["n1"] if e["n0"] == node else e["n0"]

    used_edges = set()

    def walk_chain(start_edge, start_node):
        """Follow unique degree-2 continuation; return edge list and end node."""
        chain = [start_edge]
        used_edges.add(start_edge)
        prev = start_node
        cur = other_node(start_edge, start_node)

        while True:
            cands = [ei for ei in adj[cur] if ei not in used_edges]
            # only continue automatically through degree-2 corridor
            if len(cands) != 1:
                break
            # also stop if node degree (all edges) != 2 and not continuing uniquely
            deg = len(adj[cur])
            if deg != 2 and len(cands) != 1:
                break
            nxt = cands[0]
            # style consistency along main outline
            if edges[nxt]["style"] != edges[chain[0]]["style"]:
                break
            used_edges.add(nxt)
            chain.append(nxt)
            prev, cur = cur, other_node(nxt, cur)
            if cur == start_node:
                break
        return chain, cur

    # Build maximal chains starting from every unused edge
    chains = []  # each: {edge_indices, nodes, closed, style, length, segments, path_ids}
    for ei in range(len(edges)):
        if ei in used_edges:
            continue
        e = edges[ei]
        # start from n0
        chain_edges, end_node = walk_chain(ei, e["n0"])
        start_node = e["n0"]

        # reconstruct node sequence
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
        # near-close single chain
        if not closed and len(nodes) >= 2:
            # map node -> sample xy
            # use average of constituent terminal xys
            pass

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

    # Build chain-graph at junction nodes for multi-chain cycles
    # A "chain edge" connects start_node -- end_node if not already closed
    chain_adj = defaultdict(list)  # node -> list of chain_idx
    for ci, ch in enumerate(chains):
        if ch["closed"]:
            continue
        chain_adj[ch["start_node"]].append(ci)
        chain_adj[ch["end_node"]].append(ci)

    def chain_other(ci, node):
        ch = chains[ci]
        return ch["end_node"] if ch["start_node"] == node else ch["start_node"]

    # Enumerate simple cycles over chains (generic N-chain)
    cycle_closed = []
    # include already-closed contracted chains
    for ch in chains:
        if not ch["closed"]:
            continue
        # near-area
        pts = []
        for a, b in ch["segments"]:
            pts.append(a)
            pts.append(b)
        # better ordered points:
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
        })

    # DFS cycles among open chains
    used_chain_in_cycle = set()

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

    # For each open chain, try to find a return cycle
    for start_ci, ch0 in enumerate(chains):
        if ch0["closed"]:
            continue
        if start_ci in used_chain_in_cycle:
            continue
        start = ch0["start_node"]
        # DFS: state = (node, path_chains, visited_chains)
        stack = [(ch0["end_node"], [start_ci], {start_ci})]
        found = None
        while stack:
            node, path_c, vis = stack.pop()
            if node == start and len(path_c) >= 2:
                found = path_c
                break
            if len(path_c) > 80:  # safety
                continue
            for nci in chain_adj.get(node, []):
                if nci in vis:
                    continue
                if chains[nci]["style"] != ch0["style"]:
                    continue
                nnode = chain_other(nci, node)
                stack.append((nnode, path_c + [nci], vis | {nci}))
        if not found:
            # also near-cycle: ends within gap after one chain
            # handled below
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
            })
            used_chain_in_cycle.add(ci)

    # ------------------------------------------------------------------
    # 4) Select main pieces from cycles
    # ------------------------------------------------------------------
    mains = [
        c for c in cycle_closed
        if c["length"] >= min_perimeter and c["area"] >= min_polygon_area
    ]
    # dedupe by path set
    mains = _dedupe_cycles(mains)
    mains.sort(key=lambda c: c["area"], reverse=True)

    main_path_ids = set()
    for m in mains:
        main_path_ids.update(m["path_ids"])

    # Leftover open chains not used in mains
    leftovers = []
    for ci, ch in enumerate(chains):
        if ci in used_chain_in_cycle and set(ch["path_ids"]).issubset(main_path_ids):
            continue
        if set(ch["path_ids"]) & main_path_ids == set(ch["path_ids"]) and ch["path_ids"]:
            continue
        if not ch["path_ids"]:
            continue
        # skip if fully consumed by a main
        if ch["path_ids"] and set(ch["path_ids"]).issubset(main_path_ids):
            continue
        leftovers.append(ch)

    # ------------------------------------------------------------------
    # 5) Attach darts + length options
    # ------------------------------------------------------------------
    def attaches(chain_segs, main_segs):
        if not chain_segs or not main_segs:
            return False
        ends = [chain_segs[0][0], chain_segs[0][1],
                chain_segs[-1][0], chain_segs[-1][1]]
        sample = []
        step = max(1, len(main_segs) // 100)
        for a, b in main_segs[::step]:
            sample.append(a)
            sample.append(b)
        for e in ends:
            for p in sample:
                if dist(e, p) <= attach_tol:
                    return True
        return False

    def inside_bbox(segs, bbox, pad=30.0):
        x0, y0, x1, y1 = bbox
        pts = [p for s in segs for p in s]
        if not pts:
            return False
        ok = sum(1 for p in pts
                 if x0-pad <= p[0] <= x1+pad and y0-pad <= p[1] <= y1+pad)
        return ok >= 0.6 * len(pts)

    pieces = []
    used_left = set()

    for m in mains:
        darts = []
        # group leftover by style
        style_groups = defaultdict(list)
        for li, ch in enumerate(leftovers):
            if li in used_left:
                continue
            if set(ch["path_ids"]) & set(m["path_ids"]):
                continue
            if not inside_bbox(ch["segments"], m["bbox"]) and not attaches(ch["segments"], m["segments"]):
                continue
            style_groups[ch["style"]].append(li)

        variants = []
        for st, lis in style_groups.items():
            segs = []
            pids = []
            length = 0.0
            for li in lis:
                ch = leftovers[li]
                segs.extend(ch["segments"])
                pids.extend(ch["path_ids"])
                length += ch["length"]

            if st == m["style"]:
                if length <= max_dart_length and length > 0:
                    darts.append({
                        "segments": segs,
                        "path_ids": sorted(set(pids)),
                        "length": length,
                        "color": st[0],
                    })
                    used_left.update(lis)
                continue

            if length < min_option_length:
                continue
            if not attaches(segs, m["segments"]):
                continue
            variants.append({
                "segments": segs,
                "path_ids": sorted(set(pids)),
                "length": length,
                "color": st[0],
                "width": st[1],
            })
            used_left.update(lis)

        variants.sort(key=lambda v: v["length"], reverse=True)
        pieces.append({
            "segments": m["segments"],
            "bbox": m["bbox"],
            "area": m["area"],
            "perimeter": m["length"],
            "path_ids": list(dict.fromkeys(m["path_ids"])),  # unique, contour only
            "darts": darts,
            "size_variants": variants,
            "closed": True,
        })

    print(f"Pattern pieces kept: {len(pieces)}")
    for i, p in enumerate(pieces):
        print(f"  [{i}] paths={len(p['path_ids'])} options={len(p['size_variants'])} "
              f"darts={len(p['darts'])} area={p['area']:.0f}")
    return pieces


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


# ------------------------------------------------------------------
# SPECIAL LINES detection (grain / fold)
# ------------------------------------------------------------------
def detect_special_lines(page,
                         min_shaft_length=35.0,
                         arrow_search_radius=8.0,
                         max_arrow_size=35.0,
                         max_arrow_segments=30,
                         min_arrow_width=0.7,
                         point_tol=1.0):
    """
    Detect grain / fold lines.
    """
    from collections import Counter

    drawings = page.get_drawings()

    shafts = []
    arrow_candidates = []

    def quantize(p):
        return (round(p[0] / point_tol), round(p[1] / point_tol))

    def geometric_terminals(segs):
        counts = Counter()
        examples = {}
        for a, b in segs:
            qa, qb = quantize(a), quantize(b)
            counts[qa] += 1
            counts[qb] += 1
            examples[qa] = a
            examples[qb] = b

        terminals = [examples[q] for q, c in counts.items() if c == 1]

        if len(terminals) < 2:
            pts = [p for s in segs for p in s]
            max_d = -1.0
            ep1 = ep2 = pts[0]
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    d = point_distance(pts[i], pts[j])
                    if d > max_d:
                        max_d = d
                        ep1, ep2 = pts[i], pts[j]
            terminals = [ep1, ep2]

        if len(terminals) > 2:
            max_d = -1.0
            best = (terminals[0], terminals[1])
            for i in range(len(terminals)):
                for j in range(i + 1, len(terminals)):
                    d = point_distance(terminals[i], terminals[j])
                    if d > max_d:
                        max_d = d
                        best = (terminals[i], terminals[j])
            terminals = list(best)

        return tuple(terminals[:2])

    def has_long_straight_segment(path, min_len):
        """
        True if the original path contains at least one straight 'l' segment
        long enough to be a real shaft edge.
        """
        for item in path.get("items", []):
            if item[0] != "l":
                continue
            p1, p2 = item[1], item[2]
            if point_distance((p1.x, p1.y), (p2.x, p2.y)) >= min_len:
                return True
        return False

    for path in drawings:
        segs = path_to_segments(path)
        if not segs:
            continue

        length = sum(segment_length(s) for s in segs)
        r = path.get("rect")
        if r is None:
            continue

        width = path.get("width") or 0.0
        num_segments = len(segs)

        # ---------- potential shaft ----------
        is_stroked = path.get("color") is not None or width > 0
        if is_stroked and length >= min_shaft_length:
            # Has at least one long straight "l"
            if not has_long_straight_segment(path, min_shaft_length):
                pass  # reject as shaft
            else:
                endpoints = geometric_terminals(segs)
                shafts.append({
                    "path": path,
                    "segments": segs,
                    "length": length,
                    "endpoints": endpoints,
                    "rect": r,
                    "num_segments": num_segments,
                })

        # ---------- potential external arrowhead ----------
        size = max(r.width, r.height)
        if size > max_arrow_size:
            continue
        if num_segments > max_arrow_segments:
            continue
        if 0 < width < min_arrow_width:
            continue

        arrow_candidates.append({
            "path": path,
            "segments": segs,
            "rect": r,
            "center": ((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2),
        })

    results = []
    for shaft in shafts:
        ep1, ep2 = shaft["endpoints"]
        nearby = []

        for arrow in arrow_candidates:
            if arrow["path"] is shaft["path"]:
                continue
            d1 = point_distance(arrow["center"], ep1)
            d2 = point_distance(arrow["center"], ep2)
            if min(d1, d2) <= arrow_search_radius:
                nearby.append(arrow)

        # still allow a single arrowhead
        if not nearby:
            continue

        line_type = "grain" if shaft["num_segments"] == 1 else "fold"
        score = len(nearby)

        results.append({
            "type": line_type,
            "shaft": shaft,
            "arrows": nearby,
            "score": score,
            "segments": shaft["segments"],
            "all_segments": shaft["segments"] +
                            [s for a in nearby for s in a["segments"]],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ------------------------------------------------------------------
# Write text output
# ------------------------------------------------------------------
def write_patterns_txt(pieces, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Pattern pieces: {len(pieces)}\n")
        f.write("# Goal 2 format: main outline + darts + size/length options\n\n")

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

            darts = p.get("darts", []) or []
            f.write(f"  darts     : {len(darts)}\n")
            for di, d in enumerate(darts):
                f.write(f"    DART {di}\n")
                f.write(f"      length   : {d.get('length', 0.0):.1f}\n")
                f.write(f"      color    : {d.get('color')}\n")
                f.write(f"      path_ids : {d.get('path_ids', [])}\n")
                f.write(f"      segments : {len(d.get('segments', []))}\n")
                for s in d.get("segments", []):
                    f.write(f"        {s[0]} -> {s[1]}\n")

            variants = p.get("size_variants", []) or []
            f.write(f"  options   : {len(variants)}\n")
            for vi, v in enumerate(variants):
                f.write(f"    OPTION {vi}\n")
                f.write(f"      length   : {v.get('length', 0.0):.1f}\n")
                f.write(f"      color    : {v.get('color')}\n")
                f.write(f"      path_ids : {v.get('path_ids', [])}\n")
                f.write(f"      segments : {len(v.get('segments', []))}\n")
                for s in v.get("segments", []):
                    f.write(f"        {s[0]} -> {s[1]}\n")

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
def render(page, objects, mode, out_path: Path):
    page_rect = page.rect
    fig, ax = plt.subplots(figsize=(12, 12 * page_rect.height / page_rect.width))
    ax.set_xlim(page_rect.x0, page_rect.x1)
    ax.set_ylim(page_rect.y1, page_rect.y0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{mode.upper()} detection")

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

            # Length options in a contrasting style
            for vi, variant in enumerate(piece.get("size_variants", []) or []):
                segs = variant.get("segments") or []
                if not segs:
                    continue
                lc = LineCollection(
                    segs,
                    colors=["orange"],
                    linewidths=2.2,
                    alpha=0.9,
                    linestyles="solid",
                    zorder=4,
                )
                ax.add_collection(lc)

                # Label option near first segment midpoint if possible
                (x1, y1), (x2, y2) = segs[0]
                ax.text(
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    f"{i}.L{vi}",
                    color="darkorange",
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1),
                    zorder=5,
                )

            # Darts dimmed
            for di, dart in enumerate(piece.get("darts", []) or []):
                segs = dart.get("segments") or []
                if not segs:
                    continue
                lc = LineCollection(
                    segs,
                    colors=["0.45"],
                    linewidths=1.2,
                    alpha=0.7,
                    linestyles=":",
                    zorder=2,
                )
                ax.add_collection(lc)

            # BBox + piece index
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

        # Small legend
        ax.text(
            page_rect.x0 + 10,
            page_rect.y0 + 20,
            "Solid coloured = main outline\nOrange = length options\nGrey dotted = darts",
            fontsize=8,
            color="black",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3),
            zorder=7,
        )

    else:
        # lines mode unchanged
        for i, obj in enumerate(objects):
            color = "red" if obj["type"] == "grain" else "purple"
            lc = LineCollection(obj["all_segments"], colors=color, linewidths=2.2, alpha=0.9)
            ax.add_collection(lc)
            for ep in obj["shaft"]["endpoints"]:
                ax.plot(ep[0], ep[1], "o", color="orange", markersize=6)
            ax.text(
                obj["shaft"]["endpoints"][0][0],
                obj["shaft"]["endpoints"][0][1],
                f"{i}:{obj['type'][0].upper()}",
                color=color,
                fontsize=9,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved PNG → {out_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--mode", choices=["patterns", "lines", "both"], default="both")
    parser.add_argument("--page", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None)

    # pattern params
    parser.add_argument("--gap", type=float, default=10.0)
    parser.add_argument("--point-tol", type=float, default=1.0)
    parser.add_argument("--min-perimeter", type=float, default=250.0)
    parser.add_argument("--min-area", type=float, default=8000.0)
    parser.add_argument("--min-option-length", type=float, default=80.0)
    parser.add_argument("--max-dart-length", type=float, default=220.0)

    # line params
    parser.add_argument("--min-shaft", type=float, default=35.0)
    parser.add_argument("--arrow-radius", type=float, default=8.0)
    parser.add_argument("--max-arrow-size", type=float, default=35.0)

    args = parser.parse_args()

    doc = fitz.open(args.pdf)
    page = doc[args.page]

    out_dir = args.out_dir or args.pdf.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.pdf.stem}_p{args.page:02d}"

    if args.mode in ("patterns", "both"):
        pieces = detect_patterns(
            page,
            gap_threshold=args.gap,
            point_tol=args.point_tol,
            min_perimeter=args.min_perimeter,
            min_polygon_area=args.min_area,
            min_option_length=args.min_option_length,
            max_dart_length=args.max_dart_length,
        )
        print(f"Found {len(pieces)} pattern piece(s)")
        write_patterns_txt(pieces, out_dir / f"{stem}_patterns.txt")
        render(page, pieces, "patterns", out_dir / f"{stem}_patterns.png")

    if args.mode in ("lines", "both"):
        lines = detect_special_lines(
            page,
            min_shaft_length=args.min_shaft,
            arrow_search_radius=args.arrow_radius,
            max_arrow_size=args.max_arrow_size,
            point_tol=args.point_tol,
        )
        print(f"Found {len(lines)} special line(s)")
        for i, obj in enumerate(lines):
            print(f"  [{i}] {obj['type']:5s}  score={obj['score']}  "
                  f"shaft_len={obj['shaft']['length']:.1f}  segs={obj['shaft']['num_segments']}")
        write_lines_txt(lines, out_dir / f"{stem}_lines.txt")
        render(page, lines, "lines", out_dir / f"{stem}_lines.png")

    doc.close()


if __name__ == "__main__":
    main()


# TODO:
# Improve detect_patterns runtime
