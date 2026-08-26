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
                    gap_threshold=20.0,
                    point_tol=1.0,
                    min_perimeter=250.0,
                    min_polygon_area=8000.0,
                    min_option_length=60.0,
                    attach_tol=12.0):
    """
    Goal 2 pattern detection via generic geometric graph cycles.

    - Each stroked path is an edge between two terminals
    - Same-style terminals snap within gap_threshold
    - Closed outlines = cycles of any number of paths
    - Near-cycles allowed when chain ends are within gap
    - Length options = one joined chain per alternate style attached to a main cycle
    """
    from collections import defaultdict, Counter
    import math

    drawings = page.get_drawings()

    def variant_key(v, tol=8.0):
        """
        Produce a quantization key so that near-identical lines fall into the same bucket.
        Uses the midpoint and the dominant orientation.
        """
        segs = v["segments"]
        if not segs:
            return None
        # midpoint
        pts = [p for s in segs for p in s]
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        # rough direction
        p0, p1 = segs[0][0], segs[-1][1]
        dx, dy = abs(p1[0] - p0[0]), abs(p1[1] - p0[1])
        horizontal = dx >= dy
        # quantize
        qx = round(mx / tol)
        qy = round(my / tol)
        return (horizontal, qx, qy)

    def deduplicate_variants(variants, tol=8.0):
        """
        Keep only one variant per location.
        Preference: longer one, then the one with more path_ids (or any stable rule).
        """
        groups = {}
        for v in variants:
            key = variant_key(v, tol=tol)
            if key is None:
                continue
            if key not in groups:
                groups[key] = v
            else:
                # keep the longer one (or the first, or the one with smaller path id…)
                if v["length"] > groups[key]["length"]:
                    groups[key] = v
        return list(groups.values())

    def segment_direction(p):
        """Return normalized direction vector of a path (from t0 to t1)."""
        dx = p["t1"][0] - p["t0"][0]
        dy = p["t1"][1] - p["t0"][1]
        length = math.hypot(dx, dy)
        if length < 1e-8:
            return (0.0, 0.0), 0.0
        return (dx / length, dy / length), length

    def angle_between(v1, v2):
        """Smallest angle in degrees between two normalized vectors."""
        dot = max(-1.0, min(1.0, v1[0]*v2[0] + v1[1]*v2[1]))
        return math.degrees(math.acos(dot))

    def max_endpoint_distance(p1, p2):
        """Largest distance among the four endpoints."""
        pts = [p1["t0"], p1["t1"], p2["t0"], p2["t1"]]
        best = 0.0
        for i in range(4):
            for j in range(i+1, 4):
                d = dist(pts[i], pts[j])
                if d > best:
                    best = d
        return best

    def should_join(p1, p2, gap, parallel_angle_tol=5.0, span_eps=0.08):
        """
        Decide whether two nearby paths should be merged into one component.
        
        - If they form a noticeable angle → allow join (polyline corner / continuation).
        - If nearly parallel → only join when the combined span is clearly larger
        than the longer individual segment (i.e. they extend each other,
        not sit on top of one another).
        """
        (d1, len1) = segment_direction(p1)
        (d2, len2) = segment_direction(p2)

        if len1 < 1e-6 or len2 < 1e-6:
            return False

        ang = angle_between(d1, d2)
        ang = min(ang, 180.0 - ang)          # smallest angle

        if ang > parallel_angle_tol:
            # Clearly not parallel – treat as normal continuation / corner
            return True

        # Nearly parallel: check whether the overall span grows
        combined_span = max_endpoint_distance(p1, p2)
        longer = max(len1, len2)

        # Allow join only if the farthest endpoints are meaningfully farther
        # apart than the longer segment alone
        if combined_span > longer * (1.0 + span_eps):
            return True

        # Special case: very short segment attaching to a long one
        # (common in dashed lines). Require at least a small absolute growth.
        if combined_span > longer + max(gap * 0.7, 3.0):
            return True

        return False

    def ordered_chain_endpoints(segs):
        """
        Return the two terminal points of a chain of segments.
        Handles reversed segments and unordered input reasonably well.
        """
        if not segs:
            return None, None

        # Build a simple adjacency of points (rounded to avoid float noise)
        from collections import defaultdict
        def key(p):
            return (round(p[0], 3), round(p[1], 3))

        adj = defaultdict(list)
        points = {}
        for a, b in segs:
            ka, kb = key(a), key(b)
            points[ka] = a
            points[kb] = b
            adj[ka].append(kb)
            adj[kb].append(ka)

        # Terminals = points with degree 1 (or the two farthest if the graph is messy)
        deg1 = [k for k, nbrs in adj.items() if len(set(nbrs)) == 1]
        if len(deg1) >= 2:
            # pick the pair of degree-1 points that are farthest apart
            best = 0.0
            t0 = t1 = None
            for i in range(len(deg1)):
                for j in range(i+1, len(deg1)):
                    d = dist(points[deg1[i]], points[deg1[j]])
                    if d > best:
                        best = d
                        t0, t1 = points[deg1[i]], points[deg1[j]]
            return t0, t1

        # Fallback: just the farthest pair of any points
        pts = list(points.values())
        best = 0.0
        t0 = t1 = pts[0]
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                d = dist(pts[i], pts[j])
                if d > best:
                    best = d
                    t0, t1 = pts[i], pts[j]
        return t0, t1


    def attachment_chord(segs, main_segs, tol=None):
        """
        Distance between the two points of the candidate that lie closest
        to the main outline (i.e. the true attachment points).
        Falls back to geometric endpoints if needed.
        """
        if tol is None:
            tol = attach_tol * 1.5

        # Collect candidate points (all vertices)
        cand_pts = []
        for a, b in segs:
            cand_pts.append(a)
            cand_pts.append(b)
        # dedup
        seen = set()
        uniq = []
        for p in cand_pts:
            k = (round(p[0], 3), round(p[1], 3))
            if k not in seen:
                seen.add(k)
                uniq.append(p)

        # Points that are close to the main outline
        attached = [p for p in uniq if point_to_polyline_dist(p, main_segs) <= tol]

        if len(attached) >= 2:
            # farthest pair among the attached points
            best = 0.0
            for i in range(len(attached)):
                for j in range(i+1, len(attached)):
                    d = dist(attached[i], attached[j])
                    if d > best:
                        best = d
            return best

        # Fallback: geometric endpoints of the chain
        p0, p1 = ordered_chain_endpoints(segs)
        # print(f"pids={pids}  length={length:.2f}  "
        #     f"geom_chord={dist(p0,p1) if p0 else 0:.2f}  "
        #     f"attach_chord={attachment_chord(segs, m['segments']):.2f}  "
        #     f"ratio={length / max(chord, 1e-6):.3f}")                
        if p0 is None:
            return 0.0
        return dist(p0, p1)

    def point_to_segment_dist(p, a, b):
        ax, ay = a
        bx, by = b
        px, py = p
        dx, dy = bx - ax, by - ay
        len_sq = dx*dx + dy*dy
        if len_sq < 1e-12:
            return math.hypot(px-ax, py-ay)
        t = max(0.0, min(1.0, ((px-ax)*dx + (py-ay)*dy) / len_sq))
        projx = ax + t*dx
        projy = ay + t*dy
        return math.hypot(px-projx, py-projy)

    def point_to_polyline_dist(p, segs):
        if not segs:
            return float("inf")
        return min(point_to_segment_dist(p, a, b) for a, b in segs)

    def attaches(chain_segs, main_segs, tol=None):
        if tol is None:
            tol = attach_tol
        if not chain_segs or not main_segs:
            return False
        # both endpoints of every segment in the candidate (handles multi-seg options)
        ends = []
        for a, b in chain_segs:
            ends.append(a)
            ends.append(b)
        # dedup
        seen = set()
        uniq = []
        for e in ends:
            key = (round(e[0], 2), round(e[1], 2))
            if key not in seen:
                seen.add(key)
                uniq.append(e)
        for e in uniq:
            if point_to_polyline_dist(e, main_segs) <= tol:
                return True
        return False

    def is_mostly_axis_aligned(segs, max_angle_dev_deg=15.0):
        """True if the chain is predominantly horizontal or vertical."""
        if not segs:
            return False
        total_len = 0.0
        axis_len = 0.0
        for a, b in segs:
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            ln = math.hypot(dx, dy)
            if ln < 1e-6:
                continue
            total_len += ln
            ang = abs(math.degrees(math.atan2(dy, dx))) % 180.0
            # near 0° or 90°
            if min(ang, 180 - ang, abs(ang - 90)) <= max_angle_dev_deg:
                axis_len += ln
        return total_len > 0 and axis_len / total_len >= 0.85

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
    # 5) Attach length options
    #    Options must be endpoint-connected chains (not style bags).
    # ------------------------------------------------------------------

    def inside_bbox(segs, bbox, pad=30.0):
        x0, y0, x1, y1 = bbox
        pts = [p for s in segs for p in s]
        if not pts:
            return False
        ok = sum(
            1 for p in pts
            if x0 - pad <= p[0] <= x1 + pad and y0 - pad <= p[1] <= y1 + pad
        )
        return ok >= 0.6 * len(pts)

    def chain_extent(segs):
        if not segs:
            return 0.0
        xs = [p[0] for s in segs for p in s]
        ys = [p[1] for s in segs for p in s]
        return max(max(xs) - min(xs), max(ys) - min(ys))

    # Flatten leftover paths to path-level records with terminals
    # (reuse original path terminals when possible)
    id_to_path = {p["id"]: p for p in paths}

    def leftover_path_records(ch):
        """Expand a leftover chain into individual path records."""
        recs = []
        for pid in ch["path_ids"]:
            p = id_to_path.get(pid)
            if p is None:
                continue
            recs.append(p)
        return recs

    def connect_paths_into_components(path_recs, gap):
        """
        Same-style endpoint connection → connected components.
        Returns list of components, each = list of path records.
        """
        if not path_recs:
            return []

        # only connect within identical style
        by_style = defaultdict(list)
        for i, p in enumerate(path_recs):
            by_style[(p["color"], p["width"])].append(i)

        components = []
        for style, idxs in by_style.items():
            parent = {i: i for i in idxs}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            # endpoint proximity among this style only
            terms = []  # (local_idx, xy)
            for i in idxs:
                p = path_recs[i]
                terms.append((i, p["t0"]))
                terms.append((i, p["t1"]))

            cell = max(gap * 1.5, 4.0)
            grid = defaultdict(list)
            for ti, (i, xy) in enumerate(terms):
                grid[(int(xy[0] // cell), int(xy[1] // cell))].append(ti)

            neigh = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
            for ti, (i, xy) in enumerate(terms):
                cx, cy = int(xy[0] // cell), int(xy[1] // cell)
                for dx, dy in neigh:
                    for tj in grid.get((cx + dx, cy + dy), []):
                        if tj <= ti:
                            continue
                        j, xy2 = terms[tj]
                        if i == j:
                            continue
                        if dist(xy, xy2) > gap:
                            continue
                        # only join if it actually extends the geometry
                        if not should_join(path_recs[i], path_recs[j], gap):
                            continue                        
                        union(i, j)

            groups = defaultdict(list)
            for i in idxs:
                groups[find(i)].append(path_recs[i])
            components.extend(groups.values())

        return components

    pieces = []
    used_path_ids = set()

    for m in mains:
        # Candidate leftovers near this main
        candidate_paths = []
        seen_pid = set()
        for ch in leftovers:
            if not ch["path_ids"]:
                continue
            if set(ch["path_ids"]) & set(m["path_ids"]):
                continue
            if not inside_bbox(ch["segments"], m["bbox"]) and not attaches(ch["segments"], m["segments"]):
                continue
            for p in leftover_path_records(ch):
                if p["id"] in seen_pid or p["id"] in used_path_ids:
                    continue
                if p["id"] in set(m["path_ids"]):
                    continue
                seen_pid.add(p["id"])
                candidate_paths.append(p)

        # CRITICAL: connect by endpoints, not by style bag
        comps = connect_paths_into_components(candidate_paths, gap_threshold)

        variants = []
        main_style = m["style"]

        for comp in comps:
            segs = []
            pids = []
            length = 0.0
            for p in comp:
                segs.extend(p["segments"])
                pids.append(p["id"])
                length += p["length"]

            if not segs:
                continue

            style = (comp[0]["color"], comp[0]["width"])
            extent = chain_extent(segs)

            if length < min_option_length:
                continue
            if extent < min_option_length * 0.5:
                continue

            # Must touch the main outline
            if not attaches(segs, m["segments"]):
                continue

            chord = attachment_chord(segs, m["segments"])
            if length > 1.8 * chord:               # dart filter                              
                continue

            # main_w = m["bbox"][2] - m["bbox"][0]
            # main_h = m["bbox"][3] - m["bbox"][1]
            
            # # Reject tiny base even if the ratio is ok (very small darts / notches)
            # if chord < 0.08 * max(main_w, main_h):   # or 0.12 * min(...)
            #     continue

            variants.append({
                "segments": segs,
                "path_ids": sorted(set(pids)),
                "length": length,
                "color": style[0],
                "width": style[1],
            })
            used_path_ids.update(pids)

        variants = deduplicate_variants(variants, tol=8.0)   # 5–15 works well
        variants.sort(key=lambda v: v["length"], reverse=True)

        pieces.append({
            "segments": m["segments"],
            "bbox": m["bbox"],
            "area": m["area"],
            "perimeter": m["length"],
            "path_ids": list(dict.fromkeys(m["path_ids"])),
            "size_variants": variants,
            "closed": True,
        })

    print(f"Pattern pieces kept: {len(pieces)}")
    for i, p in enumerate(pieces):
        print(f"  [{i}] paths={len(p['path_ids'])} options={len(p['size_variants'])} "
              f"area={p['area']:.0f}")
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
        f.write("# Goal 2 format: main outline + size/length options\n\n")

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
            "Solid coloured = main outline\nOrange = length options\n",
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

    args = parser.parse_args()

    doc = fitz.open(args.pdf)
    page = doc[args.page]

    out_dir = args.out_dir or args.pdf.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.pdf.stem}_p{args.page:02d}"

    if args.mode in ("patterns", "both"):
        pieces = detect_patterns(
            page,
        )
        print(f"Found {len(pieces)} pattern piece(s)")
        write_patterns_txt(pieces, out_dir / f"{stem}_patterns.txt")
        render(page, pieces, "patterns", out_dir / f"{stem}_patterns.png")

    if args.mode in ("lines", "both"):
        lines = detect_special_lines(
            page,
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
# Add support for pattern pieces spanning 2 (Or more than 1) pages.

# current issue with "ultimate" pattern piece.
