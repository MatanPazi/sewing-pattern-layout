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
    Goal 2:
      - main near-closed outline (path_ids = contour only)
      - darts = short same-style internal chains
      - size options = one joined chain per alternate colour/style
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

    def style_key(path_rec):
        return (path_rec["color"], path_rec["width"])

    def poly_area(pts):
        if len(pts) < 3:
            return 0.0
        a = 0.0
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            a += x1 * y2 - x2 * y1
        return abs(a) * 0.5

    def bbox_of_segs(segs):
        xs = [p[0] for s in segs for p in s]
        ys = [p[1] for s in segs for p in s]
        if not xs:
            return (0.0, 0.0, 0.0, 0.0)
        return (min(xs), min(ys), max(xs), max(ys))

    def path_geometry(path):
        segs = path_to_segments(path)
        if not segs:
            return None
        length = 0.0
        counts = Counter()
        sample = {}
        for a, b in segs:
            length += dist(a, b)
            qa, qb = quant(a), quant(b)
            counts[qa] += 1
            counts[qb] += 1
            sample[qa] = a
            sample[qb] = b
        terms = [sample[q] for q, n in counts.items() if n == 1]
        if len(terms) < 2:
            pts = [p for s in segs for p in s]
            best_d = -1.0
            best = (pts[0], pts[-1])
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    d = dist(pts[i], pts[j])
                    if d > best_d:
                        best_d = d
                        best = (pts[i], pts[j])
            terms = list(best)
        elif len(terms) > 2:
            best_d = -1.0
            best = (terms[0], terms[1])
            for i in range(len(terms)):
                for j in range(i + 1, len(terms)):
                    d = dist(terms[i], terms[j])
                    if d > best_d:
                        best_d = d
                        best = (terms[i], terms[j])
            terms = list(best)
        return segs, (terms[0], terms[1]), length

    # ------------------------------------------------------------------
    # 1) Collect stroked paths
    # ------------------------------------------------------------------
    paths = []
    for idx, path in enumerate(drawings):
        color = path.get("color")
        width = float(path.get("width") or 0.0)
        if color is None and width <= 0:
            continue
        geom = path_geometry(path)
        if geom is None:
            continue
        segs, terminals, length = geom
        if length < 1.0:
            continue
        paths.append({
            "id": idx,
            "color": color_key(color),
            "width": round(width, 2),
            "segments": segs,
            "terminals": terminals,
            "length": length,
        })

    print(f"Pattern paths used: {len(paths)}")
    if not paths:
        return []

    # ------------------------------------------------------------------
    # 2) Snap terminals and connect only same-style paths
    # ------------------------------------------------------------------
    term_nodes = []  # (local_path_idx, term_i, xy)
    for i, pre in enumerate(paths):
        for ti, xy in enumerate(pre["terminals"]):
            term_nodes.append((i, ti, xy))

    parent = list(range(len(term_nodes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    cell = max(gap_threshold * 1.5, 4.0)
    grid = defaultdict(list)
    for i, (_, _, xy) in enumerate(term_nodes):
        grid[(int(xy[0] // cell), int(xy[1] // cell))].append(i)

    neigh = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    for i, (pi, _, xy) in enumerate(term_nodes):
        cx, cy = int(xy[0] // cell), int(xy[1] // cell)
        for dx, dy in neigh:
            for j in grid.get((cx + dx, cy + dy), []):
                if j <= i:
                    continue
                pj = term_nodes[j][0]
                # only snap endpoints of same style
                if style_key(paths[pi]) != style_key(paths[pj]):
                    continue
                if dist(xy, term_nodes[j][2]) <= gap_threshold:
                    union(i, j)

    snap = {}
    for i, (pi, ti, _) in enumerate(term_nodes):
        snap[(pi, ti)] = find(i)

    # adjacency among same-style paths
    node_members = defaultdict(list)
    for i, (pi, ti, _) in enumerate(term_nodes):
        node_members[find(i)].append(pi)

    adj = defaultdict(set)
    for members in node_members.values():
        uniq = sorted(set(members))
        for a in range(len(uniq)):
            for b in range(a + 1, len(uniq)):
                u, v = uniq[a], uniq[b]
                if style_key(paths[u]) == style_key(paths[v]):
                    adj[u].add(v)
                    adj[v].add(u)

    # ------------------------------------------------------------------
    # 3) Connected components per style
    # ------------------------------------------------------------------
    seen = set()
    components = []  # list of list[local_idx]
    for i in range(len(paths)):
        if i in seen:
            continue
        stack = [i]
        seen.add(i)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        components.append(comp)

    # ------------------------------------------------------------------
    # 4) Order a component into a contour by terminal walk
    # ------------------------------------------------------------------
    def order_component(comp):
        if not comp:
            return [], [], False, []

        nmap = defaultdict(list)  # snapped node -> [(local_idx, term_i)]
        for pi in comp:
            for ti in (0, 1):
                nmap[snap[(pi, ti)]].append((pi, ti))

        # start at a true terminal if possible
        start_p, start_t = comp[0], 0
        for pi in comp:
            for ti in (0, 1):
                deg = len({p for p, _ in nmap[snap[(pi, ti)]]})
                if deg == 1:
                    start_p, start_t = pi, ti
                    break

        unused = set(comp)
        ordered = []
        cur_p, cur_t = start_p, start_t
        closed = False

        while cur_p in unused:
            unused.remove(cur_p)
            ordered.append(cur_p)
            other_t = 1 - cur_t
            node = snap[(cur_p, other_t)]
            cands = [(p, t) for p, t in nmap[node] if p in unused]
            if not cands:
                # close if back to start node
                start_node = snap[(ordered[0], start_t)]
                if node == start_node and len(ordered) >= 3:
                    closed = True
                break
            # prefer continuing to a path that keeps degree-2 flow
            cands.sort(key=lambda pt: len({pp for pp, _ in nmap[snap[(pt[0], 1 - pt[1])]]}))
            cur_p, cur_t = cands[0]

        segs = []
        path_ids = []
        for pi in ordered:
            segs.extend(paths[pi]["segments"])
            path_ids.append(paths[pi]["id"])

        return segs, path_ids, closed, sorted(unused)

    def chain_len(local_ids):
        return sum(paths[i]["length"] for i in local_ids)

    def segs_of(local_ids):
        out = []
        for i in local_ids:
            out.extend(paths[i]["segments"])
        return out

    def endpoints_of_chain(local_ids):
        if not local_ids:
            return []
        # use terminals that appear once in the chain
        counts = Counter()
        sample = {}
        for pi in local_ids:
            for ti, xy in enumerate(paths[pi]["terminals"]):
                q = quant(xy)
                counts[q] += 1
                sample[q] = xy
        ends = [sample[q] for q, n in counts.items() if n == 1]
        if len(ends) >= 2:
            return ends
        # fallback: all terminals
        ends = []
        for pi in local_ids:
            ends.extend(paths[pi]["terminals"])
        return ends

    # Build candidate contours from components
    style_chains = []  # each: dict
    for comp in components:
        total = chain_len(comp)
        if total < 20:
            continue
        segs, path_ids, closed, leftover = order_component(comp)
        # If leftover exists, this component had branches; keep ordered part only
        pts = []
        if segs:
            pts = [segs[0][0], segs[0][1]]
            for a, b in segs[1:]:
                if dist(pts[-1], a) <= dist(pts[-1], b):
                    pts.append(b)
                else:
                    pts.append(a)

        area = 0.0
        if len(pts) >= 3:
            if closed or dist(pts[0], pts[-1]) <= gap_threshold * 2:
                closed = True
                area = poly_area(pts)

        style_chains.append({
            "local_ids": [i for i in range(len(paths)) if paths[i]["id"] in path_ids],
            "path_ids": path_ids,          # contour only from the walk
            "segments": segs,
            "length": sum(paths[i]["length"] for i in range(len(paths)) if paths[i]["id"] in set(path_ids)),
            "closed": closed,
            "area": area,
            "bbox": bbox_of_segs(segs),
            "color": paths[comp[0]]["color"],
            "width": paths[comp[0]]["width"],
            "leftover_local": leftover,
        })

    # Fix local_ids properly from path_ids
    id_to_local = {paths[i]["id"]: i for i in range(len(paths))}
    for ch in style_chains:
        ch["local_ids"] = [id_to_local[pid] for pid in ch["path_ids"] if pid in id_to_local]
        ch["length"] = sum(paths[i]["length"] for i in ch["local_ids"])

    # ------------------------------------------------------------------
    # 5) Main pieces = largest closed / near-closed contours
    # ------------------------------------------------------------------
    mains = [c for c in style_chains
             if c["closed"] and c["length"] >= min_perimeter and c["area"] >= min_polygon_area]
    mains.sort(key=lambda c: c["area"], reverse=True)

    # fallback: longest chains if no closed found
    if not mains:
        candidates = sorted(style_chains, key=lambda c: c["length"], reverse=True)
        mains = [c for c in candidates if c["length"] >= min_perimeter][:15]

    # Index remaining chains as option/dart candidates
    main_path_id_set = set()
    for m in mains:
        main_path_id_set.update(m["path_ids"])

    remaining = []
    for ch in style_chains:
        # skip chains fully used as mains
        if ch in mains:
            continue
        # also skip if all path ids already in some main contour
        if ch["path_ids"] and set(ch["path_ids"]).issubset(main_path_id_set):
            continue
        remaining.append(ch)

    # Also create chains from leftover branch pieces inside components
    for ch in style_chains:
        if not ch["leftover_local"]:
            continue
        loc = ch["leftover_local"]
        segs = segs_of(loc)
        remaining.append({
            "local_ids": loc,
            "path_ids": [paths[i]["id"] for i in loc],
            "segments": segs,
            "length": chain_len(loc),
            "closed": False,
            "area": 0.0,
            "bbox": bbox_of_segs(segs),
            "color": paths[loc[0]]["color"],
            "width": paths[loc[0]]["width"],
            "leftover_local": [],
        })

    def outline_points(main):
        # densify a bit from segments for attachment tests
        pts = []
        for a, b in main["segments"]:
            pts.append(a)
            pts.append(b)
        return pts

    def attaches_to_main(chain, main_pts):
        ends = endpoints_of_chain(chain["local_ids"])
        if not ends or not main_pts:
            return False
        # subsample main outline for speed
        step = max(1, len(main_pts) // 200)
        sample = main_pts[::step]
        for e in ends:
            for p in sample:
                if dist(e, p) <= attach_tol:
                    return True
        return False

    def mostly_inside_bbox(chain, bbox, pad=25.0):
        x0, y0, x1, y1 = bbox
        pts = []
        for a, b in chain["segments"]:
            pts.append(a)
            pts.append(b)
        if not pts:
            return False
        inside = 0
        for p in pts:
            if (x0 - pad <= p[0] <= x1 + pad) and (y0 - pad <= p[1] <= y1 + pad):
                inside += 1
        return inside >= 0.6 * len(pts)

    # ------------------------------------------------------------------
    # 6) For each main: group remaining by style → one option per style
    # ------------------------------------------------------------------
    pieces = []
    used_remaining = set()

    for mi, main in enumerate(mains):
        main_pts = outline_points(main)
        main_style = (main["color"], main["width"])

        # collect candidate locals near this main, not same contour paths
        cand_idxs = []
        for ri, ch in enumerate(remaining):
            if ri in used_remaining:
                continue
            if set(ch["path_ids"]) & set(main["path_ids"]):
                continue
            if not mostly_inside_bbox(ch, main["bbox"]) and not attaches_to_main(ch, main_pts):
                continue
            cand_idxs.append(ri)

        # Darts: same style, short, attached/inside
        darts = []
        option_pool = []  # remaining candidates for size options

        for ri in cand_idxs:
            ch = remaining[ri]
            ch_style = (ch["color"], ch["width"])
            if ch_style == main_style and ch["length"] <= max_dart_length:
                darts.append({
                    "segments": ch["segments"],
                    "path_ids": ch["path_ids"],
                    "length": ch["length"],
                    "color": ch["color"],
                })
                used_remaining.add(ri)
            else:
                option_pool.append(ri)

        # Group option pool by colour/style and merge into one option per style
        style_groups = defaultdict(list)
        for ri in option_pool:
            ch = remaining[ri]
            style_groups[(ch["color"], ch["width"])].append(ri)

        variants = []
        for st, rlist in style_groups.items():
            if st == main_style:
                # same style leftovers that weren't short enough for dart:
                # ignore as options (prevents dark fragments becoming "options")
                continue

            # merge all chains of this style near the piece into one option
            local_ids = []
            segs = []
            pids = []
            length = 0.0
            for ri in rlist:
                ch = remaining[ri]
                # require attachment for at least one chain of this style
                local_ids.extend(ch["local_ids"])
                segs.extend(ch["segments"])
                pids.extend(ch["path_ids"])
                length += ch["length"]

            if length < min_option_length:
                continue

            # must attach to main outline
            tmp = {
                "local_ids": local_ids,
                "segments": segs,
            }
            if not attaches_to_main(tmp, main_pts):
                # try weaker: any endpoint near main bbox edge already filtered;
                # still require geometric attach
                continue

            variants.append({
                "segments": segs,
                "path_ids": sorted(set(pids)),
                "length": length,
                "color": st[0],
                "width": st[1],
            })
            for ri in rlist:
                used_remaining.add(ri)

        variants.sort(key=lambda v: v["length"], reverse=True)

        pieces.append({
            "segments": main["segments"],
            "bbox": main["bbox"],
            "area": main["area"],
            "perimeter": main["length"],
            "path_ids": list(main["path_ids"]),  # contour only
            "darts": darts,
            "size_variants": variants,
            "closed": main["closed"],
        })

    print(f"Pattern pieces kept: {len(pieces)}")
    for i, p in enumerate(pieces):
        print(f"  [{i}] paths={len(p['path_ids'])} options={len(p['size_variants'])} darts={len(p['darts'])}")
    return pieces


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
