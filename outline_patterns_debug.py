#!/usr/bin/env python3
"""
outline_patterns_debug.py – Step 1 + 2 only

Read stroked paths from a PDF page and draw them exactly as they are
(no snap, no union, no extension). Saves a PNG + a short text summary
so you can verify the raw geometry before any later processing.

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

import matplotlib as mpl
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Point
from shapely.ops import polygonize, unary_union

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
            ts = np.linspace(0, 1, 32)
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

def step2_extend_to_first_hit(
    records,
    out_dir: Path,
    stem: str,
    max_extension: float = 12.0,
    hit_eps: float = 0.1,
    past_hit: float = 0.05,
    max_rounds: int = 5,
    lateral_tol: float = 1.0,
):
    """
    Step 2: each terminal extends at most once, to its nearest hit.

    Speed notes
    -----------
    - Proximity tests use path terminals only (not every vertex).
    - Foreign segments and terminals are bucketed in a grid so each ray
      only tests nearby geometry.
    - Multi-round, but default max_rounds=3.
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
        out = []
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                out.append((cx + dx, cy + dy))
        return out

    work = [[list(s) for s in rec["segments"]] for rec in records]
    applied = set()
    n_seg = n_ray = n_prox = 0
    extension_lengths = []
    round_log = []
    hit_points = []   # destinations of successful extensions

    # grid cell ~ max_extension so one ring of neighbours covers the ray
    cell = max(max_extension, 4.0)
    radius_cells = 1

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

        # --- grid of foreign segments ---
        seg_grid = defaultdict(list)  # cell -> list of (path_idx, LineString)
        for pi, segs in enumerate(work):
            for a, b in segs:
                if dist(a, b) <= 1e-12:
                    continue
                ls = LineString([a, b])
                # index by cells covered by bbox of segment
                x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                x1, y1 = max(a[0], b[0]), max(a[1], b[1])
                cx0, cy0 = int(x0 // cell), int(y0 // cell)
                cx1, cy1 = int(x1 // cell), int(y1 // cell)
                for cx in range(cx0, cx1 + 1):
                    for cy in range(cy0, cy1 + 1):
                        seg_grid[(cx, cy)].append((pi, ls))

        # --- path terminals only (deduped) for proximity ---
        # list of (path_idx, point); grid of same
        term_pts = []
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
                term_pts.append(sp)
                term_grid[cell_key(pt[0], pt[1], cell)].append(sp)

        candidates = []

        # ---- ray vs segments (grid-limited) + proximity (terminals only) ----
        for key, (origin, direction) in terminals.items():
            pi = key[0]
            tip = (
                origin[0] + direction[0] * max_extension,
                origin[1] + direction[1] * max_extension,
            )
            ray = LineString([origin, tip])

            # cells along the ray bbox
            rx0, ry0 = min(origin[0], tip[0]), min(origin[1], tip[1])
            rx1, ry1 = max(origin[0], tip[0]), max(origin[1], tip[1])
            cx0, cy0 = int(rx0 // cell), int(ry0 // cell)
            cx1, cy1 = int(rx1 // cell), int(ry1 // cell)

            best_t, best_pt = None, None
            tested_seg = set()  # id(LineString) or (pi, id) avoid retest
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
                                    origin[0] + direction[0] * (t + past_hit),
                                    origin[1] + direction[1] * (t + past_hit),
                                )
            if best_t is not None:
                candidates.append(("seg", best_t, key, best_pt))

            # proximity: only nearby path terminals
            best_pt_t, best_pt_pt = None, None
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
                            origin[0] + direction[0] * (t + past_hit),
                            origin[1] + direction[1] * (t + past_hit),
                        )
            if best_pt_t is not None:
                candidates.append(("prox", best_pt_t, key, best_pt_pt))

        # ---- ray vs ray (only free terminals; still O(m^2) in m free ends) ----
        keys = list(terminals.keys())
        seen = set()
        for i, k1 in enumerate(keys):
            o1, d1 = terminals[k1]
            # only pair with terminals in nearby cells
            near_keys = set()
            for ck in nearby_keys(o1[0], o1[1], cell, radius_cells + 1):
                near_keys.add(ck)
            for k2 in keys[i + 1:]:
                if k2[0] == k1[0]:
                    continue
                o2, d2 = terminals[k2]
                if cell_key(o2[0], o2[1], cell) not in near_keys:
                    # also allow if o2 is along ray within max_extension bbox
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

        # best per terminal: seg/ray first; prox only if no geometric hit
        best_for = {}
        has_geom = defaultdict(bool)

        # best per terminal (seg, ray, prox all compete — smallest t wins)
        best_for = {}
        for cand in candidates:
            if cand[0] in ("seg", "prox"):
                _, t, key, pt = cand
                if key not in best_for or t < best_for[key][0]:
                    best_for[key] = (t, cand)
            else:
                _, _, k1, t1, k2, t2, p = cand
                if k1 not in best_for or t1 < best_for[k1][0]:
                    best_for[k1] = (t1, cand)
                if k2 not in best_for or t2 < best_for[k2][0]:
                    best_for[k2] = (t2, cand)

        def set_terminal(key, pt):
            pi, is_start = key
            if not work[pi]:
                return
            if is_start:
                work[pi][0][0] = pt
            else:
                work[pi][-1][1] = pt

        applied_this_round = 0
        for key, (t, cand) in sorted(best_for.items(), key=lambda kv: kv[1][0]):
            if key in applied:
                continue
            if cand[0] in ("seg", "prox"):
                _, t, k, pt = cand
                if k in applied:
                    continue
                set_terminal(k, pt)
                hit_points.append({"pt": tuple(pt), "path": k[0], "end": "start" if k[1] else "end"})
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
                o1, d1 = terminals[k1]
                o2, d2 = terminals[k2]
                p1 = (o1[0] + d1[0] * (t1 + past_hit), o1[1] + d1[1] * (t1 + past_hit))
                p2 = (o2[0] + d2[0] * (t2 + past_hit), o2[1] + d2[1] * (t2 + past_hit))
                set_terminal(k1, p1)
                set_terminal(k2, p2)
                hit_points.append({"pt": tuple(p1), "path": k1[0], "end": "start" if k1[1] else "end"})
                hit_points.append({"pt": tuple(p2), "path": k2[0], "end": "start" if k2[1] else "end"})
                applied.add(k1)
                applied.add(k2)
                applied_this_round += 1
                n_ray += 1
                extension_lengths.append(t1)
                extension_lengths.append(t2)

        round_log.append(
            f"  round {rnd+1}: free={len(terminals)}  cand={len(candidates)}  "
            f"applied={applied_this_round}"
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
        f.write("STEP 2 – multi-round nearest hit (grid + terminal prox)\n")
        f.write(f"  max_extension     : {max_extension}\n")
        f.write(f"  lateral_tol       : {lateral_tol}\n")
        f.write(f"  max_rounds        : {max_rounds}\n")
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
    Step 3: one-pass join of hit points + open terminals, then T-join
    groups of size >= 2 onto a nearby foreign path (path_snap_radius).
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
        f.write("STEP 3 – join hits+opens, T-join groups>=2 onto paths\n")
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


def select_faces_interactive(page, faces_per_piece, assembled=None):
    selector = FaceSelector(page, faces_per_piece, assembled=assembled)
    return selector.run()
    

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
            min_segment_length=2.0,
        )
        print(f"  paths={len(records)}  segments={len(all_segments)}")

        # ----- STEP 2 - Extend to first hit -----
        ext_records, ext_segments, applied, hit_points = step2_extend_to_first_hit(
            records,
            out_dir=out_dir,
            stem=page_stem,
            max_extension=12.0,
        )
        print(f"  step2  extended_ends={len(applied)}  hit_points={len(hit_points)}")

        # ----- STEP 3 - junction snapping -----
        junc_records, junc_segments = step3_snap_open_to_hits(
            ext_records,
            applied,
            hit_points,
            out_dir=out_dir,
            stem=page_stem,
            join_radius=10.0,
            path_snap_radius=2.0,
        )
        print(f"  step3 segments={len(junc_segments)}")

        # ----- STEP 4 - Snap + unary union -----
        noded, noded_segments = step4_snap_and_union(
            junc_records,
            out_dir=out_dir,
            stem=page_stem,
            snap_tol=0.1,   # float noise only; try 0.01–0.1
        )
        print(f"  step4 noded segments={len(noded_segments)}")        

        # ----- STEP 5 - Polygonize -----
        faces = step5_polygonize(
            noded, out_dir, page_stem, min_face_area=400.0,
        )
        print(f"  step5 faces={len(faces)}")

    faces_per_piece = group_faces_into_pieces(faces)
    print(f"  pieces={len(faces_per_piece)}")
    for pi, fl in faces_per_piece:
        print(f"    piece {pi}: {len(fl)} face(s)")

    print("\n=== Interactive face selection ===")
    print("Click faces to toggle.  Enter/d=done  a=all  c=clear  Esc=cancel")
    selected = select_faces_interactive(page, faces_per_piece)
    print(f"Selected outlines: {len(selected)}")
    for i, p in enumerate(selected):
        print(f"  [{i}] area={p['area']:.0f}  peri={p['perimeter']:.0f}")

    doc.close()
    print("\nDone. Check the Step 1 PNG and summary txt.")


if __name__ == "__main__":
    main()


# TODO
# Handle converging paths.
    # Snap together points at junction if more then 3 path endpoints are within a 5 point radius? 10-HAUTS example, top left pattern bottom left corner. 
