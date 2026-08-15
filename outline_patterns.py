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
def detect_patterns(page, dist_threshold=5.0, min_area=600.0):
    drawings = page.get_drawings()
    all_segments = []
    seg_to_path = []

    for path_idx, path in enumerate(drawings):
        segs = path_to_segments(path)
        for s in segs:
            all_segments.append(s)
            seg_to_path.append(path_idx)

    n = len(all_segments)
    if n == 0:
        return []

    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if segments_are_close(all_segments[i], all_segments[j], dist_threshold):
                uf.union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)

    pieces = []
    for seg_indices in clusters.values():
        segs = [all_segments[i] for i in seg_indices]
        xs = [p[0] for s in segs for p in s]
        ys = [p[1] for s in segs for p in s]
        if not xs:
            continue
        bbox = (min(xs), min(ys), max(xs), max(ys))
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < min_area:
            continue
        path_ids = sorted(set(seg_to_path[i] for i in seg_indices))
        pieces.append({
            "segments": segs,
            "bbox": bbox,
            "area": area,
            "path_ids": path_ids,
        })

    pieces.sort(key=lambda p: p["area"], reverse=True)
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
        f.write(f"# Pattern pieces: {len(pieces)}\n\n")
        for i, p in enumerate(pieces):
            f.write(f"PIECE {i}\n")
            f.write(f"  area     : {p['area']:.1f}\n")
            f.write(f"  bbox     : {p['bbox']}\n")
            f.write(f"  path_ids : {p['path_ids']}\n")
            f.write(f"  segments : {len(p['segments'])}\n")
            for s in p["segments"]:
                f.write(f"    {s[0]} -> {s[1]}\n")
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
    ax.imshow(img, extent=[page_rect.x0, page_rect.x1, page_rect.y1, page_rect.y0],
              alpha=0.40, zorder=0)

    colors = plt.cm.tab10.colors

    if mode == "patterns":
        for i, piece in enumerate(objects):
            color = colors[i % len(colors)]
            lc = LineCollection(piece["segments"], colors=[color], linewidths=1.8, alpha=0.9)
            ax.add_collection(lc)
            x0, y0, x1, y1 = piece["bbox"]
            rect = Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor=color,
                             linestyle="--", linewidth=1.0, alpha=0.7)
            ax.add_patch(rect)
            ax.text(x0+4, y1-4, str(i), color=color, fontsize=11, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1))
    else:
        for i, obj in enumerate(objects):
            color = "red" if obj["type"] == "grain" else "purple"
            lc = LineCollection(obj["all_segments"], colors=color, linewidths=2.2, alpha=0.9)
            ax.add_collection(lc)
            for ep in obj["shaft"]["endpoints"]:
                ax.plot(ep[0], ep[1], "o", color="orange", markersize=6)
            ax.text(obj["shaft"]["endpoints"][0][0],
                    obj["shaft"]["endpoints"][0][1],
                    f"{i}:{obj['type'][0].upper()}", color=color,
                    fontsize=9, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

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
    parser.add_argument("--dist", type=float, default=5.0)
    parser.add_argument("--min-area", type=float, default=600.0)

    # line params
    parser.add_argument("--min-shaft", type=float, default=35.0)
    parser.add_argument("--arrow-radius", type=float, default=8.0)
    parser.add_argument("--max-arrow-size", type=float, default=35.0)
    parser.add_argument("--point-tol", type=float, default=1.0)

    args = parser.parse_args()

    doc = fitz.open(args.pdf)
    page = doc[args.page]

    out_dir = args.out_dir or args.pdf.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.pdf.stem}_p{args.page:02d}"

    if args.mode in ("patterns", "both"):
        pieces = detect_patterns(page, dist_threshold=args.dist, min_area=args.min_area)
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
