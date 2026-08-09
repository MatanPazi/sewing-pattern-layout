#!/usr/bin/env python3
"""
Detect sewing pattern pieces by clustering individual segments.
Also supports a simple special-lines mode.

Usage:
    python outline_patterns.py input.pdf --mode patterns --page 0
    python outline_patterns.py input.pdf --mode lines    --page 0

Useful knobs:
    --dist 5          # how close segments must be to join (points)
    --min-area 600    # minimum area of a cluster to keep
    --min-length 50   # for lines mode
"""

import argparse
from pathlib import Path
from collections import defaultdict
import fitz
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle


# ------------------------------------------------------------------
# Convert a path into a list of short line segments
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
            ts = np.linspace(0, 1, 8)          # light approximation
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


# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------
def segment_bbox(seg):
    (x1, y1), (x2, y2) = seg
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def bbox_distance(b1, b2):
    """Minimum distance between two axis-aligned boxes (0 if they overlap)."""
    dx = max(0.0, max(b1[0] - b2[2], b2[0] - b1[2]))
    dy = max(0.0, max(b1[1] - b2[3], b2[1] - b1[3]))
    return (dx*dx + dy*dy) ** 0.5


def point_distance(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])


def segments_are_close(seg1, seg2, threshold):
    """True if any endpoint of seg1 is close to any endpoint of seg2
    or the bounding boxes are close."""
    # Fast bbox check first
    if bbox_distance(segment_bbox(seg1), segment_bbox(seg2)) > threshold:
        return False
    # Endpoint proximity (most important for connecting pieces)
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
# PATTERN detection – cluster individual segments
# ------------------------------------------------------------------
def detect_patterns(page, dist_threshold=5.0, min_area=600.0):
    drawings = page.get_drawings()

    # 1. Collect every individual segment + remember which path it came from
    all_segments = []          # list of ((x1,y1), (x2,y2))
    seg_to_path = []           # parallel list: original path index

    for path_idx, path in enumerate(drawings):
        segs = path_to_segments(path)
        for s in segs:
            all_segments.append(s)
            seg_to_path.append(path_idx)

    n = len(all_segments)
    if n == 0:
        return []

    # 2. Cluster segments that are close to each other
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if segments_are_close(all_segments[i], all_segments[j], dist_threshold):
                uf.union(i, j)

    # 3. Build clusters
    clusters = defaultdict(list)
    for i in range(n):
        root = uf.find(i)
        clusters[root].append(i)

    # 4. Turn each cluster into a pattern candidate
    pieces = []
    for seg_indices in clusters.values():
        segs = [all_segments[i] for i in seg_indices]

        # Combined bounding box
        xs = [p[0] for s in segs for p in s]
        ys = [p[1] for s in segs for p in s]
        if not xs:
            continue
        bbox = (min(xs), min(ys), max(xs), max(ys))
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < min_area:
            continue

        # Which original paths contributed to this cluster
        path_ids = sorted(set(seg_to_path[i] for i in seg_indices))

        pieces.append({
            "segments": segs,
            "bbox": bbox,
            "area": area,
            "path_ids": path_ids,
        })

    # Largest first
    pieces.sort(key=lambda p: p["area"], reverse=True)
    return pieces


# ------------------------------------------------------------------
# LINE mode (unchanged, simple long-thin filter)
# ------------------------------------------------------------------
def detect_special_lines(page, min_length=50.0, max_thickness=6.0):
    drawings = page.get_drawings()
    special = []

    for path in drawings:
        segs = path_to_segments(path)
        if not segs:
            continue
        length = sum(np.hypot(s[1][0]-s[0][0], s[1][1]-s[0][1]) for s in segs)
        r = path.get("rect")
        if r is None:
            continue
        thickness = min(r.width, r.height)
        if length >= min_length and thickness <= max_thickness:
            special.append({
                "segments": segs,
                "length": length,
                "rect": r,
            })
    special.sort(key=lambda x: x["length"], reverse=True)
    return special


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
    ax.set_title(f"{mode.upper()} detection (segment clustering)")

    # Background
    pix = page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    ax.imshow(img, extent=[page_rect.x0, page_rect.x1, page_rect.y1, page_rect.y0],
              alpha=0.4, zorder=0)

    colors = plt.cm.tab10.colors

    if mode == "patterns":
        for i, piece in enumerate(objects):
            color = colors[i % len(colors)]
            lc = LineCollection(piece["segments"], colors=[color],
                                linewidths=1.8, alpha=0.9)
            ax.add_collection(lc)

            x0, y0, x1, y1 = piece["bbox"]
            rect = Rectangle((x0, y0), x1-x0, y1-y0,
                             fill=False, edgecolor=color,
                             linestyle="--", linewidth=1.0, alpha=0.7)
            ax.add_patch(rect)
            ax.text(x0 + 4, y1 - 4, str(i),
                    color=color, fontsize=11, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1))
    else:
        for i, line in enumerate(objects):
            lc = LineCollection(line["segments"], colors="red",
                                linewidths=2.5, alpha=0.9)
            ax.add_collection(lc)
            r = line["rect"]
            ax.text(r.x0, r.y1, str(i), color="red", fontsize=10, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--mode", choices=["patterns", "lines"], required=True)
    parser.add_argument("--page", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dist", type=float, default=5.0,
                        help="Max distance between segment endpoints to join them")
    parser.add_argument("--min-area", type=float, default=600.0)
    parser.add_argument("--min-length", type=float, default=50.0)
    args = parser.parse_args()

    doc = fitz.open(args.pdf)
    page = doc[args.page]

    if args.mode == "patterns":
        objects = detect_patterns(page, dist_threshold=args.dist, min_area=args.min_area)
        print(f"Found {len(objects)} pattern candidate(s)")
        for i, p in enumerate(objects):
            print(f"  [{i}] area={p['area']:.0f}  paths={p['path_ids']}")
    else:
        objects = detect_special_lines(page, min_length=args.min_length)
        print(f"Found {len(objects)} special line candidate(s)")

    out_path = args.out or args.pdf.with_name(
        f"{args.pdf.stem}_p{args.page:02d}_{args.mode}.png"
    )
    render(page, objects, args.mode, out_path)
    doc.close()


if __name__ == "__main__":
    main()