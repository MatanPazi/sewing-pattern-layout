#!/usr/bin/env python3
"""
place_patterns.py — V1 strip nesting for sewing-pattern outlines.

Reads *_selected.json from outline_patterns_debug.py and packs each fabric's
pieces onto a strip of fixed width, minimizing length.

Rules (V1):
  - Grain (or fold-derived grain) aligned with fabric length (selvage).
  - Directional fabric: 0° only; otherwise 0° and 180°.
  - No orientation line: 45° steps.
  - Copies alternate mirror across the length axis.
  - ~1 cm gap via half-gap buffer on each piece.
  - Bottom-left / lowest-then-left grid search with Shapely collision.

Usage:
  python place_patterns.py --json path/to/foo_assembled_selected.json
  python place_patterns.py --json ... --out-dir ./layout --gap-cm 1.0 --grid-cm 0.5
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.affinity import rotate, scale as shp_scale, translate
from shapely.geometry import Polygon, LineString

# ------------------------------------------------------------------
# General helpers
# ------------------------------------------------------------------

def label_point_inside(poly, tolerance=0.5):
    """
    Point suitable for a label: inside the polygon, near the visual center.
    poly: Shapely Polygon (any units).
    Returns (x, y).
    """

    if poly is None or poly.is_empty:
        return (0.0, 0.0)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.geom_type != "Polygon":
        c = poly.centroid
        return (float(c.x), float(c.y))

    try:
        from shapely.ops import polylabel
        # tolerance in the same units as poly (cm in place_patterns, PDF units in debug)
        pt = polylabel(poly, tolerance=tolerance)
        return (float(pt.x), float(pt.y))
    except Exception:
        pt = poly.representative_point()
        return (float(pt.x), float(pt.y))

# ------------------------------------------------------------------
# Units
# ------------------------------------------------------------------

def pdf_units_per_cm() -> float:
    """PDF points per centimetre at true size (100% print scale)."""
    return 72.0 / 2.54


def poly_from_points(points) -> Polygon | None:
    if not points or len(points) < 3:
        return None
    poly = Polygon([(float(x), float(y)) for x, y in points])
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    if poly.geom_type != "Polygon":
        return None
    return poly


def to_cm(poly: Polygon, units_per_cm: float) -> Polygon:
    return shp_scale(
        poly, xfact=1.0 / units_per_cm, yfact=1.0 / units_per_cm, origin=(0, 0)
    )


# ------------------------------------------------------------------
# Orientation helpers
# ------------------------------------------------------------------

def grain_angle_deg(piece: dict) -> float | None:
    """Undirected grain axis in [0, 180), or None if missing."""
    for ln in piece.get("lines") or []:
        if ln.get("type") == "grain":
            if "angle_deg" in ln:
                return float(ln["angle_deg"]) % 180.0
            eps = ln.get("endpoints") or []
            if len(eps) >= 2:
                dx = float(eps[1][0]) - float(eps[0][0])
                dy = float(eps[1][1]) - float(eps[0][1])
                return math.degrees(math.atan2(dy, dx)) % 180.0
    return None


def allowed_rotations(has_grain: bool, directional: bool) -> list[float]:
    if has_grain:
        if directional:
            return [0.0]
        return [0.0, 180.0]
    return [i * 45.0 for i in range(8)]


def orient_grain_to_length(poly: Polygon, angle_deg: float | None) -> Polygon:
    """
    Rotate so the grain axis is parallel to fabric length (+Y).
    angle_deg is the undirected axis in PDF coords (0 = horizontal).
    """
    if angle_deg is None:
        return poly
    delta = 90.0 - float(angle_deg)
    return rotate(poly, delta, origin="centroid", use_radians=False)


def mirror_across_y(poly: Polygon) -> Polygon:
    """Reflect across a vertical line through the centroid (length axis)."""
    cx = poly.centroid.x
    return shp_scale(poly, xfact=-1.0, yfact=1.0, origin=(cx, poly.centroid.y))


def apply_pose(poly: Polygon, rotation_deg: float, mirrored: bool) -> Polygon:
    out = poly
    if mirrored:
        out = mirror_across_y(out)
    if abs(rotation_deg) > 1e-9:
        out = rotate(out, rotation_deg, origin="centroid", use_radians=False)
    return out


# ------------------------------------------------------------------
# Instance expansion
# ------------------------------------------------------------------

def build_instances(fabric: dict, units_per_cm: float) -> list[dict]:
    """
    One dict per physical copy to place.
    Geometry is in centimetres, grain-aligned to +Y when a grain exists.
    """
    directional = bool(fabric.get("directional"))
    instances = []

    for pi, piece in enumerate(fabric.get("pieces") or []):
        raw = poly_from_points(piece.get("points") or [])
        if raw is None:
            continue
        poly_cm = to_cm(raw, units_per_cm)
        g_angle = grain_angle_deg(piece)
        has_grain = g_angle is not None
        base = orient_grain_to_length(poly_cm, g_angle)
        base_axes = []
        for ax in (piece.get("fold_axes") or []):
            if len(ax) < 4:
                continue
            line = LineString([(float(ax[0]), float(ax[1])),
                               (float(ax[2]), float(ax[3]))])
            line = shp_scale(line, 1.0 / units_per_cm, 1.0 / units_per_cm, origin=(0, 0))
            if g_angle is not None:
                delta = 90.0 - float(g_angle)
                line = rotate(line, delta, origin=poly_cm.centroid, use_radians=False)
            coords = list(line.coords)
            base_axes.append([coords[0][0], coords[0][1], coords[1][0], coords[1][1]])        
        rots = allowed_rotations(has_grain, directional)
        n_copies = max(1, int(piece.get("copies") or 1))

        for c in range(n_copies):
            mirrored = (c % 2 == 1)
            instances.append({
                "piece_index": pi,
                "copy_index": c,
                "source_piece": piece.get("source_piece"),
                "name": piece.get("name"),
                "base_poly": base,
                "rotations": rots,
                "mirrored": mirrored,
                "has_grain": has_grain,
                "grain_angle_deg": g_angle,
                "folded": bool(piece.get("folded")),
                "base_fold_axes": base_axes,
            })
    return instances


# ------------------------------------------------------------------
# Bottom-left placer
# ------------------------------------------------------------------

def _candidate_y_values(placed_buffers, poly_h: float, grid: float, max_y: float) -> list[float]:
    ys = {0.0}
    for pb in placed_buffers:
        minx, miny, maxx, maxy = pb.bounds
        ys.add(maxy)
        ys.add(max(0.0, maxy - poly_h))
    y = 0.0
    limit = max(max_y, poly_h) + grid
    while y <= limit + 1e-9:
        ys.add(round(y / grid) * grid)
        y += grid
    return sorted(ys)


def _candidate_x_values(placed_buffers, poly_w: float, fabric_w: float, grid: float) -> list[float]:
    xs = {0.0}
    for pb in placed_buffers:
        minx, miny, maxx, maxy = pb.bounds
        xs.add(maxx)
        xs.add(max(0.0, minx - poly_w))
    x = 0.0
    while x <= fabric_w + 1e-9:
        xs.add(round(x / grid) * grid)
        x += grid
    out = []
    for x in sorted(xs):
        if x < -1e-9:
            continue
        if x + poly_w > fabric_w + 1e-6:
            continue
        out.append(x)
    if not out:
        out = [0.0]
    return out


def place_fabric(
    instances: list[dict],
    fabric_width_cm: float,
    gap_cm: float = 1.0,
    grid_cm: float = 0.5,
) -> tuple[list[dict], float]:
    """
    Greedy bottom-left placement.
    Returns (placements, length_used_cm).
    """
    half_gap = gap_cm * 0.5
    placed_buffers: list[Polygon] = []
    placements: list[dict] = []

    order = sorted(
        range(len(instances)),
        key=lambda i: instances[i]["base_poly"].area,
        reverse=True,
    )

    n = len(order)
    print(f"  placing {n} instance(s)...", flush=True)

    for k, idx in enumerate(order, start=1):
        inst = instances[idx]
        print(f"\r  placing {k}/{n}", end="", flush=True)
        best = None  # (top_y, min_x, rotation, poly, buf)

        for rot in inst["rotations"]:
            posed = apply_pose(inst["base_poly"], rot, inst["mirrored"])
            minx, miny, maxx, maxy = posed.bounds
            w = maxx - minx
            if w > fabric_width_cm + 1e-6:
                continue

            buf = posed.buffer(half_gap, join_style=2)
            if buf.is_empty:
                buf = posed
            bw = buf.bounds[2] - buf.bounds[0]
            bh = buf.bounds[3] - buf.bounds[1]

            current_length = 0.0
            if placed_buffers:
                current_length = max(p.bounds[3] for p in placed_buffers)

            for y0 in _candidate_y_values(placed_buffers, bh, grid_cm, current_length):
                for x0 in _candidate_x_values(placed_buffers, bw, fabric_width_cm, grid_cm):
                    dx = x0 - buf.bounds[0]
                    dy = y0 - buf.bounds[1]
                    trial_buf = translate(buf, xoff=dx, yoff=dy)
                    trial_poly = translate(posed, xoff=dx, yoff=dy)

                    tminx, tminy, tmaxx, tmaxy = trial_poly.bounds
                    if tminx < -1e-6 or tmaxx > fabric_width_cm + 1e-6:
                        continue
                    if tminy < -1e-6:
                        continue

                    collides = False
                    for other in placed_buffers:
                        if trial_buf.intersects(other) and not trial_buf.touches(other):
                            collides = True
                            break
                    if collides:
                        continue

                    top = trial_poly.bounds[3]
                    left = trial_poly.bounds[0]
                    score = (top, left)
                    if best is None or score < (best[0], best[1]):
                        best = (top, left, rot, trial_poly, trial_buf)

        if best is None:
            posed = apply_pose(inst["base_poly"], inst["rotations"][0], inst["mirrored"])
            minx, miny, maxx, maxy = posed.bounds
            y0 = 0.0
            if placed_buffers:
                y0 = max(p.bounds[3] for p in placed_buffers) + half_gap
            dx = -minx
            dy = y0 - miny
            trial = translate(posed, xoff=dx, yoff=dy)
            if trial.bounds[2] > fabric_width_cm:
                dx -= (trial.bounds[2] - fabric_width_cm)
                trial = translate(posed, xoff=dx, yoff=dy)
            trial_buf = trial.buffer(half_gap, join_style=2)
            best = (trial.bounds[3], trial.bounds[0], inst["rotations"][0], trial, trial_buf)

        top, left, rot, trial_poly, trial_buf = best
        placed_buffers.append(trial_buf)
        coords = list(trial_poly.exterior.coords)

        # Pose fold axes the same way as the polygon
        posed = apply_pose(inst["base_poly"], rot, inst["mirrored"])
        dx = float(trial_poly.centroid.x - posed.centroid.x)
        dy = float(trial_poly.centroid.y - posed.centroid.y)

        fold_axes_cm = []
        for ax_seg in inst.get("base_fold_axes") or []:
            if len(ax_seg) < 4:
                continue
            line = LineString([(ax_seg[0], ax_seg[1]), (ax_seg[2], ax_seg[3])])
            if inst["mirrored"]:
                line = mirror_across_y(line)
            if abs(rot) > 1e-9:
                pre = inst["base_poly"]
                if inst["mirrored"]:
                    pre = mirror_across_y(pre)
                line = rotate(line, rot, origin=pre.centroid, use_radians=False)
            line = translate(line, xoff=dx, yoff=dy)
            c = list(line.coords)
            fold_axes_cm.append([c[0][0], c[0][1], c[1][0], c[1][1]])

        placements.append({
            "piece_index": inst["piece_index"],
            "copy_index": inst["copy_index"],
            "source_piece": inst["source_piece"],
            "rotation_deg": float(rot),
            "mirrored": bool(inst["mirrored"]),
            "has_grain": inst["has_grain"],
            "grain_angle_deg": inst["grain_angle_deg"],
            "folded": inst["folded"],
            "fold_axes_cm": fold_axes_cm,
            "bbox_cm": [float(v) for v in trial_poly.bounds],
            "points_cm": [[float(x), float(y)] for x, y in coords[:-1]],
        })

    print(flush=True)  # newline after \r progress

    length = 0.0
    if placements:
        length = max(p["bbox_cm"][3] for p in placements)
    return placements, float(length)


# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------

def draw_layout(fabric_name: str, width_cm: float, length_cm: float,
                placements: list[dict], out_path: Path):
    fig_w = 10
    fig_h = max(4.0, fig_w * (length_cm / max(width_cm, 1e-6)) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, min(fig_h, 24)))
    ax.set_aspect("equal")
    ax.set_xlim(-1, width_cm + 1)
    ax.set_ylim(length_cm + 1, -1)
    ax.add_patch(MplPolygon(
        [[0, 0], [width_cm, 0], [width_cm, length_cm], [0, length_cm]],
        closed=True, fill=False, edgecolor="0.4", linewidth=1.2, linestyle="--",
    ))
    fill = "#4c78a8"
    for p in placements:
        pts = p["points_cm"]
        if len(pts) < 3:
            continue
        for ax_seg in p.get("fold_axes_cm") or []:
            if len(ax_seg) >= 4:
                ax.plot(
                    [ax_seg[0], ax_seg[2]], [ax_seg[1], ax_seg[3]],
                    color="#9b59b6", linestyle="--", linewidth=1.5, zorder=5,
                )        
        ax.add_patch(MplPolygon(
            pts, closed=True, facecolor=fill, edgecolor="black",
            alpha=0.55, linewidth=0.8,
        ))
        poly = Polygon(pts)
        cx, cy = label_point_inside(poly, tolerance=0.5)
        ax.text(
            cx, cy,
            f"{p['piece_index']}:{p['copy_index']}",
            ha="center", va="center", fontsize=7, color="black",
        )
    ax.set_xlabel("width (cm) — cut edge")
    ax.set_ylabel("length (cm) — selvage direction")
    ax.set_title(
        f"{fabric_name}  —  {width_cm:.1f} cm × {length_cm:.1f} cm  "
        f"({len(placements)} pieces)"
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Layout PNG → {out_path}")


def place_from_json(
    data: dict,
    out_dir: Path,
    stem: str,
    gap_cm: float = 1.0,
    grid_cm: float = 0.5,
) -> dict:
    upc = pdf_units_per_cm()
    print(f"Using {upc:.4f} PDF units / cm")

    result = {
        "gap_cm": gap_cm,
        "grid_cm": grid_cm,
        "fabrics": [],
    }

    for fi, fab in enumerate(data.get("fabrics") or []):
        name = fab.get("name") or f"fabric_{fi}"
        width = float(fab.get("width") or 0.0)
        if width <= 0:
            print(f"  skip {name}: invalid width")
            continue
        print(
            f"\n=== Fabric {fi}: {name}  width={width} cm  "
            f"directional={fab.get('directional')} ==="
        )

        instances = build_instances(fab, upc)
        print(f"  instances to place: {len(instances)}")
        if not instances:
            result["fabrics"].append({
                "name": name,
                "width_cm": width,
                "directional": bool(fab.get("directional")),
                "length_cm": 0.0,
                "placements": [],
            })
            continue

        placements, length = place_fabric(
            instances, fabric_width_cm=width, gap_cm=gap_cm, grid_cm=grid_cm,
        )
        print(f"  length used: {length:.2f} cm")

        entry = {
            "name": name,
            "width_cm": width,
            "directional": bool(fab.get("directional")),
            "length_cm": length,
            "placements": placements,
        }
        result["fabrics"].append(entry)

        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
        draw_layout(
            name, width, length, placements,
            out_dir / f"{stem}_layout_{safe or fi}.png",
        )

    json_path = out_dir / f"{stem}_layout.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nLayout JSON → {json_path}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="V1 bottom-left nesting of sewing pattern outlines"
    )
    parser.add_argument(
        "--json", type=Path, required=True,
        help="*_selected.json from outline_patterns_debug.py",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--gap-cm", type=float, default=1.0,
        help="Minimum gap between pieces (default 1 cm)",
    )
    parser.add_argument(
        "--grid-cm", type=float, default=0.5,
        help="Placement grid step (default 0.5 cm)",
    )
    args = parser.parse_args()

    if not args.json.is_file():
        raise SystemExit(f"JSON not found: {args.json}")

    out_dir = args.out_dir or args.json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.json.stem.replace("_selected", "")
    stem = stem + "_placed" if not stem.endswith("_placed") else stem

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)

    place_from_json(
        data, out_dir, stem,
        gap_cm=args.gap_cm, grid_cm=args.grid_cm,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
