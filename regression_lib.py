#!/usr/bin/env python3
"""Helpers for outline selected.json regression."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


def find_pattern_cases(patterns_root: Path) -> list[dict[str, Any]]:
    """
    Each subfolder of patterns_root that contains at least one PDF is a case.
    """
    cases = []
    if not patterns_root.is_dir():
        return cases
    for folder in sorted(patterns_root.iterdir()):
        if not folder.is_dir():
            continue
        pdfs = sorted(folder.glob("*.pdf")) + sorted(folder.glob("*.PDF"))
        if not pdfs:
            continue
        reg = folder / "regression"
        cases.append({
            "name": folder.name,
            "folder": folder,
            "pdf": pdfs[0],
            "reg_dir": reg,
            "manifest": reg / "manifest.json",
            "golden": reg / "golden_selected.json",
            "actual": reg / "actual_selected.json",
        })
    return cases


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_manifest(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if "pattern_layer" not in data or "lines_layer" not in data:
        raise ValueError(f"manifest missing pattern_layer/lines_layer: {path}")
    return data


def default_manifest(
    pattern_layer: str,
    lines_layer: str,
    pages: str | None = None,
) -> dict[str, Any]:
    return {
        "pattern_layer": pattern_layer,
        "lines_layer": lines_layer,
        "pages": pages,
        "fabric": {
            "name": "Fabric 1",
            "width": 140.0,
            "width_unit": "cm",
            "directional": False,
        },
        "default_copies": 1,
        # Compare tolerances
        "float_abs": 0.05,
        "float_rel": 1e-4,
        "area_rel": 1e-3,
    }


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def normalize_selected(data: dict[str, Any], ndigits: int = 3) -> dict[str, Any]:
    """
    Stable structure for comparison: round floats, drop volatile keys,
    sort lines by type/angle/length.
    """
    skip_keys = {"path_ids", "segments"}  # segments huge; points+bbox enough

    def norm(x: Any) -> Any:
        if _is_number(x):
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return None
            return round(float(x), ndigits)
        if isinstance(x, list):
            return [norm(v) for v in x]
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                if k in skip_keys:
                    continue
                out[k] = norm(v)
            return out
        return x

    cleaned = norm(data)

    for fab in cleaned.get("fabrics") or []:
        pieces = fab.get("pieces") or []
        for p in pieces:
            lines = p.get("lines") or []
            lines.sort(
                key=lambda ln: (
                    ln.get("type") or "",
                    float(ln.get("angle_deg") or 0.0),
                    float(ln.get("length") or 0.0),
                )
            )
            p["lines"] = lines
    return cleaned


def _near(a: float, b: float, abs_tol: float, rel_tol: float) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b), 1.0))


def compare_selected(
    golden: dict[str, Any],
    actual: dict[str, Any],
    abs_tol: float = 0.05,
    rel_tol: float = 1e-4,
    area_rel: float = 1e-3,
) -> list[str]:
    """Return list of human-readable diffs (empty => pass)."""
    diffs: list[str] = []
    g = normalize_selected(golden)
    a = normalize_selected(actual)

    gf = g.get("fabrics") or []
    af = a.get("fabrics") or []
    if len(gf) != len(af):
        diffs.append(f"fabric count: golden={len(gf)} actual={len(af)}")
        return diffs

    for fi, (gfab, afab) in enumerate(zip(gf, af)):
        prefix = f"fabric[{fi}]"
        for key in ("name", "width_unit", "directional"):
            if gfab.get(key) != afab.get(key):
                diffs.append(f"{prefix}.{key}: {gfab.get(key)!r} vs {afab.get(key)!r}")
        if _is_number(gfab.get("width")) and _is_number(afab.get("width")):
            if not _near(float(gfab["width"]), float(afab["width"]), abs_tol, rel_tol):
                diffs.append(f"{prefix}.width: {gfab['width']} vs {afab['width']}")

        gp = gfab.get("pieces") or []
        ap = afab.get("pieces") or []
        if len(gp) != len(ap):
            diffs.append(f"{prefix}.pieces count: {len(gp)} vs {len(ap)}")
            continue

        for pi, (gpiece, apiece) in enumerate(zip(gp, ap)):
            pp = f"{prefix}.piece[{pi}]"
            for key in ("copies", "folded", "source_piece"):
                if gpiece.get(key) != apiece.get(key):
                    diffs.append(
                        f"{pp}.{key}: {gpiece.get(key)!r} vs {apiece.get(key)!r}"
                    )

            if _is_number(gpiece.get("area")) and _is_number(apiece.get("area")):
                ga, aa = float(gpiece["area"]), float(apiece["area"])
                if not _near(ga, aa, abs_tol, area_rel):
                    diffs.append(f"{pp}.area: {ga:.3f} vs {aa:.3f}")

            gb = gpiece.get("bbox") or []
            ab = apiece.get("bbox") or []
            if len(gb) == 4 and len(ab) == 4:
                for i, (x, y) in enumerate(zip(gb, ab)):
                    if not _near(float(x), float(y), abs_tol, rel_tol):
                        diffs.append(f"{pp}.bbox[{i}]: {x} vs {y}")
                        break

            gl = gpiece.get("lines") or []
            al = apiece.get("lines") or []
            if len(gl) != len(al):
                diffs.append(f"{pp}.lines count: {len(gl)} vs {len(al)}")
            else:
                for li, (gln, aln) in enumerate(zip(gl, al)):
                    lp = f"{pp}.line[{li}]"
                    if gln.get("type") != aln.get("type"):
                        diffs.append(
                            f"{lp}.type: {gln.get('type')!r} vs {aln.get('type')!r}"
                        )
                    for key in ("angle_deg", "length"):
                        if _is_number(gln.get(key)) and _is_number(aln.get(key)):
                            if not _near(
                                float(gln[key]), float(aln[key]), abs_tol, rel_tol
                            ):
                                diffs.append(
                                    f"{lp}.{key}: {gln[key]} vs {aln[key]}"
                                )

            gpts = gpiece.get("points") or []
            apts = apiece.get("points") or []
            if abs(len(gpts) - len(apts)) > max(2, int(0.02 * max(len(gpts), 1))):
                diffs.append(f"{pp}.points count: {len(gpts)} vs {len(apts)}")

            gfa = gpiece.get("fold_axes") or []
            afa = apiece.get("fold_axes") or []
            if len(gfa) != len(afa):
                diffs.append(f"{pp}.fold_axes count: {len(gfa)} vs {len(afa)}")

    return diffs


def resolve_layer(layers: list[dict], spec: str) -> dict:
    """
    spec may be layer display name or key (substring match, case-insensitive).
    """
    s = spec.strip().lower()
    for layer in layers:
        if layer["key"].lower() == s or layer["name"].lower() == s:
            return layer
    for layer in layers:
        if s in layer["name"].lower() or s in layer["key"].lower():
            return layer
    names = ", ".join(f"{L['name']} ({L['key']})" for L in layers)
    raise SystemExit(f"Layer not found: {spec!r}. Available: {names}")


def build_select_all(
    faces_per_piece: list,
    fabric: dict[str, Any],
    default_copies: int = 1,
) -> dict[str, Any]:
    """Non-interactive selection: every piece, every face, fixed fabric."""
    from shapely.ops import unary_union

    pieces = []
    for pi, fl in faces_per_piece:
        polys = [f["polygon"] for f in fl]
        if not polys:
            continue
        u = unary_union(polys)
        if u.is_empty:
            continue
        if u.geom_type == "MultiPolygon":
            u = max(u.geoms, key=lambda g: g.area)
        if u.geom_type != "Polygon":
            continue
        # Micro-gap close (same idea as UI outline)
        try:
            u = u.buffer(0.5, join_style=2).buffer(-0.5, join_style=2)
            if u.geom_type == "MultiPolygon":
                u = max(u.geoms, key=lambda g: g.area)
        except Exception:
            pass
        if u.geom_type != "Polygon" or u.is_empty:
            continue
        coords = list(u.exterior.coords)
        pieces.append({
            "segments": [(coords[i], coords[i + 1]) for i in range(len(coords) - 1)],
            "points": coords[:-1],
            "area": float(u.area),
            "bbox": u.bounds,
            "source_piece": pi,
            "selected_faces": list(range(len(fl))),
            "closed": True,
            "size_variants": [],
            "path_ids": [],
            "perimeter": float(u.length),
            "copies": int(default_copies),
            "lines": [],
        })

    return {
        "fabrics": [{
            "name": fabric.get("name", "Fabric 1"),
            "width": float(fabric.get("width", 140.0)),
            "width_unit": fabric.get("width_unit", "cm"),
            "directional": bool(fabric.get("directional", False)),
            "pieces": pieces,
        }]
    }


def auto_resolve_grains(selected: dict[str, Any]) -> None:
    """If multiple grains on a piece, keep the longest; keep all folds."""
    for fab in selected.get("fabrics") or []:
        for p in fab.get("pieces") or []:
            grains = [ln for ln in (p.get("lines") or []) if ln.get("type") == "grain"]
            folds = [ln for ln in (p.get("lines") or []) if ln.get("type") == "fold"]
            if len(grains) <= 1:
                continue
            keep = max(grains, key=lambda ln: float(ln.get("length") or 0.0))
            p["lines"] = [keep] + folds


def run_outline_headless(
    pdf: Path,
    manifest: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """
    Run outline_patterns_debug pipeline without UI.
    Returns the selected payload (same structure as *_selected.json).
    """
    import fitz
    import outline_patterns_debug as op

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{pdf.stem}_assembled"

    layers = op.list_pdf_layers(pdf)
    if not layers:
        raise SystemExit(f"No layers in {pdf}")

    pat = resolve_layer(layers, str(manifest["pattern_layer"]))
    lin = resolve_layer(layers, str(manifest["lines_layer"]))

    pattern_path = out_dir / "layer_pattern.pdf"
    lines_path = out_dir / "layer_lines.pdf"
    op.extract_one_layer(pdf, pat["key"], pattern_path)
    op.extract_one_layer(pdf, lin["key"], lines_path)

    pattern_doc = fitz.open(pattern_path)
    instr_doc = fitz.open(lines_path)
    pages = manifest.get("pages")
    pat_pages = op.parse_page_spec(pages, pattern_doc)
    line_pages = op.parse_page_spec(pages, instr_doc)

    assembled_pat = op.assemble_pages(
        pattern_doc, pat_pages, pattern_doc, pat_pages,
        overlap=float(manifest.get("overlap", 0.0)),
    )
    page = op.AssembledPage(assembled_pat["paths"], assembled_pat["global_rect"])

    assembled_lines = op.assemble_pages(
        instr_doc, line_pages, instr_doc, line_pages,
        overlap=float(manifest.get("overlap", 0.0)),
    )
    page_lines = op.AssembledPage(
        assembled_lines["paths"], assembled_lines["global_rect"]
    )

    lines = op.step_lines_detect(
        page_lines, out_dir, stem, assembled=assembled_lines,
    )

    records, _ = op.step1_read_and_draw(
        page, out_dir=out_dir, stem=stem,
        min_segment_length=float(manifest.get("min_seg_length", 2.0)),
    )
    ext_records, _, applied, hit_points = op.step2_extend_to_first_hit(
        records, out_dir=out_dir, stem=stem,
        max_extension=float(manifest.get("max_extension", 15.0)),
    )
    junc_records, _ = op.step3_snap_open_to_hits(
        ext_records, applied, hit_points,
        out_dir=out_dir, stem=stem,
        join_radius=float(manifest.get("join_radius", 7.0)),
        path_snap_radius=float(manifest.get("path_snap_radius", 2.0)),
    )
    noded, _ = op.step4_snap_and_union(
        junc_records, out_dir=out_dir, stem=stem,
        snap_tol=float(manifest.get("snap_tol", 0.1)),
    )
    faces = op.step5_polygonize(
        noded, out_dir, stem,
        min_face_area=float(manifest.get("min_face_area", 400.0)),
    )
    faces_per_piece = op.group_faces_into_pieces(faces)
    lines = op.filter_lines_inside_pieces(lines, faces, tol=20.0)

    fabric = manifest.get("fabric") or {}
    selected = build_select_all(
        faces_per_piece,
        fabric,
        default_copies=int(manifest.get("default_copies", 1)),
    )
    op.attach_lines_to_selection(selected, lines, tol=20.0)
    auto_resolve_grains(selected)
    op.expand_folds_in_selection(selected)

    # Write via existing helper for identical on-disk shape
    op.step_write_selected(selected, out_dir, stem)

    pattern_doc.close()
    instr_doc.close()

    # Prefer the file just written
    written = out_dir / f"{stem}_selected.json"
    if written.is_file():
        return load_json(written)
    return selected
