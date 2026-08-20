#!/usr/bin/env python3
"""
Batch regression against golden outline txt files.

Usage:
  python regression_test.py
  python regression_test.py --patterns-root /path/to/patterns
  python regression_test.py --only patterns
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fitz

# Import your detection + writers from outline_patterns
from outline_patterns import detect_patterns, detect_special_lines, write_patterns_txt, write_lines_txt
from regression_lib import (
    parse_patterns_txt,
    parse_lines_txt,
    compare_patterns,
    compare_lines,
)


def find_pdf_for_golden(golden_txt: Path) -> Path | None:
    """
    Supports:
      extracted_layers/golden/34_p00_patterns.txt  → extracted_layers/34.pdf
      pattern/golden/34_p00_patterns.txt           → pattern/extracted_layers/34.pdf
    """
    import re

    name = golden_txt.name
    # 34_p00_patterns.txt → 34
    # REPÈRES_&_LÉGENDE_p00_lines.txt → REPÈRES_&_LÉGENDE
    m = re.match(r"^(.*)_p\d{2}_(patterns|lines)\.txt$", name)
    if not m:
        layer_stem = name.split("_p")[0]
    else:
        layer_stem = m.group(1)

    golden_dir = golden_txt.parent          # .../extracted_layers/golden
    layers_dir = golden_dir.parent          # .../extracted_layers

    candidates = [
        layers_dir / f"{layer_stem}.pdf",                    # sibling of golden/
        golden_dir.parent / "extracted_layers" / f"{layer_stem}.pdf",  # if golden at pattern root
        golden_dir.parent.parent / "extracted_layers" / f"{layer_stem}.pdf",
    ]

    for c in candidates:
        if c.exists():
            return c.resolve()

    # last resort: search nearby for exact stem
    for root in {layers_dir, golden_dir.parent, golden_dir.parent.parent}:
        if not root.exists():
            continue
        hit = root / f"{layer_stem}.pdf"
        if hit.exists():
            return hit.resolve()
        hits = list(root.glob(f"**/{layer_stem}.pdf"))
        if hits:
            return hits[0].resolve()

    return None


def re_sub_page(stem: str) -> str:
    import re
    return re.sub(r"_p\d{2}$", "", stem)


def parse_page_mode(golden_txt: Path) -> tuple[int, str]:
    import re
    name = golden_txt.name
    mode = "patterns" if name.endswith("_patterns.txt") else "lines"
    m = re.search(r"_p(\d{2})_", name)
    page = int(m.group(1)) if m else 0
    return page, mode


def load_meta(golden_txt: Path) -> dict:
    meta_path = golden_txt.with_suffix(".meta.json")
    # also allow foo_p00_patterns.meta.json from full name
    alt = Path(str(golden_txt) + ".meta.json")
    for p in (meta_path, alt, golden_txt.parent / (golden_txt.stem + ".meta.json")):
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def run_case(golden_txt: Path, tmp_dir: Path) -> tuple[bool, str]:
    meta = load_meta(golden_txt)
    page, mode = parse_page_mode(golden_txt)
    page = int(meta.get("page", page))
    mode = meta.get("mode", mode)

    pdf = Path(meta["pdf"]) if meta.get("pdf") else find_pdf_for_golden(golden_txt)
    if pdf and not pdf.is_absolute():
        pdf = (golden_txt.parent / pdf).resolve()
    if pdf is None or not pdf.exists():
        return False, f"PDF not found for golden {golden_txt}"

    doc = fitz.open(pdf)
    if page < 0 or page >= len(doc):
        doc.close()
        return False, f"bad page {page} for {pdf.name}"
    page_obj = doc[page]

    out_txt = tmp_dir / golden_txt.name

    if mode == "patterns":
        pieces = detect_patterns(
            page_obj,
            gap_threshold=float(meta.get("gap", 10.0)),
            point_tol=float(meta.get("point_tol", 1.0)),
            min_perimeter=float(meta.get("min_perimeter", 250.0)),
            min_polygon_area=float(meta.get("min_area", 8000.0)),
            min_option_length=float(meta.get("min_option_length", 60.0)),
            attach_tol=float(meta.get("attach_tol", 12.0)),
        )
        write_patterns_txt(pieces, out_txt)
        errors = compare_patterns(parse_patterns_txt(out_txt), parse_patterns_txt(golden_txt))
    else:
        lines = detect_special_lines(
            page_obj,
            min_shaft_length=float(meta.get("min_shaft", 35.0)),
            arrow_search_radius=float(meta.get("arrow_radius", 8.0)),
            max_arrow_size=float(meta.get("max_arrow_size", 35.0)),
            point_tol=float(meta.get("point_tol", 1.0)),
        )
        write_lines_txt(lines, out_txt)
        errors = compare_lines(parse_lines_txt(out_txt), parse_lines_txt(golden_txt))

    doc.close()
    if errors:
        msg = f"FAIL {golden_txt}\n  pdf={pdf.name} page={page} mode={mode}\n  " + "\n  ".join(errors)
        return False, msg
    return True, f"PASS {golden_txt.name} ({pdf.name} p{page} {mode})"


def main():
    parser = argparse.ArgumentParser(description="Batch regression for outline goldens")
    parser.add_argument(
        "--patterns-root",
        type=Path,
        default=Path(__file__).resolve().parent / "patterns",
        help="Root folder containing pattern subfolders with golden/",
    )
    parser.add_argument("--only", choices=["patterns", "lines", "all"], default="all")
    parser.add_argument("--tmp", type=Path, default=Path("/tmp/outline_regression"))
    args = parser.parse_args()

    root = args.patterns_root
    goldens = sorted(root.glob("**/golden/*_patterns.txt")) + sorted(
        root.glob("**/golden/*_lines.txt")
    )
    if args.only == "patterns":
        goldens = [g for g in goldens if g.name.endswith("_patterns.txt")]
    elif args.only == "lines":
        goldens = [g for g in goldens if g.name.endswith("_lines.txt")]

    if not goldens:
        print(f"No goldens under {root}")
        return 1

    args.tmp.mkdir(parents=True, exist_ok=True)
    passes = fails = 0
    details = []

    print(f"Found {len(goldens)} golden file(s) under {root}\n")
    for g in goldens:
        ok, msg = run_case(g, args.tmp)
        print(msg)
        if ok:
            passes += 1
        else:
            fails += 1
            details.append(msg)

    print("\n" + "=" * 60)
    print(f"Summary: {passes} passed, {fails} failed, {len(goldens)} total")
    if fails:
        print("\nFailures:")
        for d in details:
            print("-" * 40)
            print(d)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())