#!/usr/bin/env python3
"""Parse outline txt goldens and compare with tolerances."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class OptionResult:
    length: float
    color: tuple[float, ...] | None = None
    path_ids: list[int] = field(default_factory=list)


@dataclass
class PieceResult:
    index: int
    area: float
    perimeter: float
    bbox: tuple[float, float, float, float]
    path_ids: list[int] = field(default_factory=list)
    options: list[OptionResult] = field(default_factory=list)

    @property
    def centre(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


@dataclass
class LineResult:
    index: int
    type: str  # grain | fold
    shaft_length: float
    endpoints: tuple[tuple[float, float], tuple[float, float]] | None = None
    score: float | None = None


def _fnums(s: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)]


def parse_patterns_txt(path: Path) -> list[PieceResult]:
    text = path.read_text(encoding="utf-8")
    pieces: list[PieceResult] = []
    cur: dict[str, Any] | None = None
    cur_opt: dict[str, Any] | None = None

    def flush_opt():
        nonlocal cur_opt, cur
        if cur is None or cur_opt is None:
            cur_opt = None
            return
        color = None
        if cur_opt.get("color_nums"):
            nums = cur_opt["color_nums"]
            color = tuple(nums[:3]) if len(nums) >= 3 else tuple(nums)
        cur["options"].append(
            OptionResult(
                length=float(cur_opt.get("length", 0.0)),
                color=color,
                path_ids=cur_opt.get("path_ids", []),
            )
        )
        cur_opt = None

    def flush_piece():
        nonlocal cur
        flush_opt()
        if cur is None:
            return
        bbox = cur.get("bbox") or (0.0, 0.0, 0.0, 0.0)
        pieces.append(
            PieceResult(
                index=cur["index"],
                area=float(cur.get("area", 0.0)),
                perimeter=float(cur.get("perimeter", 0.0)),
                bbox=tuple(bbox),  # type: ignore
                path_ids=cur.get("path_ids", []),
                options=cur.get("options", []),
            )
        )
        cur = None

    for raw in text.splitlines():
        line = raw.rstrip()
        m = re.match(r"^PIECE\s+(\d+)\s*$", line)
        if m:
            flush_piece()
            cur = {"index": int(m.group(1)), "options": []}
            continue
        if cur is None:
            continue

        if line.startswith("    OPTION"):
            flush_opt()
            cur_opt = {}
            continue

        if cur_opt is not None:
            if "length" in line and ":" in line:
                nums = _fnums(line)
                if nums:
                    cur_opt["length"] = nums[0]
            elif "color" in line and ":" in line:
                cur_opt["color_nums"] = _fnums(line)
            elif "path_ids" in line and ":" in line:
                cur_opt["path_ids"] = [int(x) for x in _fnums(line)]
            continue

        if "area" in line and ":" in line:
            nums = _fnums(line)
            if nums:
                cur["area"] = nums[0]
        elif "perimeter" in line and ":" in line:
            nums = _fnums(line)
            if nums:
                cur["perimeter"] = nums[0]
        elif "bbox" in line and ":" in line:
            nums = _fnums(line)
            if len(nums) >= 4:
                cur["bbox"] = (nums[0], nums[1], nums[2], nums[3])
        elif "path_ids" in line and ":" in line:
            cur["path_ids"] = [int(x) for x in _fnums(line)]

    flush_piece()
    return pieces


def parse_lines_txt(path: Path) -> list[LineResult]:
    text = path.read_text(encoding="utf-8")
    lines: list[LineResult] = []
    cur: dict[str, Any] | None = None

    def flush():
        nonlocal cur
        if cur is None:
            return
        eps = None
        nums = cur.get("endpoint_nums") or []
        if len(nums) >= 4:
            eps = ((nums[0], nums[1]), (nums[2], nums[3]))
        lines.append(
            LineResult(
                index=cur["index"],
                type=cur.get("type", "grain"),
                shaft_length=float(cur.get("shaft_length", 0.0)),
                endpoints=eps,
                score=cur.get("score"),
            )
        )
        cur = None

    for raw in text.splitlines():
        line = raw.rstrip()
        m = re.match(r"^LINE\s+(\d+)\s+type=(\w+)", line)
        if m:
            flush()
            cur = {"index": int(m.group(1)), "type": m.group(2)}
            continue
        if cur is None:
            continue
        if "shaft_length" in line or "shaft_len" in line:
            nums = _fnums(line)
            if nums:
                cur["shaft_length"] = nums[0]
        elif "score" in line and ":" in line:
            nums = _fnums(line)
            if nums:
                cur["score"] = nums[0]
        elif "endpoints" in line and ":" in line:
            cur["endpoint_nums"] = _fnums(line)
    flush()
    return lines


def _centre_dist(a: PieceResult, b: PieceResult) -> float:
    return ((a.centre[0] - b.centre[0]) ** 2 + (a.centre[1] - b.centre[1]) ** 2) ** 0.5


def compare_patterns(
    actual: list[PieceResult],
    golden: list[PieceResult],
    *,
    centre_tol: float = 40.0,
    area_tol_frac: float = 0.10,
    perimeter_tol_frac: float = 0.10,
    option_length_tol_frac: float = 0.15,
    require_option_count: bool = True,
) -> list[str]:
    """Return list of error messages; empty => pass."""
    errors: list[str] = []

    if len(actual) != len(golden):
        errors.append(f"piece count actual={len(actual)} golden={len(golden)}")

    used = set()
    for gi, g in enumerate(golden):
        # match nearest actual centre
        best_i, best_d = None, 1e18
        for ai, a in enumerate(actual):
            if ai in used:
                continue
            d = _centre_dist(a, g)
            if d < best_d:
                best_d, best_i = d, ai
        if best_i is None or best_d > centre_tol:
            errors.append(
                f"golden piece@{g.centre} unmatched (best dist={best_d:.1f})"
            )
            continue
        used.add(best_i)
        a = actual[best_i]

        if g.area > 0 and abs(a.area - g.area) > area_tol_frac * g.area:
            errors.append(
                f"piece centre={g.centre}: area actual={a.area:.1f} golden={g.area:.1f}"
            )
        if g.perimeter > 0 and abs(a.perimeter - g.perimeter) > perimeter_tol_frac * g.perimeter:
            errors.append(
                f"piece centre={g.centre}: perimeter actual={a.perimeter:.1f} golden={g.perimeter:.1f}"
            )

        if require_option_count and len(a.options) != len(g.options):
            errors.append(
                f"piece centre={g.centre}: options actual={len(a.options)} golden={len(g.options)}"
            )
            continue

        # match options by length
        gopts = sorted(g.options, key=lambda o: o.length, reverse=True)
        aopts = sorted(a.options, key=lambda o: o.length, reverse=True)
        for go, ao in zip(gopts, aopts):
            if go.length > 0 and abs(ao.length - go.length) > option_length_tol_frac * go.length:
                errors.append(
                    f"piece centre={g.centre}: option length actual={ao.length:.1f} golden={go.length:.1f}"
                )

    for ai, a in enumerate(actual):
        if ai not in used:
            errors.append(f"extra actual piece@{a.centre} area={a.area:.1f}")

    return errors


def compare_lines(
    actual: list[LineResult],
    golden: list[LineResult],
    *,
    length_tol_frac: float = 0.10,
    endpoint_tol: float = 20.0,
) -> list[str]:
    errors: list[str] = []
    if len(actual) != len(golden):
        errors.append(f"line count actual={len(actual)} golden={len(golden)}")

    # match by shaft length + type
    used = set()
    for g in golden:
        best_i, best_score = None, 1e18
        for ai, a in enumerate(actual):
            if ai in used:
                continue
            if a.type != g.type:
                continue
            score = abs(a.shaft_length - g.shaft_length)
            if score < best_score:
                best_score, best_i = score, ai
        if best_i is None:
            errors.append(f"golden {g.type} len={g.shaft_length:.1f} unmatched")
            continue
        used.add(best_i)
        a = actual[best_i]
        if g.shaft_length > 0 and abs(a.shaft_length - g.shaft_length) > length_tol_frac * g.shaft_length:
            errors.append(
                f"{g.type}: shaft_length actual={a.shaft_length:.1f} golden={g.shaft_length:.1f}"
            )
        if a.endpoints and g.endpoints:
            d0 = ((a.endpoints[0][0]-g.endpoints[0][0])**2 + (a.endpoints[0][1]-g.endpoints[0][1])**2)**0.5
            d1 = ((a.endpoints[1][0]-g.endpoints[1][0])**2 + (a.endpoints[1][1]-g.endpoints[1][1])**2)**0.5
            # ends may be swapped
            d_sw = (
                ((a.endpoints[0][0]-g.endpoints[1][0])**2 + (a.endpoints[0][1]-g.endpoints[1][1])**2)**0.5
                + ((a.endpoints[1][0]-g.endpoints[0][0])**2 + (a.endpoints[1][1]-g.endpoints[0][1])**2)**0.5
            )
            d_n = d0 + d1
            if min(d_n, d_sw) > 2 * endpoint_tol:
                errors.append(f"{g.type} len={g.shaft_length:.1f}: endpoints too far from golden")

    for ai, a in enumerate(actual):
        if ai not in used:
            errors.append(f"extra actual {a.type} len={a.shaft_length:.1f}")
    return errors