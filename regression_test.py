#!/usr/bin/env python3
"""
Outline regression: patterns/*/regression/golden_selected.json

Usage:
  python regression_test.py list
  python regression_test.py init --case "Folder Name" --pattern-layer "SIZE" --lines-layer "LINES"
  python regression_test.py update-golden --case "Folder Name"
  python regression_test.py run
  python regression_test.py run --only "Folder Name"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from regression_lib import (
    compare_selected,
    default_manifest,
    find_pattern_cases,
    load_json,
    load_manifest,
    run_outline_headless,
    save_json,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def cmd_list(cases):
    print(f"{len(cases)} pattern folder(s):\n")
    for c in cases:
        flags = []
        if c["manifest"].is_file():
            flags.append("manifest")
        if c["golden"].is_file():
            flags.append("golden")
        tag = ", ".join(flags) if flags else "no regression setup"
        print(f"  {c['name']}")
        print(f"    pdf:  {c['pdf'].name}")
        print(f"    setup: {tag}")


def cmd_init(case, pattern_layer: str, lines_layer: str, pages: str | None):
    case["reg_dir"].mkdir(parents=True, exist_ok=True)
    man = default_manifest(pattern_layer, lines_layer, pages)
    save_json(case["manifest"], man)
    print(f"Wrote {case['manifest']}")
    print("Next: run outline interactively once, then:")
    print(f'  python regression_test.py update-golden --case "{case["name"]}"')


def cmd_update_golden(case):
    """Copy latest *_selected.json from debug_outline into golden."""
    if not case["manifest"].is_file():
        raise SystemExit(f"Missing manifest. Run init first for {case['name']}")

    debug = case["folder"] / "debug_outline"
    candidates = sorted(debug.glob("*_selected.json")) if debug.is_dir() else []
    # also allow regression/actual
    if case["actual"].is_file():
        candidates.append(case["actual"])

    if not candidates:
        raise SystemExit(
            f"No *_selected.json under {debug}. "
            "Run outline_patterns_debug.py on this PDF first."
        )

    src = candidates[-1]
    data = load_json(src)
    save_json(case["golden"], data)
    print(f"Golden updated from {src}")
    print(f"  -> {case['golden']}")


def cmd_run(cases, only: str | None):
    selected = cases
    if only:
        selected = [c for c in cases if only.lower() in c["name"].lower()]
        if not selected:
            raise SystemExit(f"No case matching {only!r}")

    failed = 0
    skipped = 0
    passed = 0

    for c in selected:
        print("\n" + "=" * 60)
        print(c["name"])
        print("=" * 60)

        if not c["manifest"].is_file():
            print("SKIP: no manifest.json (run init)")
            skipped += 1
            continue
        if not c["golden"].is_file():
            print("SKIP: no golden_selected.json (run update-golden)")
            skipped += 1
            continue

        manifest = load_manifest(c["manifest"])
        out_dir = c["reg_dir"] / "run"
        try:
            actual = run_outline_headless(c["pdf"], manifest, out_dir)
        except Exception as e:
            print(f"FAIL: run error: {e}")
            failed += 1
            continue

        save_json(c["actual"], actual)
        golden = load_json(c["golden"])
        diffs = compare_selected(
            golden,
            actual,
            abs_tol=float(manifest.get("float_abs", 0.05)),
            rel_tol=float(manifest.get("float_rel", 1e-4)),
            area_rel=float(manifest.get("area_rel", 1e-3)),
        )
        if diffs:
            print(f"FAIL ({len(diffs)} diff(s)):")
            for d in diffs[:40]:
                print(f"  - {d}")
            if len(diffs) > 40:
                print(f"  ... {len(diffs) - 40} more")
            failed += 1
        else:
            print("PASS")
            passed += 1

    print("\n" + "-" * 40)
    print(f"passed={passed}  failed={failed}  skipped={skipped}")
    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Outline selected.json regression")
    parser.add_argument(
        "--patterns-root",
        type=Path,
        default=None,
        help="Default: <repo>/patterns",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List pattern folders and setup status")

    p_init = sub.add_parser("init", help="Create regression/manifest.json for a case")
    p_init.add_argument("--case", required=True, help="Folder name under patterns/")
    p_init.add_argument("--pattern-layer", required=True, help="Layer name or key")
    p_init.add_argument("--lines-layer", required=True, help="Layer name or key")
    p_init.add_argument("--pages", default=None, help='Optional page spec e.g. "0-2"')

    p_up = sub.add_parser("update-golden", help="Copy latest selected.json to golden")
    p_up.add_argument("--case", required=True)

    p_run = sub.add_parser("run", help="Headless re-run and compare to golden")
    p_run.add_argument("--only", default=None, help="Substring filter on folder name")

    args = parser.parse_args()
    root = args.patterns_root or (repo_root() / "patterns")
    cases = find_pattern_cases(root)

    if args.cmd == "list":
        cmd_list(cases)
        return 0

    def pick(name: str):
        hits = [c for c in cases if c["name"] == name or name.lower() in c["name"].lower()]
        if not hits:
            raise SystemExit(f"Case not found: {name!r}")
        if len(hits) > 1:
            # prefer exact
            exact = [c for c in hits if c["name"] == name]
            if len(exact) == 1:
                return exact[0]
            raise SystemExit(
                "Ambiguous case; matches: " + ", ".join(c["name"] for c in hits)
            )
        return hits[0]

    if args.cmd == "init":
        cmd_init(pick(args.case), args.pattern_layer, args.lines_layer, args.pages)
        return 0

    if args.cmd == "update-golden":
        cmd_update_golden(pick(args.case))
        return 0

    if args.cmd == "run":
        return cmd_run(cases, args.only)

    return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
