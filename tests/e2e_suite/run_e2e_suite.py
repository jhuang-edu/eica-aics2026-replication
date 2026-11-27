# -*- coding: utf-8 -*-
"""
Master CLI Runner for the 4-Tier E2E Verification Suite (IEEE AI-CS 2026).

Usage:
  python code/tests/e2e_suite/run_e2e_suite.py --tier all --verbose
  python code/tests/e2e_suite/run_e2e_suite.py --tier 1 --json-out summary.json
  python -m unittest discover -s code/tests/e2e_suite -p "test_*.py"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unittest
import urllib.request
from pathlib import Path
from typing import Dict, List


SUITE_DIR = Path(__file__).resolve().parent
TESTS_ROOT = SUITE_DIR.parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

TIER_MODULES: Dict[str, str] = {
    "1": "e2e_suite.test_tier1_requirements_compliance",
    "2": "e2e_suite.test_tier2_boundary_cases",
    "3": "e2e_suite.test_tier3_cross_feature",
    "4": "e2e_suite.test_tier4_submission_scenarios",
}


def check_reference_urls(timeout_sec: float = 8.0) -> Dict[str, object]:
    """Optional live HTTP/Crossref check for DOIs/URLs in paper_9p/main.tex (only with --check-urls)."""
    import urllib.parse
    from e2e_suite import e2e_helpers as h

    main_tex = h.read_paper_sources()["main.tex"]
    urls = sorted(set(re.findall(r"https?://[^\s}]+", main_tex)))
    dois = sorted(set(d.rstrip(".") for d in re.findall(r"doi:\s*([^\s,}]+)", main_tex, re.IGNORECASE)))
    targets = [(u, u) for u in urls] + [
        (f"doi:{d}", f"https://api.crossref.org/works/{urllib.parse.quote(d, safe='')}")
        for d in dois
    ]

    ok_count = 0
    failures: List[dict] = []
    headers = {
        "User-Agent": "EICA-E2E-Audit/1.0 (mailto:junjun@ieee.org)",
        "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
    }
    for label, url in targets:
        req = urllib.request.Request(url, method="GET", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                if 200 <= resp.status < 400:
                    ok_count += 1
                else:
                    failures.append({"target": label, "url": url, "status": resp.status})
        except Exception as exc:
            failures.append({"target": label, "url": url, "error": str(exc)})
    return {"total": len(targets), "ok": ok_count, "failures": failures}


def run_tier(tier_key: str, verbosity: int) -> dict:
    mod_name = TIER_MODULES[tier_key]
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(mod_name)
    t0 = time.perf_counter()
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=verbosity)
    result = runner.run(suite)
    elapsed = time.perf_counter() - t0

    return {
        "tier": f"Tier {tier_key}",
        "module": mod_name,
        "ran": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "passed": result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
        "ok": result.wasSuccessful(),
        "elapsed_sec": round(elapsed, 2),
        "failure_ids": [t.id() for t, _ in result.failures],
        "error_ids": [t.id() for t, _ in result.errors],
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IEEE AI-CS 2026 4-Tier E2E Verification Runner")
    parser.add_argument("--tier", choices=["1", "2", "3", "4", "all"], default="all", help="Tier to run (default: all)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose unittest output")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON summary output path")
    parser.add_argument("--check-urls", action="store_true", help="Optionally probe live DOI/URL reachability")
    args = parser.parse_args(argv)

    tiers = ["1", "2", "3", "4"] if args.tier == "all" else [args.tier]
    verbosity = 2 if args.verbose else 1

    summaries: List[dict] = []
    t_start = time.perf_counter()
    for t in tiers:
        print(f"\n{'=' * 78}\nRUNNING TIER {t} ({TIER_MODULES[t]})\n{'=' * 78}")
        summaries.append(run_tier(t, verbosity))
    total_elapsed = round(time.perf_counter() - t_start, 2)

    url_report = None
    if args.check_urls:
        print(f"\n{'=' * 78}\nRUNNING OPTIONAL LIVE URL REACHABILITY CHECK\n{'=' * 78}")
        url_report = check_reference_urls()
        print(f"URLs checked: {url_report['ok']}/{url_report['total']} reachable")

    total_ran = sum(s["ran"] for s in summaries)
    total_passed = sum(s["passed"] for s in summaries)
    total_skipped = sum(s["skipped"] for s in summaries)
    total_failures = sum(s["failures"] for s in summaries)
    total_errors = sum(s["errors"] for s in summaries)
    overall_ok = all(s["ok"] for s in summaries) and (url_report is None or not url_report["failures"])

    from e2e_suite import e2e_helpers as h
    mode_label = "FULL WORKSPACE" if h.HAS_FULL_WORKSPACE else "STANDALONE REPLICATION CLONE"

    print(f"\n{'=' * 86}")
    print(f"E2E SUITE SUMMARY (IEEE AI-CS 2026 - A2INZUIKKH) [{mode_label}]")
    print(f"{'=' * 86}")
    print(f"{'Tier':<10} {'Ran':>6} {'Passed':>8} {'Skipped':>8} {'Failed':>8} {'Errors':>8} {'Time(s)':>9} {'Status'}")
    print("-" * 86)
    for s in summaries:
        st = "PASS" if s["ok"] else "FAIL"
        print(f"{s['tier']:<10} {s['ran']:>6} {s['passed']:>8} {s['skipped']:>8} {s['failures']:>8} {s['errors']:>8} {s['elapsed_sec']:>9.2f} {st}")
    print("-" * 86)
    print(f"{'TOTAL':<10} {total_ran:>6} {total_passed:>8} {total_skipped:>8} {total_failures:>8} {total_errors:>8} {total_elapsed:>9.2f} {'PASS' if overall_ok else 'FAIL'}")
    print("=" * 86)

    if args.json_out is not None:
        payload = {
            "mode": mode_label,
            "total_ran": total_ran,
            "total_passed": total_passed,
            "total_skipped": total_skipped,
            "total_failures": total_failures,
            "total_errors": total_errors,
            "overall_ok": overall_ok,
            "elapsed_sec": total_elapsed,
            "tiers": summaries,
            "url_check": url_report,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
