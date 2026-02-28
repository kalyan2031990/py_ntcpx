#!/usr/bin/env python3
"""
Run All Tests for py_ntcpx
==========================

Runs the full test suite via pytest (same as CI).
Optionally writes a JUnit-style report to test_reports/.

Usage:
    python run_all_tests.py [--output_dir test_reports] [--verbose]

Or use pytest directly:
    pytest -q
    pytest -v
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Windows-safe encoding
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Run all tests for py_ntcpx via pytest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_reports",
        help="Directory for test reports (default: test_reports)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose pytest output",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "pytest_report.xml"

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "test_ntcp_pipeline.py",
        "test_data_validation.py",
        "--tb=short",
        "-q" if not args.verbose else "-v",
        f"--junitxml={report_path}",
    ]

    try:
        result = subprocess.run(cmd)
    except FileNotFoundError:
        print("pytest not found. Install with: pip install pytest")
        return 1
    if result.returncode == 0:
        print(f"\nTest report: {report_path}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
