#!/usr/bin/env python3
"""
cleanup_legacy_outputs.py
==========================

Utility script to remove legacy, now-redundant output folders from older runs:

- structured_output/           (old consolidated copy tree)
- code3_output/DR_plots/       (old duplicate DR plots; now under code3_output/plots/)

By default it cleans these folders under the current working directory's
pipeline base (e.g. out2/), but you can point it to any base directory.

Usage examples (from repository root):

  python cleanup_legacy_outputs.py                # clean ./out2 by default
  python cleanup_legacy_outputs.py --base out2
  python cleanup_legacy_outputs.py --base out2 --base out3_run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def remove_dir_if_exists(path: Path) -> None:
    """Remove directory tree if it exists, logging to stdout."""
    if path.exists():
        if path.is_dir():
            print(f"[CLEANUP] Removing directory tree: {path}")
            try:
                shutil.rmtree(path)
            except Exception as e:
                print(f"[WARN] Could not remove {path}: {e}")
        else:
            print(f"[SKIP] {path} exists but is not a directory")
    else:
        print(f"[SKIP] Not found (already clean): {path}")


def cleanup_base(base_dir: Path) -> None:
    """Clean legacy folders under a single pipeline base directory."""
    base_dir = base_dir.resolve()
    print(f"\n=== Cleaning legacy outputs under base: {base_dir} ===")

    # Old consolidated tree
    structured = base_dir / "structured_output"

    # Old DR_plots tree (now replaced by code3_output/plots/)
    dr_plots = base_dir / "code3_output" / "DR_plots"

    remove_dir_if_exists(structured)
    remove_dir_if_exists(dr_plots)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove legacy structured_output/ and code3_output/DR_plots/ from older runs."
    )
    parser.add_argument(
        "--base",
        dest="bases",
        action="append",
        default=None,
        help="Pipeline base directory to clean (default: out2 in current working directory). "
             "Can be specified multiple times.",
    )

    args = parser.parse_args()

    bases = args.bases or ["out2"]
    for b in bases:
        cleanup_base(Path(b))

    print("\n[OK] Legacy cleanup completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

