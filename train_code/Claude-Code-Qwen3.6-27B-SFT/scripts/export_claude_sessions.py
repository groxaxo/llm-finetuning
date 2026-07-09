#!/usr/bin/env python3
"""Copy Claude Code JSON/JSONL session exports into this recipe's raw-data area."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude-home", type=Path, default=Path.home() / ".claude")
    ap.add_argument("--output-dir", type=Path, default=Path("data/raw/claude-code"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    candidates = []
    for suffix in ("*.jsonl", "*.json"):
        candidates.extend(args.claude_home.rglob(suffix))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(set(candidates)):
        if not src.is_file():
            continue
        rel = src.relative_to(args.claude_home)
        dst = args.output_dir / rel
        print(f"{src} -> {dst}")
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        copied += 1
    print(f"sessions copied: {copied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
