#!/usr/bin/env python3
"""Check that a merged/exported checkpoint still contains MTP/next-token prediction tensors."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    args = ap.parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")
    hits = []
    for path in args.checkpoint.rglob("*"):
        name = path.name.lower()
        if any(key in name for key in ("mtp", "nextn", "next_n", "multi_token")):
            hits.append(str(path))
    if hits:
        print("MTP-like files/tensors detected:")
        for hit in hits[:50]:
            print(hit)
        return 0
    print("No obvious MTP file names found. If this is an adapter-only checkpoint, compare against the base model before merge.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
