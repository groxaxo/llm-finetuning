#!/usr/bin/env python3
"""Copy Claude Code session transcripts into an immutable raw-data snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=dst.parent)
    os.close(fd)
    try:
        shutil.copy2(src, tmp_name)
        os.replace(tmp_name, dst)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def default_roots(claude_home: Path) -> list[Path]:
    return [root for root in (claude_home / "projects", claude_home / "sessions", claude_home / "transcripts") if root.is_dir()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude-home", type=Path, default=Path.home() / ".claude")
    ap.add_argument("--source", type=Path, action="append", default=[], help="Explicit session root; repeatable")
    ap.add_argument("--output-dir", type=Path, default=Path("data/raw/claude-code"))
    ap.add_argument("--include-json", action="store_true", help="Also copy .json files; JSONL-only is safer by default")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    roots = args.source or default_roots(args.claude_home)
    if not roots:
        raise SystemExit(f"no Claude Code session roots found under {args.claude_home}")
    output_resolved = args.output_dir.resolve()
    suffixes = {".jsonl"} | ({".json"} if args.include_json else set())

    candidates: list[tuple[Path, Path]] = []
    for root in roots:
        if not root.is_dir():
            raise SystemExit(f"session root does not exist: {root}")
        root_resolved = root.resolve()
        for src in sorted(root.rglob("*")):
            if not src.is_file() or src.suffix.lower() not in suffixes:
                continue
            resolved = src.resolve()
            if resolved == output_resolved or output_resolved in resolved.parents:
                continue
            rel = Path(root.name) / resolved.relative_to(root_resolved)
            candidates.append((src, args.output_dir / rel))

    if not candidates:
        raise SystemExit("no session transcript files found")

    manifest = []
    for src, dst in candidates:
        record = {
            "source": str(src),
            "destination": str(dst),
            "size_bytes": src.stat().st_size,
            "sha256": sha256_file(src),
        }
        manifest.append(record)
        print(f"{src} -> {dst}")
        if not args.dry_run:
            copy_atomic(src, dst)

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "MANIFEST.json").write_text(
            json.dumps({"files": manifest}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"session files {'found' if args.dry_run else 'copied'}: {len(candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
