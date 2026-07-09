#!/usr/bin/env python3
"""Prepare Claude Code exports for Qwen3.6 Axolotl SFT."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

RECIPE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RECIPE_DIR))

from claude_code_pipeline.converter import ApproxTokenCounter, ConversionError, HFTokenCounter, chunk_episode, parse_session_file


def iter_inputs(path: Path):
    if path.is_file():
        yield path
        return
    for suffix in ("*.jsonl", "*.json"):
        yield from sorted(path.rglob(suffix))


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--model", default="")
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--validation-ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--secret-policy", choices=["quarantine", "allow"], default="quarantine")
    args = ap.parse_args()

    counter = HFTokenCounter(args.model) if args.model else ApproxTokenCounter()
    rows = []
    rejects = []
    for path in iter_inputs(args.input):
        try:
            session = parse_session_file(path, secret_policy=args.secret_policy)
            for episode in [{"messages": session.messages, "tools": session.tools}]:
                for idx, chunk in enumerate(chunk_episode(episode["messages"], episode["tools"], counter, args.max_seq_length)):
                    chunk["metadata"] = session.metadata | {"session_id": session.session_id, "chunk_index": idx}
                    rows.append(chunk)
        except Exception as exc:
            rejects.append({"source_path": str(path), "reason": type(exc).__name__, "detail": str(exc)})

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    val_n = max(1, int(len(rows) * args.validation_ratio)) if len(rows) > 1 else 0
    validation = rows[:val_n]
    train = rows[val_n:]

    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    write_jsonl(args.output_dir / "rejected.jsonl", rejects)
    report = {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "rejected_rows": len(rejects),
        "max_seq_length": args.max_seq_length,
    }
    (args.output_dir / "REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if train else 1


if __name__ == "__main__":
    raise SystemExit(main())
