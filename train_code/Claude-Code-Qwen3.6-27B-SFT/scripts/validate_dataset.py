#!/usr/bin/env python3
"""Validate prepared Qwen tool-trace JSONL before Axolotl training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RECIPE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RECIPE_DIR))

from claude_code_pipeline.converter import ApproxTokenCounter, HFTokenCounter, validate_trace


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                yield line_no, json.loads(line)


def validate_file(path: Path, counter, max_seq_length: int, render_count: int):
    rows = 0
    max_tokens = 0
    tool_rows = 0
    for line_no, row in load_jsonl(path):
        rows += 1
        messages = row.get("messages")
        tools = row.get("tools", [])
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{path}:{line_no}: missing messages list")
        validate_trace(messages)
        tokens = counter.count_messages(messages, tools)
        max_tokens = max(max_tokens, tokens)
        if tokens > max_seq_length:
            raise ValueError(f"{path}:{line_no}: {tokens} tokens exceeds {max_seq_length}")
        if any(m.get("tool_calls") for m in messages):
            tool_rows += 1
        if render_count and rows <= render_count:
            print(f"\n--- {path.name}:{line_no} tokens={tokens} ---")
            for msg in messages[:8]:
                role = msg.get("role")
                print(f"{role}: {str(msg.get('content',''))[:200]}")
                if msg.get("reasoning_content"):
                    print(f"reasoning: {msg['reasoning_content'][:200]}")
                if msg.get("tool_calls"):
                    print(json.dumps(msg["tool_calls"], ensure_ascii=False)[:500])
    return {"path": str(path), "rows": rows, "tool_rows": tool_rows, "max_tokens": max_tokens}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--validation", type=Path, required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--render-count", type=int, default=0)
    args = ap.parse_args()
    counter = HFTokenCounter(args.model) if args.model else ApproxTokenCounter()
    summary = [
        validate_file(args.train, counter, args.max_seq_length, args.render_count),
        validate_file(args.validation, counter, args.max_seq_length, args.render_count),
    ]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
