#!/usr/bin/env python3
"""Validate prepared Qwen tool-trace JSONL before Axolotl training."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

RECIPE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RECIPE_DIR))

from claude_code_pipeline.converter import (  # noqa: E402
    ApproxTokenCounter,
    HFTokenCounter,
    SCHEMA_VERSION,
    from_arrow_safe_row,
    validate_trace,
)


def load_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be an object")
            yield line_no, row


def canonical_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_file(path: Path, counter, max_seq_length: int, render_count: int) -> dict[str, Any]:
    rows = 0
    max_tokens = 0
    tool_rows = 0
    trainable_assistant_turns = 0
    sessions: set[str] = set()
    content_hashes: set[str] = set()
    row_hashes: set[str] = set()

    for line_no, stored_row in load_jsonl(path):
        rows += 1
        row_hash = canonical_hash(stored_row)
        if row_hash in row_hashes:
            raise ValueError(f"{path}:{line_no}: exact duplicate row")
        row_hashes.add(row_hash)

        row = from_arrow_safe_row(stored_row)
        messages = row.get("messages")
        tools = row.get("tools", [])
        metadata = row.get("metadata")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{path}:{line_no}: missing messages list")
        if not isinstance(metadata, dict):
            raise ValueError(f"{path}:{line_no}: missing metadata")
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{path}:{line_no}: schema_version={metadata.get('schema_version')!r}, expected {SCHEMA_VERSION!r}"
            )
        session_id = str(metadata.get("session_id") or "")
        content_hash = str(metadata.get("content_hash") or "")
        if not session_id or not content_hash:
            raise ValueError(f"{path}:{line_no}: missing session_id/content_hash")
        sessions.add(session_id)
        content_hashes.add(content_hash)

        validate_trace(messages)
        for index, msg in enumerate(messages):
            train = msg.get("train")
            if not isinstance(train, bool):
                raise ValueError(f"{path}:{line_no}: messages[{index}].train must be boolean")
            if msg.get("role") != "assistant" and train:
                raise ValueError(f"{path}:{line_no}: non-assistant message marked trainable")
            if msg.get("role") == "assistant" and train:
                trainable_assistant_turns += 1

        used_tools = {
            call.get("function", {}).get("name")
            for msg in messages
            for call in (msg.get("tool_calls") or [])
        }
        declared_tools = {
            tool.get("function", {}).get("name")
            for tool in tools
            if isinstance(tool, dict)
        }
        if used_tools != declared_tools:
            raise ValueError(
                f"{path}:{line_no}: tool registry mismatch used={sorted(used_tools)} declared={sorted(declared_tools)}"
            )

        tokens = counter.count_messages(messages, tools)
        max_tokens = max(max_tokens, tokens)
        if tokens > max_seq_length:
            raise ValueError(f"{path}:{line_no}: {tokens} tokens exceeds {max_seq_length}")
        if used_tools:
            tool_rows += 1

        if render_count and rows <= render_count:
            print(f"\n--- EXACT RENDER {path.name}:{line_no} tokens={tokens} ---")
            print(counter.render_text(messages, tools))

    return {
        "path": str(path),
        "rows": rows,
        "sessions": sessions,
        "content_hashes": content_hashes,
        "row_hashes": row_hashes,
        "tool_rows": tool_rows,
        "trainable_assistant_turns": trainable_assistant_turns,
        "max_tokens": max_tokens,
    }


def public_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (len(value) if isinstance(value, set) else value)
        for key, value in result.items()
        if key not in {"row_hashes"}
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--validation", type=Path, required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--render-count", type=int, default=0)
    args = ap.parse_args()

    counter = (
        HFTokenCounter(args.model, revision=args.model_revision)
        if args.model
        else ApproxTokenCounter()
    )
    train = validate_file(args.train, counter, args.max_seq_length, args.render_count)
    validation = validate_file(args.validation, counter, args.max_seq_length, args.render_count)

    session_overlap = train["content_hashes"] & validation["content_hashes"]
    row_overlap = train["row_hashes"] & validation["row_hashes"]
    if session_overlap:
        raise ValueError(f"train/validation session leakage: {sorted(session_overlap)[:10]}")
    if row_overlap:
        raise ValueError(f"train/validation exact row leakage: {len(row_overlap)} rows")
    if train["rows"] == 0:
        raise ValueError("training file is empty")

    summary = {
        "train": public_summary(train),
        "validation": public_summary(validation),
        "session_leakage": False,
        "exact_row_leakage": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
