#!/usr/bin/env python3
"""Prepare Claude Code exports for Qwen3.6 Axolotl SFT."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

RECIPE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RECIPE_DIR))

from claude_code_pipeline.converter import (  # noqa: E402
    ApproxTokenCounter,
    HFTokenCounter,
    SCHEMA_VERSION,
    chunk_episode,
    parse_session_file,
    to_arrow_safe_row,
    tools_for_messages,
)


def iter_inputs(path: Path, *, excluded_dir: Path | None = None) -> Iterable[Path]:
    if not path.exists():
        raise FileNotFoundError(path)
    excluded = excluded_dir.resolve() if excluded_dir else None
    if path.is_file():
        if path.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError(f"unsupported input file: {path}")
        yield path
        return
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file() or candidate.suffix.lower() not in {".json", ".jsonl"}:
            continue
        resolved = candidate.resolve()
        if excluded and (resolved == excluded or excluded in resolved.parents):
            continue
        yield candidate


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def split_score(content_hash: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{content_hash}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def load_tool_registry(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("tools")
    if not isinstance(data, list) or not all(isinstance(tool, dict) for tool in data):
        raise ValueError("tool registry must be a JSON list or an object containing a tools list")
    names = [tool.get("function", {}).get("name") for tool in data]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("tool registry has missing or duplicate function names")
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--model", default="")
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--overlap-messages", type=int, default=6)
    ap.add_argument("--validation-ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--secret-policy", choices=["redact", "quarantine", "allow"], default="redact")
    ap.add_argument("--tool-registry", type=Path)
    args = ap.parse_args()

    if not 0.0 <= args.validation_ratio < 1.0:
        ap.error("--validation-ratio must be in [0, 1)")
    if args.max_seq_length <= 0:
        ap.error("--max-seq-length must be positive")

    registry = load_tool_registry(args.tool_registry)
    counter = (
        HFTokenCounter(args.model, revision=args.model_revision)
        if args.model
        else ApproxTokenCounter()
    )

    groups: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    seen_content: dict[str, str] = {}
    reject_reasons: Counter[str] = Counter()
    inputs = list(iter_inputs(args.input, excluded_dir=args.output_dir))
    if not inputs:
        raise SystemExit(f"no JSON/JSONL session files found under {args.input}")

    for path in inputs:
        try:
            session = parse_session_file(path, secret_policy=args.secret_policy)
            content_hash = str(session.metadata["content_hash"])
            if content_hash in seen_content:
                reason = "exact_duplicate_session"
                rejects.append(
                    {
                        "source_path": str(path),
                        "reason": reason,
                        "detail": f"duplicate of {seen_content[content_hash]}",
                        "content_hash": content_hash,
                    }
                )
                reject_reasons[reason] += 1
                continue
            seen_content[content_hash] = str(path)

            session_registry = registry or session.tools
            chunks = chunk_episode(
                session.messages,
                session_registry,
                counter,
                max_tokens=args.max_seq_length,
                overlap_messages=args.overlap_messages,
            )
            prepared_chunks: list[dict[str, Any]] = []
            for idx, chunk in enumerate(chunks):
                chunk["tools"] = tools_for_messages(chunk["messages"], session_registry)
                chunk["metadata"] = session.metadata | {
                    "session_id": session.session_id,
                    "chunk_index": idx,
                    "chunk_count": len(chunks),
                }
                prepared_chunks.append(to_arrow_safe_row(chunk))
            groups.append(
                {
                    "content_hash": content_hash,
                    "session_id": session.session_id,
                    "score": split_score(content_hash, args.seed),
                    "chunks": prepared_chunks,
                }
            )
        except Exception as exc:
            reason = type(exc).__name__
            rejects.append({"source_path": str(path), "reason": reason, "detail": str(exc)})
            reject_reasons[reason] += 1

    if not groups:
        write_jsonl_atomic(args.output_dir / "rejected.jsonl", rejects)
        raise SystemExit("all sessions were rejected; inspect rejected.jsonl")

    validation_groups = [g for g in groups if g["score"] < args.validation_ratio]
    train_groups = [g for g in groups if g["score"] >= args.validation_ratio]
    if args.validation_ratio > 0 and len(groups) > 1 and not validation_groups:
        chosen = min(train_groups, key=lambda g: g["score"])
        train_groups.remove(chosen)
        validation_groups.append(chosen)
    if not train_groups and len(validation_groups) > 1:
        chosen = max(validation_groups, key=lambda g: g["score"])
        validation_groups.remove(chosen)
        train_groups.append(chosen)

    rng = random.Random(args.seed)
    rng.shuffle(train_groups)
    rng.shuffle(validation_groups)
    train = [row for group in train_groups for row in group["chunks"]]
    validation = [row for group in validation_groups for row in group["chunks"]]

    train_sessions = {g["content_hash"] for g in train_groups}
    val_sessions = {g["content_hash"] for g in validation_groups}
    if train_sessions & val_sessions:
        raise AssertionError("session leakage between train and validation")

    write_jsonl_atomic(args.output_dir / "train.jsonl", train)
    write_jsonl_atomic(args.output_dir / "validation.jsonl", validation)
    write_jsonl_atomic(args.output_dir / "rejected.jsonl", rejects)
    report = {
        "schema_version": SCHEMA_VERSION,
        "input_files": len(inputs),
        "unique_sessions": len(groups),
        "train_sessions": len(train_groups),
        "validation_sessions": len(validation_groups),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "rejected_rows": len(rejects),
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "max_seq_length": args.max_seq_length,
        "overlap_messages": args.overlap_messages,
        "secret_policy": args.secret_policy,
        "tokenizer_model": args.model or None,
        "tokenizer_revision": args.model_revision,
        "seed": args.seed,
        "session_leakage": False,
    }
    report_path = args.output_dir / "REPORT.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if train else 1


if __name__ == "__main__":
    raise SystemExit(main())
