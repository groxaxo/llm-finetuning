#!/usr/bin/env python3
"""Tiny first-tool behavior evaluator for the trained endpoint."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


def post_json(url: str, payload: dict):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def first_tool(resp: dict) -> str | None:
    msg = resp.get("choices", [{}])[0].get("message", {})
    calls = msg.get("tool_calls") or []
    if not calls:
        return None
    return calls[0].get("function", {}).get("name")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen36-claude-code")
    ap.add_argument("--cases", type=Path, default=Path("examples/tool_eval_cases.json"))
    ap.add_argument("--output", type=Path, default=Path("outputs/tool_eval_results.json"))
    args = ap.parse_args()
    tools = [
        {"type": "function", "function": {"name": "Read", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "Bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "Edit", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}}}},
    ]
    cases = json.loads(args.cases.read_text())
    results = []
    for case in cases:
        payload = {"model": args.model, "messages": [{"role": "user", "content": case["prompt"]}], "tools": tools, "tool_choice": "auto", "temperature": 0.0, "max_tokens": 512}
        resp = post_json(args.base_url.rstrip("/") + "/chat/completions", payload)
        got = first_tool(resp)
        expected = case.get("expected_first_tools", [])
        results.append({"id": case.get("id"), "got_first_tool": got, "pass": got in expected, "expected_first_tools": expected})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
