#!/usr/bin/env python3
"""Smoke-test a trained model through an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import json
import urllib.request


def post_json(url: str, payload: dict):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen36-claude-code")
    args = ap.parse_args()
    tools = [
        {"type": "function", "function": {"name": "Read", "description": "Read a file", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}},
        {"type": "function", "function": {"name": "Bash", "description": "Run a shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    ]
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "A parser test is failing. Inspect the relevant files before editing."}],
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.2,
        "max_tokens": 512,
    }
    result = post_json(args.base_url.rstrip("/") + "/chat/completions", payload)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
