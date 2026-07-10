#!/usr/bin/env python3
"""Smoke-test structured tool calling through an OpenAI-compatible endpoint."""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from typing import Any


def post_json(url: str, payload: dict[str, Any], *, timeout: int, api_key: str | None) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("endpoint returned a non-object JSON response")
    return result


def validate_first_call(result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("response has no choices")
    message = choices[0].get("message", {})
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    if not isinstance(calls, list) or not calls:
        raise RuntimeError("model did not emit a structured tool call")
    fn = calls[0].get("function", {})
    name = fn.get("name")
    arguments = fn.get("arguments")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not name or not isinstance(arguments, dict):
        raise RuntimeError("first tool call has invalid name/arguments")
    return str(name), arguments


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen36-claude-code")
    ap.add_argument("--prompt", default="A parser test is failing. Inspect the relevant files before editing.")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    args = ap.parse_args()
    tools = [
        {"type": "function", "function": {"name": "Read", "description": "Read a file", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}},
        {"type": "function", "function": {"name": "Bash", "description": "Run a shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    ]
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.0,
        "max_tokens": 512,
    }
    result = post_json(
        args.base_url.rstrip("/") + "/chat/completions",
        payload,
        timeout=args.timeout,
        api_key=args.api_key,
    )
    name, arguments = validate_first_call(result)
    print(json.dumps({"first_tool": name, "arguments": arguments, "response": result}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
