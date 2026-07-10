#!/usr/bin/env python3
"""Deterministic first-tool protocol evaluator for an OpenAI-compatible endpoint."""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
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
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("endpoint returned non-object JSON")
    return result


def first_tool(resp: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None, str | None]:
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, None, "missing choices"
    message = choices[0].get("message", {})
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    if not isinstance(calls, list) or not calls:
        return None, None, "missing structured tool_calls"
    fn = calls[0].get("function", {})
    name = fn.get("name")
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError as exc:
            return str(name) if name else None, None, f"arguments are invalid JSON: {exc}"
    if not isinstance(args, dict):
        return str(name) if name else None, None, "arguments are not an object"
    return str(name) if name else None, args, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen36-claude-code")
    ap.add_argument("--cases", type=Path, default=Path("examples/tool_eval_cases.json"))
    ap.add_argument("--output", type=Path, default=Path("outputs/tool_eval_results.json"))
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    args = ap.parse_args()
    tools = [
        {"type": "function", "function": {"name": "Read", "description": "Read a file", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}},
        {"type": "function", "function": {"name": "Bash", "description": "Run a shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
        {"type": "function", "function": {"name": "Edit", "description": "Replace an exact string", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}, "required": ["file_path", "old_string", "new_string"]}}},
    ]
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise SystemExit("evaluation cases must be a non-empty JSON list")

    results = []
    for case in cases:
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": case["prompt"]}],
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.0,
            "max_tokens": 512,
        }
        try:
            resp = post_json(
                args.base_url.rstrip("/") + "/chat/completions",
                payload,
                timeout=args.timeout,
                api_key=args.api_key,
            )
            got, arguments, error = first_tool(resp)
        except Exception as exc:
            got, arguments, error = None, None, str(exc)
        expected = case.get("expected_first_tools", [])
        passed = error is None and got in expected
        results.append(
            {
                "id": case.get("id"),
                "got_first_tool": got,
                "arguments": arguments,
                "error": error,
                "pass": passed,
                "expected_first_tools": expected,
            }
        )

    passed = sum(bool(result["pass"]) for result in results)
    report = {
        "model": args.model,
        "passed": passed,
        "total": len(results),
        "pass_rate": passed / len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
