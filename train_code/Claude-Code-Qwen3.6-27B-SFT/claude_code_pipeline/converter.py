"""Claude Code export -> Qwen-native tool-trace training rows.

The converter is intentionally conservative: preserve structured tool calls,
reject broken call/result ordering, mask non-assistant tokens from loss, and do
not write Qwen control tokens by hand. The tokenizer chat template owns
<think>, <tool_call>, and <tool_response> rendering.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class ConversionError(ValueError):
    """Raised when a Claude Code trace cannot be faithfully converted."""


@dataclass
class ParsedSession:
    session_id: str
    source_path: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    metadata: dict[str, Any]


SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private_key", re.compile("-----BEGIN (?:RSA |EC |OPENSSH |DSA )?" + "PRIVATE KEY-----")),
    ("generic_secret", re.compile(r"(?i)\b(secret|api[_-]?key|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{16,}")),
    ("github_token", re.compile(r"\b(?:" + "ghp" + r"|github_pat)_[A-Za-z0-9_]{20,}\b")),
    ("aws_access_key", re.compile(r"\b(?:" + "AKIA" + r"|ASIA)[A-Z0-9]{16}\b")),
]


def stable_id(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def split_name(name: str) -> str:
    """Normalize Claude/MCP tool names without losing their identity."""
    if not name:
        return "unknown"
    return str(name).strip().replace(" ", "_")


class ApproxTokenCounter:
    """Cheap deterministic token estimate for CI and no-tokenizer smoke tests."""

    def count_messages(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
        text = json.dumps({"messages": messages, "tools": tools or []}, ensure_ascii=False)
        return max(1, math.ceil(len(text) / 3.6))


class HFTokenCounter:
    """Exact rendered token counter using a target tokenizer's chat template."""

    def __init__(self, model: str, preserve_thinking: bool = True):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
        self.preserve_thinking = preserve_thinking

    def count_messages(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
        ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=True,
            add_generation_prompt=False,
            preserve_thinking=self.preserve_thinking,
        )
        return len(ids)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif item.get("type") == "tool_result":
                    parts.append(_as_text(item.get("content")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(_as_text(item.get("content")))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _arguments(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ConversionError(f"tool arguments are not JSON: {exc}") from exc
        if isinstance(parsed, dict):
            return parsed
    raise ConversionError(f"tool arguments must be a JSON object, got {type(value).__name__}")


def _claude_rows(path: Path) -> Iterable[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    if path.suffix == ".jsonl":
        rows = []
        for line_no, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ConversionError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
        return rows
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("messages", "conversation", "events", "transcript"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    raise ConversionError(f"unsupported JSON root: {type(data).__name__}")


def _message_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _claude_rows(path):
        if not isinstance(row, dict):
            continue
        if isinstance(row.get("message"), dict):
            row = row["message"]
        if row.get("role") or row.get("type") in {"user", "assistant", "system", "tool_result"}:
            rows.append(row)
    return rows


def parse_session_file(path: str | Path, *, secret_policy: str = "quarantine") -> ParsedSession:
    """Parse one Claude Code session export into structured Qwen/HF messages."""
    p = Path(path)
    source_rows = _message_rows(p)
    messages: list[dict[str, Any]] = []

    for idx, row in enumerate(source_rows):
        role = row.get("role") or row.get("type")
        content = row.get("content", "")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": _as_text(content)}]

        if role == "system":
            text = _as_text(blocks)
            if text:
                messages.append({"role": "system", "content": text, "train": False})
            continue

        if role == "user":
            user_text: list[str] = []
            for block in blocks:
                if not isinstance(block, dict):
                    user_text.append(str(block))
                    continue
                btype = block.get("type")
                if btype == "tool_result":
                    call_id = block.get("tool_use_id") or block.get("id") or block.get("tool_call_id")
                    if not call_id:
                        raise ConversionError(f"tool_result without tool_use_id in {p}:{idx}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": str(call_id),
                        "name": split_name(str(block.get("name") or "tool")),
                        "content": _as_text(block.get("content")),
                        "train": False,
                    })
                elif btype == "text" or "text" in block:
                    user_text.append(str(block.get("text", "")))
                else:
                    user_text.append(_as_text(block))
            joined = "\n\n".join(x for x in user_text if x).strip()
            if joined:
                messages.append({"role": "user", "content": joined, "train": False})
            continue

        if role == "assistant":
            text_parts: list[str] = []
            reasoning: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in blocks:
                if not isinstance(block, dict):
                    text_parts.append(str(block))
                    continue
                btype = block.get("type")
                if btype in {"thinking", "reasoning"}:
                    reasoning.append(str(block.get("thinking") or block.get("text") or ""))
                elif btype == "tool_use":
                    name = split_name(str(block.get("name") or "tool"))
                    call_id = str(block.get("id") or f"call_{stable_id([p.name, idx, name, block.get('input')])}")
                    tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": _arguments(block.get("input") or block.get("arguments") or {})},
                    })
                elif btype == "text" or "text" in block:
                    text_parts.append(str(block.get("text", "")))
                else:
                    text_parts.append(_as_text(block))
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": "\n\n".join(x for x in text_parts if x).strip(),
                "reasoning_content": "\n\n".join(x for x in reasoning if x).strip(),
                "train": True,
            }
            if tool_calls:
                msg["tool_calls"] = tool_calls
            if msg["content"] or msg["reasoning_content"] or tool_calls:
                messages.append(msg)
            continue

    if not messages:
        raise ConversionError(f"no messages found in {p}")
    if secret_policy == "quarantine" and find_secrets(messages):
        raise ConversionError(f"secret-like content found in {p}")
    validate_trace(messages)
    tools = build_tool_registry(messages)
    return ParsedSession(stable_id([str(p), messages]), str(p), messages, tools, {"source_file": str(p)})


def validate_trace(messages: list[dict[str, Any]]) -> None:
    pending: dict[str, str] = {}
    seen_user = False
    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ConversionError(f"messages[{i}] invalid role {role!r}")
        if role == "system" and seen_user:
            raise ConversionError("late system message would corrupt chat-template ordering")
        if role == "user":
            seen_user = True
        if role == "assistant":
            for call in msg.get("tool_calls") or []:
                call_id = call.get("id")
                name = call.get("function", {}).get("name")
                args = call.get("function", {}).get("arguments")
                if not call_id or not name or not isinstance(args, dict):
                    raise ConversionError(f"messages[{i}] malformed tool call")
                pending[str(call_id)] = str(name)
        if role == "tool":
            call_id = str(msg.get("tool_call_id") or "")
            if call_id not in pending:
                raise ConversionError(f"messages[{i}] orphan tool result {call_id!r}")
            pending.pop(call_id, None)
    if pending:
        raise ConversionError(f"unresolved tool calls: {sorted(pending)}")


def _schema_for_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, list):
        return {"type": "array", "items": {"type": "string"}}
    if isinstance(value, dict):
        return {"type": "object", "additionalProperties": True}
    return {"type": "string"}


def build_tool_registry(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    schemas: dict[str, dict[str, Any]] = {}
    for msg in messages:
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", {})
            name = split_name(str(fn.get("name") or "tool"))
            args = fn.get("arguments") if isinstance(fn.get("arguments"), dict) else {}
            entry = schemas.setdefault(name, {"type": "object", "properties": {}, "additionalProperties": True})
            for key, value in args.items():
                entry["properties"].setdefault(str(key), _schema_for_value(value))
    return [
        {"type": "function", "function": {"name": name, "description": f"Claude Code tool: {name}", "parameters": schema}}
        for name, schema in sorted(schemas.items())
    ]


def tools_for_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return build_tool_registry(messages)


def split_episodes(session: ParsedSession) -> list[dict[str, Any]]:
    return [{"messages": session.messages, "tools": session.tools, "metadata": session.metadata | {"session_id": session.session_id}}]


def _clone_with_train(messages: list[dict[str, Any]], train: bool | None = None) -> list[dict[str, Any]]:
    out = json.loads(json.dumps(messages, ensure_ascii=False))
    if train is not None:
        for msg in out:
            msg["train"] = bool(train and msg.get("role") == "assistant")
    return out


def _atoms(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    atoms: list[list[dict[str, Any]]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        atom = [msg]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            pending = {str(c["id"]) for c in msg.get("tool_calls", [])}
            j = i + 1
            while j < len(messages) and pending:
                atom.append(messages[j])
                if messages[j].get("role") == "tool":
                    pending.discard(str(messages[j].get("tool_call_id")))
                j += 1
            i = j
        else:
            i += 1
        atoms.append(atom)
    return atoms


def chunk_episode(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    counter: ApproxTokenCounter | HFTokenCounter | None = None,
    max_tokens: int = 8192,
    overlap_messages: int = 6,
) -> list[dict[str, Any]]:
    counter = counter or ApproxTokenCounter()
    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    for atom in _atoms(messages):
        candidate = current + atom
        if current and counter.count_messages(candidate, tools) > max_tokens:
            chunks.append({"messages": _clone_with_train(current), "tools": tools})
            overlap = current[-overlap_messages:] if overlap_messages else []
            current = _clone_with_train(overlap, train=False) + atom
            if counter.count_messages(current, tools) > max_tokens:
                raise ConversionError("one atomic tool transaction exceeds max_tokens")
        else:
            current = candidate
    if current:
        chunks.append({"messages": _clone_with_train(current), "tools": tools})
    return chunks


def find_secrets(value: Any) -> list[str]:
    hits: list[str] = []
    if isinstance(value, str):
        for name, pattern in SECRET_PATTERNS:
            if pattern.search(value):
                hits.append(name)
    elif isinstance(value, dict):
        for item in value.values():
            hits.extend(find_secrets(item))
    elif isinstance(value, list):
        for item in value:
            hits.extend(find_secrets(item))
    return sorted(set(hits))
