"""Claude Code export -> Qwen-native tool-trace training rows.

The converter is deliberately conservative. It repairs only lossless protocol
artifacts, preserves native structured calls/results, rejects ambiguous causal
ordering, and leaves Qwen control-token rendering to the target tokenizer.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "claude-code-qwen-tooltrace/v2"
FORBIDDEN_CONTROL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
)


class ConversionError(ValueError):
    """Raised when a Claude Code trace cannot be faithfully converted."""


@dataclass
class ParsedSession:
    session_id: str
    source_path: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    metadata: dict[str, Any]


ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
LOCAL_CAVEAT_RE = re.compile(
    r"<local-command-caveat\b[^>]*>.*?</local-command-caveat>", re.I | re.S
)
COMMAND_NAME_RE = re.compile(r"<command-name\b[^>]*>.*?</command-name>", re.I | re.S)
STANDALONE_CLIENT_COMMAND_RE = re.compile(
    r"(?im)^\s*/(?:model|effort)(?:\s+[^\r\n]*)?\s*$"
)
TOOL_CALL_MARKER_RE = re.compile(r"\[TOOL_CALL:\s*([^\]\r\n]+?)\s*\]", re.I)

SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "private_key",
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----.*?"
            r"-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
            re.S,
        ),
    ),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("openai_or_anthropic_key", re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("bearer_token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{20,}")),
    (
        "generic_secret",
        re.compile(
            r"(?i)\b(?:secret|api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd)"
            r"\s*[:=]\s*['\"]?[A-Za-z0-9_./+=:@-]{16,}"
        ),
    ),
]

KNOWN_TOOL_DESCRIPTIONS = {
    "Bash": "Run a shell command and return its output.",
    "Read": "Read a file or a bounded range from a file.",
    "Edit": "Replace an exact string in an existing file.",
    "Write": "Write complete content to a file.",
    "Glob": "Find files matching a glob pattern.",
    "Grep": "Search file contents for a pattern.",
}


def stable_id(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def split_name(name: str) -> str:
    """Normalize whitespace in a tool name without collapsing MCP identity."""
    if not name:
        return "unknown"
    return re.sub(r"\s+", "_", str(name).strip())


def find_secrets(value: Any) -> list[str]:
    hits: list[str] = []
    if isinstance(value, str):
        for name, pattern in SECRET_PATTERNS:
            if pattern.search(value):
                hits.append(name)
    elif isinstance(value, Mapping):
        for item in value.values():
            hits.extend(find_secrets(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            hits.extend(find_secrets(item))
    return sorted(set(hits))


def _redact_text(text: str) -> str:
    for name, pattern in SECRET_PATTERNS:
        text = pattern.sub(f"[REDACTED_{name.upper()}]", text)
    return text


def sanitize_text(text: str, *, secret_policy: str = "redact") -> str:
    """Remove local-client artifacts and ANSI escapes; optionally redact secrets."""
    text = ANSI_ESCAPE_RE.sub("", text)
    text = LOCAL_CAVEAT_RE.sub("", text)
    text = COMMAND_NAME_RE.sub("", text)
    text = STANDALONE_CLIENT_COMMAND_RE.sub("", text)
    if secret_policy == "redact":
        text = _redact_text(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def sanitize_value(value: Any, *, secret_policy: str = "redact") -> Any:
    if isinstance(value, str):
        return sanitize_text(value, secret_policy=secret_policy)
    if isinstance(value, Mapping):
        return {str(k): sanitize_value(v, secret_policy=secret_policy) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_value(v, secret_policy=secret_policy) for v in value]
    return value


class ApproxTokenCounter:
    """Cheap deterministic token estimate for CI and no-tokenizer smoke tests."""

    def count_messages(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
        text = json.dumps({"messages": messages, "tools": tools or []}, ensure_ascii=False, separators=(",", ":"))
        return max(1, math.ceil(len(text) / 3.6))

    def render_text(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> str:
        return json.dumps({"messages": messages, "tools": tools or []}, ensure_ascii=False, indent=2)


class HFTokenCounter:
    """Exact rendered token counter using the target tokenizer's own template."""

    def __init__(self, model: str, preserve_thinking: bool = True, revision: str | None = None):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            revision=revision,
            trust_remote_code=False,
            use_fast=True,
        )
        self.preserve_thinking = preserve_thinking

    def render_text(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=False,
            preserve_thinking=self.preserve_thinking,
        )

    def count_messages(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=True,
            add_generation_prompt=False,
            preserve_thinking=self.preserve_thinking,
            return_dict=False,
        )
        if isinstance(rendered, dict):
            rendered = rendered["input_ids"]
        return len(rendered)


def _as_text(value: Any, *, secret_policy: str = "redact") -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return sanitize_text(value, secret_policy=secret_policy)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(_as_text(item.get("text"), secret_policy=secret_policy))
                elif item.get("type") == "tool_result":
                    parts.append(_as_text(item.get("content"), secret_policy=secret_policy))
                elif "text" in item:
                    parts.append(_as_text(item.get("text"), secret_policy=secret_policy))
                elif "content" in item:
                    parts.append(_as_text(item.get("content"), secret_policy=secret_policy))
            else:
                parts.append(_as_text(item, secret_policy=secret_policy))
        return "\n".join(p for p in parts if p).strip()
    if isinstance(value, Mapping):
        sanitized = sanitize_value(value, secret_policy=secret_policy)
        return json.dumps(sanitized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def _arguments(value: Any, *, secret_policy: str = "redact") -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(sanitize_value(value, secret_policy=secret_policy))
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ConversionError(f"tool arguments are not JSON: {exc}") from exc
        if isinstance(parsed, dict):
            return dict(sanitize_value(parsed, secret_policy=secret_policy))
    raise ConversionError(f"tool arguments must be a JSON object, got {type(value).__name__}")


def _extract_inline_tool_calls(
    text: str,
    *,
    identity: Any,
    secret_policy: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Extract one or more ``[TOOL_CALL: Name] {json}`` markers from assistant text."""
    decoder = json.JSONDecoder()
    calls: list[dict[str, Any]] = []
    kept: list[str] = []
    cursor = 0
    occurrence = 0
    while True:
        match = TOOL_CALL_MARKER_RE.search(text, cursor)
        if not match:
            kept.append(text[cursor:])
            break
        kept.append(text[cursor:match.start()])
        tail = text[match.end():]
        leading = len(tail) - len(tail.lstrip())
        try:
            args, consumed = decoder.raw_decode(tail.lstrip())
        except json.JSONDecodeError as exc:
            raise ConversionError(f"malformed inline tool call {match.group(1)!r}: {exc}") from exc
        if not isinstance(args, dict):
            raise ConversionError(f"inline tool call {match.group(1)!r} arguments must be an object")
        name = split_name(match.group(1))
        args = dict(sanitize_value(args, secret_policy=secret_policy))
        call_id = f"call_{stable_id([identity, occurrence, name, args])}"
        calls.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": args}})
        occurrence += 1
        cursor = match.end() + leading + consumed
    return sanitize_text("".join(kept), secret_policy=secret_policy), calls


def _claude_rows(path: Path) -> Iterable[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConversionError(f"cannot read {path}: {exc}") from exc
    if not text.strip():
        return []
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line_no, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ConversionError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
        return rows
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConversionError(f"{path}: invalid JSON: {exc}") from exc
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
    for outer in _claude_rows(path):
        if not isinstance(outer, dict):
            continue
        row = dict(outer)
        if isinstance(outer.get("message"), dict):
            row = dict(outer["message"])
            row.setdefault("_event_type", outer.get("type"))
            row.setdefault("_uuid", outer.get("uuid"))
            row.setdefault("_session_id", outer.get("sessionId") or outer.get("session_id"))
        role = row.get("role") or row.get("type") or row.get("_event_type")
        if role in {"human", "user", "assistant", "model", "system", "tool", "tool_result"}:
            rows.append(row)
    return rows


def _tool_result_message(block: Mapping[str, Any], *, source: str, index: int, secret_policy: str) -> dict[str, Any]:
    call_id = block.get("tool_use_id") or block.get("tool_call_id") or block.get("id")
    if not call_id:
        raise ConversionError(f"tool_result without tool_use_id in {source}:{index}")
    content = _as_text(block.get("content"), secret_policy=secret_policy)
    if block.get("is_error"):
        content = json.dumps({"is_error": True, "content": content}, ensure_ascii=False, separators=(",", ":"))
    name = split_name(str(block.get("name") or "unknown"))
    return {
        "role": "tool",
        "tool_call_id": str(call_id),
        "name": name,
        "content": content,
        "train": False,
    }


def _merge_assistant(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(left)
    for key in ("reasoning_content", "content"):
        parts = [str(out.get(key, "")).strip(), str(right.get(key, "")).strip()]
        out[key] = "\n\n".join(p for p in parts if p)
    calls = list(out.get("tool_calls") or []) + list(right.get("tool_calls") or [])
    if calls:
        out["tool_calls"] = calls
    out["train"] = bool(out.get("train", True) or right.get("train", True))
    return out


def normalize_trace(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Losslessly repair adjacent same-role streaming fragments."""
    out: list[dict[str, Any]] = []
    for original in messages:
        msg = deepcopy(original)
        role = msg.get("role")
        if role in {"system", "user"} and not str(msg.get("content", "")).strip():
            continue
        if role == "assistant" and not (
            str(msg.get("content", "")).strip()
            or str(msg.get("reasoning_content", "")).strip()
            or msg.get("tool_calls")
        ):
            continue
        if out and out[-1].get("role") == role and role in {"system", "user", "assistant"}:
            if role == "assistant":
                out[-1] = _merge_assistant(out[-1], msg)
            else:
                left = str(out[-1].get("content", "")).strip()
                right = str(msg.get("content", "")).strip()
                out[-1]["content"] = "\n\n".join(p for p in (left, right) if p)
            continue
        out.append(msg)
    for msg in out:
        msg["train"] = bool(msg.get("role") == "assistant" and msg.get("train", True))
        if msg.get("role") == "assistant":
            msg.setdefault("content", "")
            msg.setdefault("reasoning_content", "")
    return out


def validate_trace(messages: list[dict[str, Any]]) -> None:
    pending: dict[str, str] = {}
    seen_call_ids: set[str] = set()
    seen_non_system = False
    seen_user = False
    trainable_assistants = 0

    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ConversionError(f"messages[{i}] invalid role {role!r}")
        if role == "system":
            if seen_non_system:
                raise ConversionError("late system message would corrupt chat-template ordering")
            continue
        seen_non_system = True

        if not seen_user and role != "user":
            raise ConversionError("first non-system message must be user")
        if role == "user":
            if pending:
                raise ConversionError(f"user message before tool calls resolved: {sorted(pending)}")
            seen_user = True
            msg["train"] = False
            continue
        if role == "assistant":
            if pending:
                raise ConversionError(f"assistant message before tool calls resolved: {sorted(pending)}")
            if msg.get("train", True):
                trainable_assistants += 1
            calls = msg.get("tool_calls") or []
            if not isinstance(calls, list):
                raise ConversionError(f"messages[{i}] tool_calls must be a list")
            for call in calls:
                if not isinstance(call, dict):
                    raise ConversionError(f"messages[{i}] malformed tool call")
                call_id = str(call.get("id") or "")
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                name = split_name(str(fn.get("name") or ""))
                args = fn.get("arguments")
                if not call_id or not name or name == "unknown" or not isinstance(args, dict):
                    raise ConversionError(f"messages[{i}] malformed tool call")
                if call_id in seen_call_ids:
                    raise ConversionError(f"duplicate tool call id {call_id!r}")
                seen_call_ids.add(call_id)
                pending[call_id] = name
            continue
        if role == "tool":
            call_id = str(msg.get("tool_call_id") or "")
            if call_id not in pending:
                raise ConversionError(f"messages[{i}] orphan or duplicate tool result {call_id!r}")
            expected = pending.pop(call_id)
            supplied = split_name(str(msg.get("name") or "unknown"))
            if supplied in {"unknown", "tool"}:
                msg["name"] = expected
            elif supplied != expected:
                raise ConversionError(
                    f"messages[{i}] tool name {supplied!r} does not match call {expected!r}"
                )
            msg["train"] = False

    if pending:
        raise ConversionError(f"unresolved tool calls: {sorted(pending)}")
    if not seen_user:
        raise ConversionError("trace has no user message")
    if trainable_assistants == 0:
        raise ConversionError("trace has no trainable assistant message")


def _forbidden_violations(messages: list[dict[str, Any]]) -> list[str]:
    violations: list[str] = []
    for i, msg in enumerate(messages):
        values = [msg.get("content", ""), msg.get("reasoning_content", "")]
        for call in msg.get("tool_calls") or []:
            values.append(json.dumps(call, ensure_ascii=False))
        hay = "\n".join(str(v) for v in values)
        for token in FORBIDDEN_CONTROL_TOKENS:
            if token in hay:
                violations.append(f"messages[{i}] contains target control token {token!r}")
    return violations


def parse_session_file(path: str | Path, *, secret_policy: str = "redact") -> ParsedSession:
    """Parse one Claude Code session export into structured Qwen/HF messages."""
    if secret_policy not in {"redact", "quarantine", "allow"}:
        raise ValueError("secret_policy must be redact, quarantine, or allow")
    p = Path(path)
    source_rows = _message_rows(p)
    messages: list[dict[str, Any]] = []
    source_session_ids: list[str] = []

    for idx, row in enumerate(source_rows):
        if row.get("_session_id"):
            source_session_ids.append(str(row["_session_id"]))
        role = row.get("role") or row.get("type") or row.get("_event_type")
        role = {"human": "user", "model": "assistant", "tool": "tool_result"}.get(role, role)
        content = row.get("content", "")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": _as_text(content, secret_policy=secret_policy)}]

        if role == "system":
            text = _as_text(blocks, secret_policy=secret_policy)
            if text:
                messages.append({"role": "system", "content": text, "train": False})
            continue

        if role == "tool_result":
            block = dict(row)
            if "content" not in block:
                block["content"] = content
            messages.append(_tool_result_message(block, source=str(p), index=idx, secret_policy=secret_policy))
            continue

        if role == "user":
            user_text: list[str] = []
            tool_results: list[dict[str, Any]] = []
            for block in blocks:
                if not isinstance(block, dict):
                    user_text.append(_as_text(block, secret_policy=secret_policy))
                    continue
                btype = block.get("type")
                if btype == "tool_result":
                    tool_results.append(
                        _tool_result_message(block, source=str(p), index=idx, secret_policy=secret_policy)
                    )
                elif btype == "text" or "text" in block:
                    user_text.append(_as_text(block.get("text", ""), secret_policy=secret_policy))
                else:
                    user_text.append(_as_text(block, secret_policy=secret_policy))
            messages.extend(tool_results)
            joined = "\n\n".join(x for x in user_text if x).strip()
            if joined:
                messages.append({"role": "user", "content": joined, "train": False})
            continue

        if role == "assistant":
            text_parts: list[str] = []
            reasoning: list[str] = []
            structured_calls: list[dict[str, Any]] = []
            inline_calls: list[dict[str, Any]] = []
            for block_idx, block in enumerate(blocks):
                if not isinstance(block, dict):
                    text, calls = _extract_inline_tool_calls(
                        _as_text(block, secret_policy=secret_policy),
                        identity=[idx, block_idx],
                        secret_policy=secret_policy,
                    )
                    if text:
                        text_parts.append(text)
                    inline_calls.extend(calls)
                    continue
                btype = block.get("type")
                if btype in {"thinking", "reasoning"}:
                    reasoning.append(
                        _as_text(block.get("thinking") or block.get("text") or "", secret_policy=secret_policy)
                    )
                elif btype == "tool_use":
                    name = split_name(str(block.get("name") or "unknown"))
                    args = _arguments(block.get("input") if "input" in block else block.get("arguments"), secret_policy=secret_policy)
                    call_id = str(block.get("id") or f"call_{stable_id([idx, block_idx, name, args])}")
                    structured_calls.append(
                        {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}
                    )
                elif btype == "text" or "text" in block:
                    text, calls = _extract_inline_tool_calls(
                        _as_text(block.get("text", ""), secret_policy=secret_policy),
                        identity=[idx, block_idx],
                        secret_policy=secret_policy,
                    )
                    if text:
                        text_parts.append(text)
                    inline_calls.extend(calls)
                else:
                    text = _as_text(block, secret_policy=secret_policy)
                    if text:
                        text_parts.append(text)

            calls: list[dict[str, Any]] = list(structured_calls)
            structured_signatures = {
                stable_id([call["function"]["name"], call["function"]["arguments"]])
                for call in structured_calls
            }
            for call in inline_calls:
                fn = call["function"]
                signature = stable_id([fn["name"], fn["arguments"]])
                if signature not in structured_signatures:
                    calls.append(call)
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": "\n\n".join(x for x in text_parts if x).strip(),
                "reasoning_content": "\n\n".join(x for x in reasoning if x).strip(),
                "train": True,
            }
            if calls:
                msg["tool_calls"] = calls
            if msg["content"] or msg["reasoning_content"] or calls:
                messages.append(msg)

    if not messages:
        raise ConversionError(f"no messages found in {p}")
    if secret_policy == "quarantine":
        secret_hits = find_secrets(messages)
        if secret_hits:
            raise ConversionError(f"secret-like content found in {p}: {secret_hits}")

    messages = normalize_trace(messages)
    violations = _forbidden_violations(messages)
    if violations:
        raise ConversionError("; ".join(violations))
    validate_trace(messages)
    tools = build_tool_registry(messages)
    content_hash = stable_id(messages)
    source_session_id = source_session_ids[0] if source_session_ids else None
    session_id = source_session_id or content_hash
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_file": str(p),
        "content_hash": content_hash,
        "source_session_id": source_session_id,
    }
    return ParsedSession(session_id, str(p), messages, tools, metadata)


def _schema_for_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, list):
        if not value:
            return {"type": "array"}
        item_schema = _schema_for_value(value[0])
        for item in value[1:]:
            item_schema = _merge_schema(item_schema, _schema_for_value(item))
        return {"type": "array", "items": item_schema}
    if isinstance(value, Mapping):
        return {
            "type": "object",
            "properties": {str(k): _schema_for_value(v) for k, v in value.items()},
            "additionalProperties": True,
        }
    if value is None:
        return {"type": "null"}
    return {"type": "string"}


def _merge_schema(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    if left == right:
        return deepcopy(left)
    if left.get("type") == right.get("type") == "object":
        props = deepcopy(left.get("properties", {}))
        for key, schema in right.get("properties", {}).items():
            props[key] = _merge_schema(props[key], schema) if key in props else deepcopy(schema)
        return {"type": "object", "properties": props, "additionalProperties": True}
    if left.get("type") == right.get("type") == "array":
        if "items" not in left:
            return deepcopy(right)
        if "items" not in right:
            return deepcopy(left)
        return {"type": "array", "items": _merge_schema(left["items"], right["items"])}
    variants: list[dict[str, Any]] = []
    for schema in (left, right):
        for candidate in schema.get("anyOf", [schema]):
            if candidate not in variants:
                variants.append(deepcopy(candidate))
    return {"anyOf": variants}


def build_tool_registry(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for msg in messages:
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", {})
            name = split_name(str(fn.get("name") or "unknown"))
            args = fn.get("arguments") if isinstance(fn.get("arguments"), dict) else {}
            entry = stats.setdefault(name, {"calls": 0, "presence": {}, "properties": {}})
            entry["calls"] += 1
            for key, value in args.items():
                key = str(key)
                entry["presence"][key] = entry["presence"].get(key, 0) + 1
                schema = _schema_for_value(value)
                if key in entry["properties"]:
                    entry["properties"][key] = _merge_schema(entry["properties"][key], schema)
                else:
                    entry["properties"][key] = schema
    result = []
    for name, entry in sorted(stats.items()):
        parameters: dict[str, Any] = {
            "type": "object",
            "properties": entry["properties"],
            "additionalProperties": True,
        }
        required = sorted(k for k, count in entry["presence"].items() if count == entry["calls"])
        if required:
            parameters["required"] = required
        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": KNOWN_TOOL_DESCRIPTIONS.get(name, f"Claude Code tool: {name}"),
                    "parameters": parameters,
                },
            }
        )
    return result


def tools_for_messages(
    messages: list[dict[str, Any]],
    registry: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    used = {
        split_name(str(call.get("function", {}).get("name") or "unknown"))
        for msg in messages
        for call in (msg.get("tool_calls") or [])
    }
    source = registry or build_tool_registry(messages)
    filtered = [tool for tool in source if tool.get("function", {}).get("name") in used]
    known = {tool.get("function", {}).get("name") for tool in filtered}
    if used - known:
        inferred = build_tool_registry(messages)
        filtered.extend(tool for tool in inferred if tool.get("function", {}).get("name") in used - known)
    return sorted(filtered, key=lambda t: str(t.get("function", {}).get("name")))


def split_episodes(session: ParsedSession) -> list[dict[str, Any]]:
    return [
        {
            "messages": session.messages,
            "tools": session.tools,
            "metadata": session.metadata | {"session_id": session.session_id},
        }
    ]


def _clone_with_train(messages: list[dict[str, Any]], train: bool | None = None) -> list[dict[str, Any]]:
    out = deepcopy(messages)
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
                nxt = messages[j]
                if nxt.get("role") != "tool":
                    raise ConversionError("non-tool message inside an unresolved tool transaction")
                atom.append(nxt)
                pending.discard(str(nxt.get("tool_call_id")))
                j += 1
            if pending:
                raise ConversionError(f"unresolved atomic transaction: {sorted(pending)}")
            i = j
        else:
            i += 1
        atoms.append(atom)
    return atoms


def _flatten(atoms: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [msg for atom in atoms for msg in atom]


def _has_trainable_assistant(messages: list[dict[str, Any]]) -> bool:
    return any(msg.get("role") == "assistant" and msg.get("train", True) for msg in messages)


def _context_atoms(current_atoms: list[list[dict[str, Any]]], overlap_messages: int) -> list[list[dict[str, Any]]]:
    if not current_atoms:
        return []
    leading_system_count = 0
    for atom in current_atoms:
        if atom[0].get("role") == "system":
            leading_system_count += 1
        else:
            break
    latest_user_index = next(
        (idx for idx in range(len(current_atoms) - 1, -1, -1) if current_atoms[idx][0].get("role") == "user"),
        None,
    )
    floor = latest_user_index if latest_user_index is not None else leading_system_count

    chosen_indices: set[int] = set(range(leading_system_count))
    if latest_user_index is not None:
        chosen_indices.add(latest_user_index)

    count = 0
    for idx in range(len(current_atoms) - 1, floor - 1, -1):
        if count >= overlap_messages:
            break
        chosen_indices.add(idx)
        count += len(current_atoms[idx])
    return [atom for idx, atom in enumerate(current_atoms) if idx in chosen_indices]


def chunk_episode(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    counter: ApproxTokenCounter | HFTokenCounter | None = None,
    max_tokens: int = 8192,
    overlap_messages: int = 6,
) -> list[dict[str, Any]]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_messages < 0:
        raise ValueError("overlap_messages must be non-negative")
    validate_trace(messages)
    counter = counter or ApproxTokenCounter()
    all_atoms = _atoms(messages)
    chunks: list[dict[str, Any]] = []
    current_atoms: list[list[dict[str, Any]]] = []

    def fits(atoms: list[list[dict[str, Any]]]) -> bool:
        candidate_messages = _flatten(atoms)
        candidate_tools = tools_for_messages(candidate_messages, tools)
        return counter.count_messages(candidate_messages, candidate_tools) <= max_tokens

    def emit(atoms: list[list[dict[str, Any]]]) -> None:
        candidate_messages = _clone_with_train(_flatten(atoms))
        if not _has_trainable_assistant(candidate_messages):
            return
        validate_trace(candidate_messages)
        candidate_tools = tools_for_messages(candidate_messages, tools)
        chunks.append({"messages": candidate_messages, "tools": candidate_tools})

    for atom in all_atoms:
        if fits(current_atoms + [atom]):
            current_atoms.append(atom)
            continue
        if not current_atoms:
            raise ConversionError("one atomic transaction exceeds max_tokens")
        emit(current_atoms)

        context = _context_atoms(current_atoms, overlap_messages)
        context_clones = [
            _clone_with_train(atom_messages, train=False) for atom_messages in context
        ]
        current_atoms = context_clones + [deepcopy(atom)]

        while not fits(current_atoms):
            removable = None
            for idx, candidate in enumerate(current_atoms[:-1]):
                role = candidate[0].get("role")
                if role not in {"system", "user"}:
                    removable = idx
                    break
            if removable is None:
                break
            current_atoms.pop(removable)
        if not fits(current_atoms):
            raise ConversionError("atomic transaction plus required system/user context exceeds max_tokens")

    if current_atoms:
        emit(current_atoms)
    if not chunks:
        raise ConversionError("chunking produced no trainable rows")
    return chunks


def to_arrow_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    """Serialize heterogeneous JSON objects so Hugging Face Arrow inference stays stable."""
    out = deepcopy(row)
    for msg in out.get("messages", []):
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", {})
            if isinstance(fn.get("arguments"), dict):
                fn["arguments"] = json.dumps(
                    fn["arguments"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
    if isinstance(out.get("tools"), list):
        out["tools"] = json.dumps(
            out["tools"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    return out


def from_arrow_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(row)
    tools = out.get("tools", [])
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError as exc:
            raise ConversionError(f"tools is not valid JSON: {exc}") from exc
    if not isinstance(tools, list):
        raise ConversionError("tools must decode to a list")
    out["tools"] = tools
    for msg in out.get("messages", []):
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", {})
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"])
                except json.JSONDecodeError as exc:
                    raise ConversionError(f"tool arguments is not valid JSON: {exc}") from exc
    return out
