import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from claude_code_pipeline.converter import (
    ApproxTokenCounter,
    ConversionError,
    SCHEMA_VERSION,
    build_tool_registry,
    chunk_episode,
    find_secrets,
    from_arrow_safe_row,
    parse_session_file,
    sanitize_text,
    to_arrow_safe_row,
    validate_trace,
)


class ClaudeCodePipelineTests(unittest.TestCase):
    def write_session(self, rows, suffix=".jsonl"):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / f"session{suffix}"
        if suffix == ".jsonl":
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
        else:
            path.write_text(json.dumps(rows), encoding="utf-8")
        return tmp, path

    def test_structured_tool_call_round_trip_and_name_resolution(self):
        tmp, path = self.write_session([
            {"role": "user", "content": "Fix the parser."},
            {"role": "assistant", "content": [
                {"type": "thinking", "text": "Inspect first."},
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "src/parser.py"}},
            ]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "def parse(): pass"}]},
            {"role": "assistant", "content": "I found it."},
        ])
        with tmp:
            session = parse_session_file(path)
        assistant = next(m for m in session.messages if m["role"] == "assistant")
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"], "Read")
        tool = next(m for m in session.messages if m["role"] == "tool")
        self.assertEqual(tool["name"], "Read")
        self.assertEqual(session.metadata["schema_version"], SCHEMA_VERSION)

    def test_top_level_tool_result_is_not_silently_dropped(self):
        tmp, path = self.write_session([
            {"role": "user", "content": "Inspect."},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "a.py"}}]},
            {"type": "tool_result", "tool_use_id": "t1", "content": "hello"},
            {"role": "assistant", "content": "Done."},
        ])
        with tmp:
            session = parse_session_file(path)
        self.assertEqual([m["role"] for m in session.messages], ["user", "assistant", "tool", "assistant"])

    def test_inline_tool_call_pattern_is_structured_and_deterministic(self):
        from claude_code_pipeline.converter import stable_id

        expected = f"call_{stable_id([[1, 0], 0, 'Bash', {'command': 'pytest -q'}])}"
        rows = [
            {"role": "user", "content": "Inspect."},
            {"role": "assistant", "content": '[TOOL_CALL: Bash] {"command":"pytest -q"}'},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": expected, "content": "ok"}]},
            {"role": "assistant", "content": "Tests pass."},
        ]
        tmp, path = self.write_session(rows)
        with tmp:
            session = parse_session_file(path)
        call = session.messages[1]["tool_calls"][0]
        self.assertEqual(call["id"], expected)
        self.assertEqual(call["function"]["arguments"], {"command": "pytest -q"})

    def test_consecutive_assistant_stream_fragments_are_repaired(self):
        tmp, path = self.write_session([
            {"role": "user", "content": "Task"},
            {"role": "assistant", "content": [{"type": "thinking", "text": "part one"}]},
            {"role": "assistant", "content": "part two"},
        ])
        with tmp:
            session = parse_session_file(path)
        assistants = [m for m in session.messages if m["role"] == "assistant"]
        self.assertEqual(len(assistants), 1)
        self.assertEqual(assistants[0]["reasoning_content"], "part one")
        self.assertEqual(assistants[0]["content"], "part two")

    def test_duplicate_call_id_rejected(self):
        messages = [
            {"role": "user", "content": "x", "train": False},
            {"role": "assistant", "content": "", "reasoning_content": "", "train": True, "tool_calls": [
                {"id": "same", "type": "function", "function": {"name": "Read", "arguments": {"file_path": "a"}}},
                {"id": "same", "type": "function", "function": {"name": "Read", "arguments": {"file_path": "b"}}},
            ]},
        ]
        with self.assertRaises(ConversionError):
            validate_trace(messages)

    def test_tool_name_mismatch_rejected(self):
        messages = [
            {"role": "user", "content": "x", "train": False},
            {"role": "assistant", "content": "", "reasoning_content": "", "train": True, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "Read", "arguments": {"file_path": "a"}}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "name": "Bash", "content": "x", "train": False},
        ]
        with self.assertRaises(ConversionError):
            validate_trace(messages)

    def test_sanitization_removes_client_artifacts_ansi_and_redacts(self):
        text = "\x1b[31mred\x1b[0m\n<local-command-caveat>noise</local-command-caveat>\n/model x\napi_key='abcdefghijklmnopqrstuvwxyz'"
        clean = sanitize_text(text, secret_policy="redact")
        self.assertNotIn("\x1b", clean)
        self.assertNotIn("local-command", clean)
        self.assertNotIn("/model", clean)
        self.assertIn("[REDACTED_GENERIC_SECRET]", clean)
        self.assertFalse(find_secrets(clean))

    def test_quarantine_policy_rejects_secret(self):
        tmp, path = self.write_session([
            {"role": "user", "content": "api_key='abcdefghijklmnopqrstuvwxyz'"},
            {"role": "assistant", "content": "No."},
        ])
        with tmp, self.assertRaises(ConversionError):
            parse_session_file(path, secret_policy="quarantine")

    def test_chunk_overlap_never_starts_with_orphan_tool_result(self):
        messages = [{"role": "user", "content": "root task", "train": False}]
        for i in range(7):
            messages.extend([
                {"role": "assistant", "content": "", "reasoning_content": f"inspect {i}", "train": True, "tool_calls": [
                    {"id": f"c{i}", "type": "function", "function": {"name": "Read", "arguments": {"file_path": f"f{i}.py"}}}
                ]},
                {"role": "tool", "tool_call_id": f"c{i}", "name": "Read", "content": "x" * 180, "train": False},
                {"role": "assistant", "content": f"observed {i}", "reasoning_content": "", "train": True},
            ])
        tools = build_tool_registry(messages)
        chunks = chunk_episode(messages, tools, ApproxTokenCounter(), max_tokens=420, overlap_messages=2)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            validate_trace(chunk["messages"])
            first_non_system = next(m for m in chunk["messages"] if m["role"] != "system")
            self.assertEqual(first_non_system["role"], "user")
        self.assertTrue(any(not m["train"] for m in chunks[1]["messages"] if m["role"] == "assistant"))

    def test_arrow_safe_round_trip(self):
        row = {
            "messages": [
                {"role": "user", "content": "x", "train": False},
                {"role": "assistant", "content": "", "reasoning_content": "", "train": True, "tool_calls": [
                    {"id": "c", "type": "function", "function": {"name": "Bash", "arguments": {"command": "pwd", "timeout": 5}}}
                ]},
                {"role": "tool", "tool_call_id": "c", "name": "Bash", "content": "/tmp", "train": False},
            ],
            "tools": [],
        }
        row["tools"] = build_tool_registry(row["messages"])
        safe = to_arrow_safe_row(row)
        self.assertIsInstance(safe["tools"], str)
        self.assertIsInstance(safe["messages"][1]["tool_calls"][0]["function"]["arguments"], str)
        self.assertEqual(from_arrow_safe_row(safe), row)

    def test_schema_merges_types_and_infers_required(self):
        messages = [
            {"role": "assistant", "tool_calls": [
                {"id": "1", "function": {"name": "Tool", "arguments": {"x": 1, "always": True}}},
                {"id": "2", "function": {"name": "Tool", "arguments": {"x": "one", "always": False, "optional": 3}}},
            ]}
        ]
        schema = build_tool_registry(messages)[0]["function"]["parameters"]
        self.assertEqual(schema["required"], ["always", "x"])
        self.assertIn("anyOf", schema["properties"]["x"])

    def test_trace_without_trainable_assistant_rejected(self):
        with self.assertRaises(ConversionError):
            validate_trace([{"role": "user", "content": "x", "train": False}])


if __name__ == "__main__":
    unittest.main()
