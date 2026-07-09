import json
import tempfile
import unittest
from pathlib import Path

from claude_code_pipeline.converter import (
    ApproxTokenCounter,
    ConversionError,
    chunk_episode,
    find_secrets,
    parse_session_file,
)


class ClaudeCodePipelineTests(unittest.TestCase):
    def write_session(self, rows):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "session.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return tmp, path

    def test_tool_call_round_trip(self):
        tmp, path = self.write_session([
            {"role": "user", "content": "Fix the parser."},
            {"role": "assistant", "content": [
                {"type": "thinking", "text": "Need to inspect before editing."},
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "src/parser.py"}},
            ]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "def parse(): pass"}]},
            {"role": "assistant", "content": "I found the parser."},
        ])
        with tmp:
            session = parse_session_file(path)
        assistant = [m for m in session.messages if m["role"] == "assistant"][0]
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"], "Read")
        self.assertIsInstance(assistant["tool_calls"][0]["function"]["arguments"], dict)
        self.assertTrue(session.tools)

    def test_rejects_orphan_tool_result(self):
        tmp, path = self.write_session([
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "missing", "content": "oops"}]},
        ])
        with tmp, self.assertRaises(ConversionError):
            parse_session_file(path)

    def test_chunking_masks_overlap_assistant(self):
        messages = []
        for i in range(12):
            messages.append({"role": "user", "content": f"u{i}", "train": False})
            messages.append({"role": "assistant", "content": "x" * 200, "reasoning_content": "", "train": True})
        chunks = chunk_episode(messages, [], ApproxTokenCounter(), max_tokens=350, overlap_messages=2)
        self.assertGreater(len(chunks), 1)
        self.assertFalse(chunks[1]["messages"][0].get("train", True))

    def test_secret_detection(self):
        self.assertIn("generic_secret", find_secrets({"x": "api_key='abcdefghijklmnopqrstuvwxyz'"}))


if __name__ == "__main__":
    unittest.main()
