import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREPARE = ROOT / "scripts" / "prepare_dataset.py"
VERIFY_MTP = ROOT / "scripts" / "verify_mtp.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


class PipelineScriptTests(unittest.TestCase):
    def test_prepare_uses_session_level_split_and_arrow_safe_storage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw"
            out = root / "out"
            raw.mkdir()
            first_content = None
            for idx in range(5):
                call_id = f"c{idx}"
                rows = [
                    {"type": "user", "sessionId": f"s{idx}", "message": {"role": "user", "content": f"task {idx}"}},
                    {"type": "assistant", "sessionId": f"s{idx}", "message": {"role": "assistant", "content": [{"type": "tool_use", "id": call_id, "name": "Bash", "input": {"command": "pwd"}}]}},
                    {"type": "user", "sessionId": f"s{idx}", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id, "content": "/tmp"}]}},
                    {"type": "assistant", "sessionId": f"s{idx}", "message": {"role": "assistant", "content": "done"}},
                ]
                text = "\n".join(json.dumps(row) for row in rows) + "\n"
                (raw / f"s{idx}.jsonl").write_text(text, encoding="utf-8")
                if idx == 0:
                    first_content = text
            assert first_content is not None
            (raw / "duplicate.jsonl").write_text(first_content, encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(PREPARE),
                    str(raw),
                    "--output-dir",
                    str(out),
                    "--validation-ratio",
                    "0.4",
                    "--max-seq-length",
                    "1000",
                ],
                check=True,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            train = [json.loads(line) for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines() if line]
            validation = [json.loads(line) for line in (out / "validation.jsonl").read_text(encoding="utf-8").splitlines() if line]
            train_hashes = {row["metadata"]["content_hash"] for row in train}
            validation_hashes = {row["metadata"]["content_hash"] for row in validation}
            self.assertFalse(train_hashes & validation_hashes)
            self.assertTrue(train and validation)
            sample = train[0]
            self.assertIsInstance(sample["tools"], str)
            call = next(message for message in sample["messages"] if message.get("tool_calls"))["tool_calls"][0]
            self.assertIsInstance(call["function"]["arguments"], str)
            rejects = (out / "rejected.jsonl").read_text(encoding="utf-8")
            self.assertIn("exact_duplicate_session", rejects)

    def test_mtp_verifier_detects_extra_decoder_layer_from_index(self):
        module = load_module(VERIFY_MTP, "verify_mtp_test")
        with tempfile.TemporaryDirectory() as td:
            checkpoint = Path(td)
            (checkpoint / "config.json").write_text(
                json.dumps({"text_config": {"num_hidden_layers": 64}}),
                encoding="utf-8",
            )
            index = {
                "weight_map": {
                    "model.language_model.layers.63.self_attn.q_proj.weight": "model-1.safetensors",
                    "model.language_model.layers.64.self_attn.q_proj.weight": "model-2.safetensors",
                }
            }
            (checkpoint / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
            keys, info = module.mtp_tensor_keys(checkpoint)
            self.assertEqual(info["normal_hidden_layers"], 64)
            self.assertIn("model.language_model.layers.64.self_attn.q_proj.weight", keys)
            self.assertNotIn("model.language_model.layers.63.self_attn.q_proj.weight", keys)


if __name__ == "__main__":
    unittest.main()
