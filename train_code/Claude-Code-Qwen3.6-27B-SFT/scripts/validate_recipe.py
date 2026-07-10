#!/usr/bin/env python3
"""Static preflight for the checked-in Axolotl recipe configs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("install PyYAML to validate configs: pip install pyyaml") from exc
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(cfg, dict), f"{path}: YAML root must be a mapping")
    require(cfg.get("chat_template") == "tokenizer_default", f"{path}: tokenizer_default template required")
    require(cfg.get("chat_template_kwargs", {}).get("preserve_thinking") is True, f"{path}: preserve_thinking must be true")
    require(cfg.get("sample_packing") is False, f"{path}: sample_packing must remain false")
    require(cfg.get("train_on_inputs") is False, f"{path}: train_on_inputs must be false")
    require(cfg.get("sequence_len") in {4096, 8192}, f"{path}: unexpected sequence_len")
    require(cfg.get("load_in_4bit") is True and cfg.get("adapter") == "qlora", f"{path}: expected QLoRA")

    for section in ("datasets", "test_datasets"):
        entries = cfg.get(section)
        require(isinstance(entries, list) and entries, f"{path}: {section} must be non-empty")
        for entry in entries:
            require(entry.get("type") == "chat_template", f"{path}: {section} must use chat_template")
            require(entry.get("message_field_training") == "train", f"{path}: loss mask field must be train")
            require(entry.get("field_tools") == "tools", f"{path}: tools field missing")

    pattern = cfg.get("lora_target_modules")
    require(isinstance(pattern, str), f"{path}: lora_target_modules must be a regex string")
    regex = re.compile(pattern)
    expected = [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.63.linear_attn.in_proj_qkv",
        "model.language_model.layers.31.mlp.down_proj",
    ]
    forbidden = [
        "model.language_model.layers.64.self_attn.q_proj",
        "model.visual.blocks.0.attn.q_proj",
        "model.mtp.layers.0.self_attn.q_proj",
    ]
    for name in expected:
        require(regex.fullmatch(name) is not None, f"{path}: LoRA regex misses {name}")
    for name in forbidden:
        require(regex.fullmatch(name) is None, f"{path}: LoRA regex incorrectly targets {name}")

    fsdp = cfg.get("fsdp_config")
    if fsdp:
        require(fsdp.get("fsdp_version") == 2, f"{path}: FSDP2 required")
        require(fsdp.get("offload_params") is True, f"{path}: validated 3x3090 profile requires CPU offload")
        require(fsdp.get("cpu_ram_efficient_loading") is True, f"{path}: RAM-efficient loading required")
        require(fsdp.get("transformer_layer_cls_to_wrap") == "Qwen3_5DecoderLayer", f"{path}: wrong wrap class")

    return {
        "path": str(path),
        "sequence_len": cfg["sequence_len"],
        "fsdp2": bool(fsdp),
        "lora_primary_layers_only": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", type=Path)
    args = ap.parse_args()
    configs = args.configs or sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))
    if not configs:
        raise SystemExit("no configs found")
    results = [validate_config(path) for path in configs]
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
