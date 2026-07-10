#!/usr/bin/env python3
"""Verify that a merged checkpoint preserves Qwen MTP/next-token tensors."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

MTP_NAME_RE = re.compile(r"(?:^|[._/])(?:mtp|nextn|next_n|multi_token)(?:[._/]|$)", re.I)
LAYER_RE = re.compile(r"(?:language_model|model)\.layers\.(\d+)\.")


def read_config(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint / "config.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def text_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def normal_layer_count(config: dict[str, Any]) -> int | None:
    value = text_config(config).get("num_hidden_layers")
    return int(value) if isinstance(value, int) else None


def mtp_config_indicators(config: dict[str, Any]) -> dict[str, Any]:
    indicators = {}
    for container_name, container in (("config", config), ("text_config", text_config(config))):
        for key, value in container.items():
            lowered = key.lower()
            if any(token in lowered for token in ("mtp", "nextn", "next_n", "multi_token")):
                indicators[f"{container_name}.{key}"] = value
    return indicators


def index_keys(checkpoint: Path) -> set[str]:
    keys: set[str] = set()
    for path in checkpoint.glob("*.safetensors.index.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map", {}) if isinstance(data, dict) else {}
        if isinstance(weight_map, dict):
            keys.update(str(key) for key in weight_map)
    return keys


def safetensor_keys(checkpoint: Path) -> set[str]:
    keys = index_keys(checkpoint)
    if keys:
        return keys
    files = sorted(checkpoint.glob("*.safetensors"))
    if not files:
        return set()
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise SystemExit("reading unsharded checkpoints requires: pip install safetensors") from exc
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys.update(handle.keys())
    return keys


def mtp_tensor_keys(checkpoint: Path) -> tuple[set[str], dict[str, Any]]:
    config = read_config(checkpoint)
    layers = normal_layer_count(config)
    all_keys = safetensor_keys(checkpoint)
    candidates: set[str] = set()
    for key in all_keys:
        if MTP_NAME_RE.search(key):
            candidates.add(key)
            continue
        match = LAYER_RE.search(key)
        if match and layers is not None and int(match.group(1)) >= layers:
            candidates.add(key)
    return candidates, {
        "all_tensor_keys": len(all_keys),
        "normal_hidden_layers": layers,
        "config_indicators": mtp_config_indicators(config),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path, help="Merged/full checkpoint to verify")
    ap.add_argument("--base", type=Path, help="Optional base checkpoint for exact MTP-key comparison")
    args = ap.parse_args()

    if not args.checkpoint.is_dir():
        raise SystemExit(f"missing checkpoint directory: {args.checkpoint}")
    if (args.checkpoint / "adapter_config.json").exists() and not list(args.checkpoint.glob("*.safetensors.index.json")):
        print("Adapter-only checkpoint detected. Verify the merged/full checkpoint instead.")
        return 2

    checkpoint_keys, info = mtp_tensor_keys(args.checkpoint)
    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        **info,
        "mtp_tensor_key_count": len(checkpoint_keys),
        "mtp_tensor_key_sample": sorted(checkpoint_keys)[:50],
    }

    if args.base:
        if not args.base.is_dir():
            raise SystemExit(f"missing base checkpoint directory: {args.base}")
        base_keys, base_info = mtp_tensor_keys(args.base)
        missing = sorted(base_keys - checkpoint_keys)
        result.update(
            {
                "base": str(args.base),
                "base_mtp_tensor_key_count": len(base_keys),
                "missing_base_mtp_keys": missing[:100],
                "missing_base_mtp_key_count": len(missing),
                "base_config_indicators": base_info["config_indicators"],
            }
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if not base_keys:
            print("Base checkpoint exposed no identifiable MTP keys; verification is inconclusive.")
            return 2
        return 1 if missing else 0

    print(json.dumps(result, indent=2, sort_keys=True))
    if checkpoint_keys:
        return 0
    print("No MTP/next-token tensors were identified. Re-run with --base for an exact comparison.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
