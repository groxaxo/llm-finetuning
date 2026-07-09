# Claude Code -> Qwen3.6 27B Tool-Trace SFT

This recipe prepares exported Claude Code sessions for Qwen3.6/ThinkingCap 27B supervised fine-tuning with Axolotl. It preserves user, assistant, reasoning, tool-call, and tool-result structure so the tokenizer's Qwen chat template can render native `<think>`, `<tool_call>`, and `<tool_response>` blocks.

## Validated hardware path

The practical local path is now validated on **3x RTX 3090 24 GB GPUs** using **Axolotl FSDP2 QLoRA with CPU parameter offload**.

Use:

```bash
axolotl preprocess configs/qwen36_27b_qlora_fsdp_3x24gb.yaml

CUDA_VISIBLE_DEVICES=0,1,2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_P2P_DISABLE=0 \
axolotl train configs/qwen36_27b_qlora_fsdp_3x24gb.yaml \
  --launcher torchrun -- --nproc_per_node 3
```

The FSDP2 config intentionally sets:

```yaml
fsdp_config:
  fsdp_version: 2
  offload_params: true
  cpu_ram_efficient_loading: true
  sharding_strategy: FULL_SHARD
  transformer_layer_cls_to_wrap: Qwen3_5DecoderLayer
```

This is slower than pure-GPU FSDP, but it makes 27B QLoRA feasible on 24 GB cards. Keep host RAM clear; 125 GB-class RAM is a sensible target.

## Why the format matters

Do **not** pre-render Qwen control tokens into the dataset. The dataset keeps structured messages:

- assistant reasoning in `reasoning_content`
- assistant tool calls in `tool_calls[].function.arguments` as JSON mappings
- tool outputs as `role: tool`
- tool schemas in the row-level `tools` field
- loss masking through per-message `train`

The Axolotl configs use:

```yaml
chat_template: tokenizer_default
chat_template_kwargs:
  preserve_thinking: true
```

`preserve_thinking: true` is required so historical reasoning traces are not silently dropped by the Qwen chat template.

## Pipeline

Run from this folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Export Claude Code sessions:

```bash
python scripts/export_claude_sessions.py \
  --claude-home ~/.claude \
  --output-dir data/raw/claude-code
```

Prepare and split the dataset:

```bash
python scripts/prepare_dataset.py \
  data/raw/claude-code \
  --output-dir data/processed \
  --model bottlecapai/ThinkingCap-Qwen3.6-27B \
  --max-seq-length 8192 \
  --validation-ratio 0.02 \
  --seed 3407 \
  --secret-policy quarantine
```

Validate before training:

```bash
python scripts/validate_dataset.py \
  --train data/processed/train.jsonl \
  --validation data/processed/validation.jsonl \
  --model bottlecapai/ThinkingCap-Qwen3.6-27B \
  --max-seq-length 8192 \
  --render-count 5
```

## Important choices

- `sample_packing: false`: avoids cross-session leakage and keeps tool transactions causal.
- LoRA targets include Qwen3.6 text-backbone attention, linear-attention, and MLP projections.
- Vision and MTP modules are not targeted by LoRA.
- Tool call/result pairs are validated before training.
- Secret-like content is quarantined by default.

## Fallbacks

If your local CUDA/PyTorch/kernel stack still OOMs at 8192, regenerate the processed dataset with `--max-seq-length 4096` and change `sequence_len` plus `dataset_prepared_path` in the Axolotl YAML. Do not let Axolotl silently truncate 8192-token rows.
