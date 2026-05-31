# 🧠 High-Fidelity Dataset Catalog

This catalog indexes the dataset folders currently stored under `High-fidelity Dataset/`. External Hugging Face links are included only when they are explicitly present in existing repository content; otherwise the local folder is linked and marked for manual external-link review.

## 🚀 Quick Start

Use [`../download_datasets.py`](../download_datasets.py) when you want to batch download the curated dataset suite for local training. Review individual dataset cards before mixing datasets, because schema, license, task type, and quality filters differ by source.

## Dataset Index

| Dataset | Category | Recommended use | Hugging Face link or local path | Notes |
|---|---|---|---|---|
| Chinese-DeepSeek-V3.2-Exp-chat-example | 💬 Instruction and Multi-turn Conversation | Chinese chat examples and conversation-style distillation review | [Local folder](Chinese-DeepSeek-V3.2-Exp-chat-example/) | External dataset link needs manual review. |
| Chinese-Qwen3-235B-Thinking-2507-Distill-100k | 🧠 Reasoning and Chain-of-Thought | Chinese reasoning and thinking-style SFT data | [HF dataset](https://huggingface.co/datasets/Jackrong/Chinese-Qwen3-235B-Thinking-2507-Distill-100k) | Existing README includes load and citation references. |
| Competitive-Programming-python-blend | 💻 Coding and Algorithms | Competitive-programming Python supervision and blended code reasoning | [Local folder](Competitive-Programming-python-blend/) | README verifies upstream source links; external dataset link for this blend needs manual review. |
| DeepSeek-V3.2-Exp-reasoning-example | 📐 Mathematics and STEM | Math reasoning comparison and distilled example inspection | [Local folder](DeepSeek-V3.2-Exp-reasoning-example/) | External dataset link needs manual review. |
| DeepSeek-v3.1-reasoner-Distilled-math-samples | 📐 Mathematics and STEM | Math distillation samples and reasoning evaluation | [HF dataset](https://huggingface.co/datasets/Jackrong/DeepSeek-v3.1-reasoner-Distilled-math-samples) | Existing README includes a citation URL. |
| GPT-OSS-120B-Distilled-Reasoning-math | 📐 Mathematics and STEM | Large reasoning-math SFT and evaluation data | [HF dataset](https://huggingface.co/datasets/Jackrong/GPT-OSS-120B-Distilled-Reasoning-math) | Existing README includes a citation URL. |
| GPT-OSS-20B-Distilled-Reasoning-Mini | 🧠 Reasoning and Chain-of-Thought | Smaller reasoning distillation experiments and validation | [HF dataset](https://huggingface.co/datasets/Jackrong/GPT-OSS-20B-Distilled-Reasoning-Mini) | Existing README includes a citation URL. |
| IELTS-writing-feedback-reasoning | 📊 Domain-Specific Data | IELTS writing feedback, scoring, and reasoning-style evaluation | [Local folder](IELTS-writing-feedback-reasoning/) | External dataset link needs manual review. |
| LogicMind-Chat-Reasoning-SFT-300K | 💬 Instruction and Multi-turn Conversation | Reasoning chat SFT and multi-message instruction data | [Local folder](LogicMind-Chat-Reasoning-SFT-300K/) | External dataset link needs manual review. |
| LogosForge-scored-sft-v1 | 🧠 Reasoning and Chain-of-Thought | Scored SFT rows with answer and chain-of-thought quality labels | [Local folder](LogosForge-scored-sft-v1/) | External dataset link needs manual review. |
| MultiReason-ChatAlpaca | 💬 Instruction and Multi-turn Conversation | Multi-turn reasoning chat and Alpaca-style SFT | [HF dataset](https://huggingface.co/datasets/Jackrong/MultiReason-ChatAlpaca) | Existing README includes load and citation references. |
| Natural-Reasoning-gpt-oss-120B-S1 | 🧠 Reasoning and Chain-of-Thought | Natural reasoning and long-form CoT supervision | [Local folder](Natural-Reasoning-gpt-oss-120B-S1/) | External dataset link needs manual review. |
| Qwen3-235B-A22B-Instruct-2507-Distilled-chat | 💬 Instruction and Multi-turn Conversation | Distilled Qwen3 instruct chat data | [HF dataset](https://huggingface.co/datasets/Jackrong/Qwen3-235B-A22B-Instruct-2507-Distilled-chat) | Existing README includes a citation URL. |
| Qwen3.5-reasoning-700x | 🧠 Reasoning and Chain-of-Thought | Compact Qwen3.5 reasoning validation and sample training data | [Local folder](Qwen3.5-reasoning-700x/) | External dataset link needs manual review. |
| ShareGPT-Qwen3-235B-A22B-Instuct-2507 | 💬 Instruction and Multi-turn Conversation | ShareGPT-style Qwen3 distilled conversations | [Local folder](ShareGPT-Qwen3-235B-A22B-Instuct-2507/) | Folder name preserves existing `Instuct` spelling. External link needs manual review. |
| ShareGPT-gpt-oss-120B-reasoning | 💬 Instruction and Multi-turn Conversation | ShareGPT-style reasoning conversations | [Local folder](ShareGPT-gpt-oss-120B-reasoning/) | External dataset link needs manual review. |
| financial-economics-reasoning | 📊 Domain-Specific Data | Finance and economics reasoning SFT/evaluation | [Local folder](financial-economics-reasoning/) | External dataset link needs manual review. |
| glm-4.7-Superior-Reasoning-stage1 | 🧠 Reasoning and Chain-of-Thought | GLM-4.7 superior-reasoning stage-one distillation | [Local folder](glm-4.7-Superior-Reasoning-stage1/) | README references upstream Alibaba Superior-Reasoning data. |
| glm-4.7-multiturn-CoT | 💬 Instruction and Multi-turn Conversation | Multi-turn GLM-4.7 chain-of-thought conversations | [Local folder](glm-4.7-multiturn-CoT/) | External dataset link needs manual review. |
| gpt-oss-120B-distilled-math-OpenAI-Harmony | 📐 Mathematics and STEM | Harmony-formatted GPT-OSS math reasoning | [HF dataset](https://huggingface.co/datasets/Jackrong/gpt-oss-120B-distilled-math-OpenAI-Harmony) | Existing README includes a citation URL. |
| gpt-oss-120B-distilled-reasoning | 🧠 Reasoning and Chain-of-Thought | GPT-OSS 120B distilled reasoning and math examples | [HF dataset](https://huggingface.co/datasets/Jackrong/gpt-oss-120B-distilled-reasoning) | Existing README includes a citation URL. |
| gpt-oss-120b-Reasoning-Instruction | 💬 Instruction and Multi-turn Conversation | Reasoning instruction following and alignment experiments | [Local folder](gpt-oss-120b-Reasoning-Instruction/) | External dataset link needs manual review. |
| gpt-oss-120b-reasoning-STEM-5K | 📐 Mathematics and STEM | STEM-focused reasoning validation and SFT subset | [Local folder](gpt-oss-120b-reasoning-STEM-5K/) | External dataset link needs manual review. |
| qwen3-coder-480b-distill-mini | 💻 Coding and Algorithms | Coding and algorithmic distillation samples | [Local folder](qwen3-coder-480b-distill-mini/) | External dataset link needs manual review. |

## 🧠 Reasoning and Chain-of-Thought

Use these datasets for step-by-step reasoning, answer verification, and CoT-style supervision. Review formatting conventions before mixing sources.

## 📐 Mathematics and STEM

Use these datasets for math, STEM, and structured problem-solving work. Keep evaluation data separate from training data when building benchmarks.

## 💻 Coding and Algorithms

Use these datasets for competitive programming, Python problem solving, and coding-instruction experiments. Check upstream license notes for blended datasets.

## 💬 Instruction and Multi-turn Conversation

Use these datasets for chat, ShareGPT-style turns, instruction following, and human-facing answer formatting.

## 📊 Domain-Specific Data

Use domain datasets when you need targeted capabilities such as IELTS writing feedback or finance/economics reasoning.

## 🧪 Small Samples and Validation Sets

Several folders contain compact examples or validation-oriented subsets. Treat small datasets as smoke-test inputs or evaluation aids unless their README describes a broader training use.

## 🧭 How to Add a New Dataset Entry

Add a folder under `High-fidelity Dataset/` and update this catalog with:

| Field | Expected content |
|---|---|
| Dataset | Folder name and public dataset name when available. |
| Category | One of reasoning, math/STEM, coding, conversation, domain-specific, or validation/sample. |
| Recommended use | A short description of the training or evaluation role. |
| Hugging Face link or local path | Use a verified external link from repository content, or link the local folder and mark external review needed. |
| Notes | Schema, license, source mix, spelling caveats, or manual-review items. |
