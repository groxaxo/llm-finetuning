# 🏋️ Training Recipe Catalog

This catalog lists the training notebooks and Python tutorials that are currently kept under `train_code/`. Legacy notebooks remain in place to preserve existing links.

## 🚀 Quick Start

| Goal | Start here |
|---|---|
| Run a browser-based SFT tutorial | Open a Colab or Kaggle notebook from the SFT table below |
| Run reinforcement learning | Start with the GRPO or GSPO entries below |
| Inspect the new Qwopus3.6 27B GSPO workflow | Open [`Qwopus3.6-27B-GSPO/README.md`](Qwopus3.6-27B-GSPO/README.md) |
| Prepare distilled training data first | See [`../data_processing_code/README.md`](../data_processing_code/README.md) |

## 📘 Supervised Fine-Tuning (SFT)

| Model | Size | Method | Environment | File | Status | Notes |
|---|---:|---|---|---|---|---|
| Qwopus3.5 | 27B | SFT | Google Colab | [`Qwopus3-5-27b-Colab.ipynb`](Qwopus3-5-27b-Colab.ipynb) | ✅ Released | End-to-end beginner Colab workflow with LoRA, export, and deployment notes. |
| Qwen3.5 Neo | 9B | SFT | Kaggle | [`Qwen3.5-9B-Neo-Kaggle.ipynb`](Qwen3.5-9B-Neo-Kaggle.ipynb) | ✅ Released | Kaggle training workflow for a compact Qwen-family fine-tune. |
| Qwopus3.5 | 35B-A3B | SFT | Kaggle | [`Qwopus-3.5-35B-A3B-Kaggle.ipynb`](Qwopus-3.5-35B-A3B-Kaggle.ipynb) | ✅ Released | Kaggle workflow for the larger Qwopus3.5 release. |

## 🧪 Reinforcement Learning (GRPO / GSPO)

| Model | Size | Method | Environment | File | Status | Notes |
|---|---:|---|---|---|---|---|
| Llama3.2-R1 | 3B | GRPO | Kaggle | [`Llama-3.2-3B-R1-Zero-GRPO.ipynb`](Llama-3.2-3B-R1-Zero-GRPO.ipynb) | ✅ Released | Browser-run reinforcement-learning notebook using GRPO terminology. |
| Qwopus3.6 | 27B | GSPO | Python script | [`Qwopus3.6-27B-GSPO/qwopus3_6_27b_gspo_training.py`](Qwopus3.6-27B-GSPO/qwopus3_6_27b_gspo_training.py) | ✅ Released | Publication-safe GSPO-style post-training tutorial with reward tests, LoRA export, 16-bit merge, GGUF notes, and `mmproj` handling. |

## 📦 Export and Deployment Notes

| Topic | Where to look |
|---|---|
| LoRA adapter save | Individual training notebooks or the Qwopus3.6 GSPO script |
| Merged 16-bit export | [`Qwopus3.6-27B-GSPO/README.md`](Qwopus3.6-27B-GSPO/README.md) and notebook export cells |
| GGUF export | [`Qwopus3.6-27B-GSPO/README.md`](Qwopus3.6-27B-GSPO/README.md) and the [Qwen MTP GGUF Skill](../qwen-mtp-gguf/) |
| Multimodal projector handling | [`Qwopus3.6-27B-GSPO/README.md#multimodal-projector-mmproj`](Qwopus3.6-27B-GSPO/README.md#multimodal-projector-mmproj) |

## 🧭 How to Add a New Training Recipe

Use this repeatable convention for new training folders:

```text
train_code/<model-family>-<size>-<method>/
```

For example:

```text
train_code/Qwopus3.6-27B-GSPO/
```

Each new recipe should include:

| Field | Expected content |
|---|---|
| `README.md` | Purpose, hardware expectations, setup, dataset schema, safe dry-run commands, training commands, export notes, and privacy notice. |
| Main script or notebook | A runnable `.py` script or `.ipynb` notebook with public-safe defaults. |
| Requirements | A requirements file or clear environment notes. |
| Example data | A tiny synthetic or public sample, never private training rows. |
| Validation notes | Smoke tests, reward checks, source parity notes, or privacy audit when relevant. |

Keep legacy notebooks directly under `train_code/` unless a migration is explicitly planned and all old links are preserved.
