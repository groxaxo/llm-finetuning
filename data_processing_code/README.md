# 🧪 Data Processing and Distillation Recipes

This catalog lists data-processing notebooks and scripts used to prepare, distill, clean, filter, or sample training data.

## 🚀 Quick Start

| Goal | Entry |
|---|---|
| Distill data with a teacher model API workflow | [`DeepSeek-v4-API-distill.ipynb`](DeepSeek-v4-API-distill.ipynb) |
| Choose a dataset after processing | [`../High-fidelity Dataset/README.md`](../High-fidelity%20Dataset/README.md) |
| Connect processed data to training | [`../train_code/README.md`](../train_code/README.md) |

## 🧪 Distillation

| Recipe | Type | File | Status | Notes |
|---|---|---|---|---|
| DeepSeek v4 API distillation | Notebook | [`DeepSeek-v4-API-distill.ipynb`](DeepSeek-v4-API-distill.ipynb) | ✅ Released | Teacher-model distillation workflow for generating or transforming training data. |

## 🧹 Cleaning and Normalization

Cleaning and normalization recipes are planned for future additions. When adding one, document input schema, output schema, deduplication behavior, and privacy checks.

## 🎯 Filtering and Sampling

Filtering and sampling recipes are planned for future additions. When adding one, document scoring criteria, sample-size controls, deterministic seeds, and validation strategy.

## 🧭 How to Add a New Processing Recipe

Each new recipe should include:

| Field | Expected content |
|---|---|
| Name | Clear task-oriented title. |
| Input schema | Required columns, JSON fields, or conversation format. |
| Output schema | Exact fields produced for downstream training. |
| Safety notes | Token handling, privacy filtering, redaction, and rate-limit behavior. |
| Reproducibility | Environment requirements, deterministic seeds, and sample command. |
| Catalog update | Add a row to this README before expanding the root homepage. |
