# Resource Portal Restructure Report

## Date

2026-05-31

## Branch

`docs/restructure-resource-portal`

## Backup Directory

`docs/archive/readme-before-resource-portal-20260531/`

## Phase A Gate

Confirmed before restructuring: `origin/main` contains `fe4a26d Add Qwopus3.6 27B GSPO training tutorial`, and the public training script exists at:

```text
train_code/Qwopus3.6-27B-GSPO/qwopus3_6_27b_gspo_training.py
```

## Pre-Edit Inspection Checklist

| Item | Result |
|---|---|
| Existing top-level headings | Inspected in `README.md`, `docs/README_zh.md`, `docs/README_ko.md`, and `docs/README_ja.md`. |
| Existing language-switch links | Preserved and normalized across English, Chinese, Korean, and Japanese landing pages. |
| Existing badge styles | Preserved Unsloth, Google Colab, PyTorch, Hugging Face, LoRA / PEFT, and Beginner Friendly badges. |
| Existing Colab, Kaggle, Python, Hugging Face, PDF, dataset, and Skill links | Inspected and reused where still relevant. |
| Existing roadmap rows | Preserved compact model-family roadmap and kept the Qwen 3.6 row. |
| Existing MTP Skill links | Preserved links to `qwen-mtp-gguf/`, Pipeline Guide, and Agent Usage Guide. |
| Existing citation block | Preserved BibTeX format and corrected the repository URL to `R6410418/Jackrong-llm-finetuning-guide`. |
| Content kept visible on homepage | Training entry points, dataset catalog, data processing, MTP Skill, PDF guides, multilingual docs, roadmap, open-source commitment, citation. |
| Content moved or summarized | Long abstract, project philosophy, full MTP table, long dataset category prose, and message to builders. |
| Terminology issues | `GRPO` and `GSPO` are distinguished; `GPRO` typo is removed from active docs. |

## Modified Files

- `README.md`
- `docs/README_zh.md`
- `docs/README_ko.md`
- `docs/README_ja.md`

## Newly Created Files

- `docs/README.md`
- `docs/PROJECT_PHILOSOPHY.md`
- `docs/MAINTAINING_THE_KNOWLEDGE_BASE.md`
- `docs/RESOURCE_PORTAL_RESTRUCTURE_REPORT.md`
- `docs/archive/readme-before-resource-portal-20260531/README.md`
- `docs/archive/readme-before-resource-portal-20260531/README_en.md`
- `docs/archive/readme-before-resource-portal-20260531/README_zh.md`
- `docs/archive/readme-before-resource-portal-20260531/README_ko.md`
- `docs/archive/readme-before-resource-portal-20260531/README_ja.md`
- `train_code/README.md`
- `High-fidelity Dataset/README.md`
- `data_processing_code/README.md`
- `guidePDF/README.md`

## Navigation Architecture

The root README is now a concise resource portal with these primary entry points:

1. `🚀 Start Here`
2. `🗺️ Repository Map`
3. `🏋️ Training Recipes`
4. `✅ Supported Workflows`
5. `🛣️ Model Support Roadmap`
6. `⚙️ Qwen MTP GGUF Conversion Skill`
7. `📘 Guides and Reports`
8. `🧠 High-Fidelity Dataset Catalog`
9. `🤝 Open-Source Commitment`
10. `📚 Citation`

Detailed documentation now lives in directory-level index pages so the homepage can stay short as the repository grows.

## Future Growth Support

New training recipes, datasets, data-processing tools, PDF guides, standalone Skills, and languages now each have a clear catalog destination. `docs/MAINTAINING_THE_KNOWLEDGE_BASE.md` documents where future content should be added and which index pages must be updated.

## Validation Results

- Required-file checks passed for all requested active files:
  - `README.md`
  - `docs/README.md`
  - `docs/README_zh.md`
  - `docs/README_ko.md`
  - `docs/README_ja.md`
  - `docs/PROJECT_PHILOSOPHY.md`
  - `docs/MAINTAINING_THE_KNOWLEDGE_BASE.md`
  - `train_code/README.md`
  - `High-fidelity Dataset/README.md`
  - `data_processing_code/README.md`
  - `guidePDF/README.md`
- `git diff --check` passed with no whitespace errors.
- Large-file scan reported only pre-existing dataset JSONL files; no newly created large artifacts were added.

## Link-Check Results

- Local Markdown link checker inspected active created and modified Markdown files.
- Relative links checked: 227.
- External links skipped: 56. Network validation was not performed for external URLs.
- Result: all active repository-relative links resolved, including PDF links, Python-script links, notebook links, Qwen MTP GGUF documentation links, GSPO tutorial links, and language-switch links.
- The four archived landing page snapshots were excluded from active link resolution because they are intentionally retained verbatim for rollback traceability. Editing their old relative links would compromise snapshot fidelity.

## Privacy-Scan Results

- Scan scope: active landing pages, new index pages, new docs, report file, and README backup directory.
- Search patterns included non-public home directories, workspace-style absolute paths, Hugging Face and W&B credential variable names, credential-marker terms, email addresses, and sensitive-value terms.
- Result: no sensitive values, non-public paths, email addresses, raw logs, or machine-specific artifacts were found.
- One benign hit was expected in `docs/MAINTAINING_THE_KNOWLEDGE_BASE.md`, where design rule 7 documents what must never be published.

## Terminology Fixes

- `GRPO` and `GSPO` are distinguished in the homepage, localized landing pages, and training catalog.
- The Qwopus3.6 27B GSPO tutorial is never labeled as GRPO.
- The Llama3.2-R1 recipe remains labeled as GRPO.
- The Qwen 3.6 roadmap row is present in English, Chinese, Korean, and Japanese pages.
- The `GPRO` typo is absent from active user-facing docs except for intentional maintenance/report references describing the typo check.

## Backup Confirmation

The original four landing pages were backed up before editing:

```text
README.md          -> docs/archive/readme-before-resource-portal-20260531/README_en.md
docs/README_zh.md  -> docs/archive/readme-before-resource-portal-20260531/README_zh.md
docs/README_ko.md  -> docs/archive/readme-before-resource-portal-20260531/README_ko.md
docs/README_ja.md  -> docs/archive/readme-before-resource-portal-20260531/README_ja.md
```

At backup time, `git diff --no-index README.md docs/archive/readme-before-resource-portal-20260531/README_en.md` returned no differences.

## Manual Review Items

- External links were not network-validated in this run.
- Several dataset rows use local paths and are marked for manual external-link review because no verified Hugging Face URL was found in existing repository content.
- Pre-existing untracked local files `LOCAL_REVIEW/` and `LOCAL_SKILL_HUB_DISCOVERY.md` were left untouched and should not be included in this PR.
