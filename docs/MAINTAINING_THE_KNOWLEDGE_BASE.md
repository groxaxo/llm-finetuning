# Maintaining the Jackrong LLM Knowledge Base

This repository is organized as a knowledge base. The root README should help visitors choose the right entry point quickly, while detailed explanations live beside the relevant tool, dataset, notebook, or guide.

## Where New Content Belongs

| New content type | Add it under | Update these index pages |
|---|---|---|
| New SFT / GRPO / GSPO training script | `train_code/` | `train_code/README.md`, root `README.md`, localized landing pages |
| New dataset | `High-fidelity Dataset/` | dataset catalog and root dataset summary when needed |
| New distillation or preprocessing tool | `data_processing_code/` | processing catalog |
| New PDF guide or technical report | `guidePDF/` | PDF catalog |
| New standalone Skill or deployment tool | dedicated top-level subproject | root repository map and tool README |
| New language | `docs/README_<language>.md` | root language selector and `docs/README.md` |

## Design Rules

1. Root README stays concise and navigational.
2. Detailed explanations live in directory-level README files.
3. Use relative links inside the repository.
4. Preserve old links where practical.
5. Add new content to one catalog before expanding the root homepage.
6. Use emojis moderately and consistently.
7. Never publish secrets, private paths, tokens, logs, or machine-specific artifacts.
8. Before pushing, run a broken-link and privacy scan.
9. Preserve backups before major documentation rewrites.
10. Keep model-family roadmap entries separate from individual training recipes.

## Pre-Push Checklist

| Check | Command or evidence |
|---|---|
| Required indexes exist | `test -f README.md docs/README.md train_code/README.md` and the other expected catalog files |
| Markdown whitespace is clean | `git diff --check` |
| Relative links resolve | Run the local Markdown link checker used by the restructuring report |
| Terminology is consistent | `rg -n "GPRO|RL \\(GPRO\\)|GRPO|GSPO|Qwen 3\\.6|Qwopus3\\.6" README.md docs train_code` |
| Privacy-sensitive strings are absent | Search staged docs for tokens, private paths, email addresses, raw logs, and server aliases |
| Large artifacts are intentional | `find . -type f -size +10M -not -path './.git/*' -print` |

## Catalog Update Order

1. Add the new artifact in its natural directory.
2. Update that directory's `README.md` catalog.
3. Update the root README only if the new artifact is a primary entry point.
4. Update localized landing pages when the root navigation changes.
5. Update `docs/README.md` if the new document affects documentation navigation.
