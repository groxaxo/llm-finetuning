<div align="center">

# Jackrong-llm-finetuning-guide 🌌
**面向初学者与开发者的端到端开源大语言模型微调教育指南**

🌐 **选择语言:**  [🇬🇧 **English**](../README.md) ｜ 🇨🇳 **中文** ｜ [🇰🇷 한국어](./README_ko.md) ｜ [🇯🇵 日本語](./README_ja.md)

🤗 **HuggingFace:** [Jackrong](https://huggingface.co/Jackrong)

<br>

[![Unsloth](https://img.shields.io/badge/Powered%20by-Unsloth-8A2BE2?style=flat-square)](https://github.com/unslothai/unsloth)
[![Google Colab](https://img.shields.io/badge/Environment-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Model%20Hub-Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![LoRA PEFT](https://img.shields.io/badge/Technique-LoRA%20%2F%20PEFT-007EC6?style=flat-square)](#)
[![Beginner Friendly](https://img.shields.io/badge/Level-Beginner%20Friendly-brightgreen?style=flat-square)](#)

</div>

---

## 📑 摘要

这是一个专为初学者和开发者设计的大语言模型（LLM）微调教育仓库。本项目提供详细的理论解释、稳健的数据处理工作流、可复现的训练管道（涵盖监督微调 SFT 以及未来的强化学习 RL 实现），以及实用的部署策略。作者开源项目中的完整训练代码均在此仓库中公开。

<div align="center">
  <img width="100%" alt="项目概览" src="https://github.com/user-attachments/assets/67dff316-e77e-4909-a338-56caeb6583b4" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 20px 0;"/>
</div>

---

## 🏛️ 关于本项目

此仓库打造成一个**“从零到一”**的学习平台。无论您是完全零基础还是经验丰富的开发者，都能在这里找到可复现的端到端指南，带您走过大语言模型生命周期的每个阶段。只需注册一个 Google 账号并打开 Colab 作为起点，您就能学会如何高效地让模型适应您的特定领域需求。

### ✨ 核心特性

| 方面 | 描述 |
| :--- | :--- |
| 🛤️ **从 0 到 1 的学习路径** | 分步指南从最基础开始，只需要一个浏览器和免费的云环境。 |
| 🔄 **多样化的训练工作流** | 代码库涵盖了监督微调（SFT）以及强化学习（RL）和其他高级范式的基础设置。 |
| ⚡ **资源高效型工程** | 利用 Unsloth 等工具和 4-bit 量化技术，在单卡资源（如标准 Google Colab）限制下运行大规模训练。 |
| 📦 **端到端交付** | 从多源数据归一化到 LoRA 适配，合并导出 16-bit 模型，以及用于本地部署的 GGUF 量化。 |

## 🧭 从这里开始

| 目标 | 入口 |
| :--- | :--- |
| 使用 Codex Goal mode 自动化 Qwopus 27B RL 工作流 | [打开可编辑 RL Goal 模板](../codex-goals/qwopus27b-rl-training/) |
| 使用 Codex 转换并量化 Qwen MTP GGUF 版本 | [打开 MTP GGUF Goal 模板](../codex-goals/qwen-mtp-gguf-release/) |

## 🗂️ 仓库地图

| 区域 | 描述 | 链接 |
| :--- | :--- | :--- |
| 🤖 Codex Goal Templates | 用于 RL 训练、MTP GGUF 转换和仓库维护的可编辑 Codex 工作流 | [打开](../codex-goals/) |
| ⚙️ Qwen MTP GGUF | 规范转换脚本、预检、smoke test 与发布文档 | [打开](../qwen-mtp-gguf/) |
| 📓 训练笔记本 | 可在浏览器中运行的 SFT 与 RL 教学笔记本 | [打开](../train_code/) |
| 📚 高保真数据集 | 训练指南中使用的精选数据集目录 | [打开](../High-fidelity%20Dataset/) |
| 📖 PDF 指南 | 面向初学者的指南和技术报告 | [打开](../guidePDF/) |

---

## 💡 给建设者的一封信

> [!NOTE]
> *“对于初学者、爱好者和任何对人工智能感到好奇的人来说：这条路是可以学会的。”*

本文档的目的不仅仅是描述一次训练过程，更是在向大家传达一个更宏大的信息：微调、后训练（Post-training），乃至于中等规模的预训练，**绝非遥不可及的技术仪式**。它们是**工程实践**，是可以被学习、复现并逐步掌握的。借助开源模型、公开数据集、云计算平台以及日益成熟的训练工具链，你往往需要的仅仅是一个 Google 账号、一台普通的笔记本电脑，以及持久的好奇心。

作为一个同样从零起步的学习者，我深知许多新手的迷茫：环境配置的复杂性、晦涩的超参数、对算力资源的焦虑，常常成为入门的第一道门槛。这正是为何像 Unsloth 这样的优化工具链极其重要的原因：通过提高训练效率并优化资源利用，它们大幅降低了大型模型微调的实际门槛，将曾经需要昂贵硬件和专业经验的事情，变成了普通开发者也可以尝试与精通的日常。

**从这个意义上说，我们都有机会站在巨人的肩膀上，去理解模型、改造模型并赋予它们全新的能力。**

*没有人一开始就是专家。但每个专家都曾勇敢地迈出第一步。*

## 🚀 后续模型支持与路线图

在不远的将来，本仓库将持续扩展对最新、最前沿的开源模型家族的支持。即将推出的教程和代码库将全面涵盖**监督微调（SFT）**和**强化学习（RL - 特别是 GRPO）**管道。

以下是即将推出的模型家族的计划支持矩阵：

| 模型家族 | SFT 支持 | RL (GRPO) 支持 |
| :--- | :---: | :---: |
| **Qwen 3.5** | ✅ 已发布 | 已排期 |
| **Qwen 3** | 已排期 | 已排期 |
| **Llama3.2-R1 (3B)** | ✅ 已包含 | ✅ 已发布 |
| **Llama** (3.1 / 3.3) | 已排期 | 已排期 |

---

## 📓 交互式训练笔记本

以下是按模型架构分类的交互式 Kaggle 与 Colab 笔记本。您可以直接在浏览器中运行从数据准备到训练和推理的整个管道。所有笔记本都存储在仓库的 [`train_code`](../train_code/) 目录下。

### 🌟 核心笔记本

#### 🗄️ 数据集准备 (Dataset Preparation)
| 🤖 模型架构 | 🛠️ 训练管道 | 🚀 快速启动 |
| :--- | :--- | :---: |
| **Teacher model distill pipeline** | 数据处理 | [![Python Code](https://img.shields.io/badge/Code-Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://github.com/R6410418/Jackrong-llm-finetuning-guide/blob/main/data_processing_code/DeepSeek-v4-API-distill.ipynb) |

#### 🏋️ 模型训练 (Model Training)
| 🤖 模型架构 | 🛠️ 训练管道 | 🚀 快速启动 (一键运行) |
| :--- | :--- | :---: |
| **Qwopus3.5 (27B)** | SFT | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/R6410418/Jackrong-llm-finetuning-guide/blob/main/train_code/Qwopus3-5-27b-Colab.ipynb) |
| **Qwen3.5 (9B)** | SFT | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/R6410418/Jackrong-llm-finetuning-guide/blob/main/train_code/Qwen3.5-9B-Neo-Kaggle.ipynb) |
| **Qwopus3.5 (35B)** | SFT | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/R6410418/Jackrong-llm-finetuning-guide/blob/main/train_code/Qwopus-3.5-35B-A3B-Kaggle.ipynb) |
| **Llama3.2-R1 (3B)** | RL (GRPO) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/R6410418/Jackrong-llm-finetuning-guide/blob/main/train_code/Llama-3.2-3B-R1-Zero-GRPO.ipynb) |

---

## ⚙️ Qwen MTP GGUF 转换 Skill

所有 Qwopus 的 MTP 版本都是通过以下转换管线完成并测试的。

> [!NOTE]
> **什么是 MTP？** MTP（**Multi Token Prediction**）speculative decoding 可以让 Qwen3.6 这类模型在**准确率不变**的情况下获得约 **1.4-2.2x** 的生成速度提升。这使 Qwen3.6 27B 和 35B-A3B 相比原始 baseline 获得**超过 1.4x** 的速度提升，尤其适合本地模型使用场景。

> [!TIP]
> [`qwen-mtp-gguf`](../qwen-mtp-gguf/) 子项目把容易出错的手工发布流程整理成一条面向 agent 的分阶段 pipeline：检查机器资源、验证兼容性、注入 MTP tensors、使用 llama.cpp 转换、本地 smoke test、量化，并支持续传上传。

### ✨ 这个 Skill 负责什么

| 阶段 | 它做什么 | 为什么重要 |
| :--- | :--- | :--- |
| 🧭 **预检环境** | 检查 RAM、磁盘、llama.cpp 工具、Hugging Face 权限，以及目标/源模型 config 兼容性。 | 在下载大模型或长时间转换前提前拦截不可行任务。 |
| 🧩 **提取 MTP** | 只下载包含 MTP/nextn tensors 的源模型 shards，并注入目标模型索引。 | 避免为了少量 MTP heads 下载完整源模型。 |
| 🧪 **本地验证** | 对生成的 GGUF 文件运行 smoke test，优先使用模型 chat template，只把原始 ChatML 作为兜底 sanity check。 | 在公开发布前发现转换损坏或运行失败的问题。 |
| 📦 **发布流程** | 构建 F16/BF16 和常见 K/IQ 量化格式，并支持 stream 上传、重试、续传检查和清理。 | 让大模型 GGUF 发布在磁盘有限的机器上也更可控。 |

### 🚀 核心优势

- **为 agent 设计：** Codex、Claude Code、OpenCode、Qwen Code 以及能执行 shell 的 agent 都可以使用同一套分阶段工作流。
- **适合大模型磁盘压力：** `stream` 模式可以一次量化、上传并清理一个 GGUF 文件。
- **适合公开发布：** 示例使用占位符，文档避免暴露私有路径、tokens 和机器日志。

**从这里开始：** [README](../qwen-mtp-gguf/README.md) · [Pipeline Guide](../qwen-mtp-gguf/docs/Qwen-MTP-GGUF-Pipeline-Guide.md) · [Agent Usage Guide](../qwen-mtp-gguf/docs/Qwen-MTP-GGUF-Agent-Usage.md)

---

## 📖 全链路模型训练指南

如需了解 Qwopus 3.5 全链路微调过程的详细步骤（包括环境配置、数据准备以及优化技巧），请参阅我们最新的 PDF 指南：

> [!TIP]
> **🔗 [下载完整指南：Qwopus3-5-27b-Colab_complete_guide_to_llm_finetuning.pdf](https://github.com/R6410418/Jackrong-llm-finetuning-guide/blob/main/guidePDF/Qwopus3-5-27b-Colab_complete_guide_to_llm_finetuning.pdf)**
>
> **🔗 [下载技术报告：Qwopus-GLM-18B-Technical-Report.pdf](https://github.com/R6410418/Jackrong-llm-finetuning-guide/blob/main/guidePDF/Qwopus-GLM-18B-Technical-Report.pdf)**
> 一份简明的技术报告，涵盖 Qwopus-GLM-18B 的模型设计、训练思路与关键实现细节。

---

## 📚 高保真蒸馏数据集

高质量的数据是模型有效适配的核心驱动力。与我们的训练代码并行，本仓库提供了 **24 个经过精心整理的高保真数据集**的获取方式，这些数据特地被收集和蒸馏，以增强模型的推理、编程及对话能力。

这些数据集主要从如 *DeepSeek-V3.2*、*Qwen3-235B*、*GLM-4.7* 以及 *GPT-OSS-120B* 等当前最先进的旗舰模型中蒸馏而来，并采用了先进的思维链（CoT）格式。

**包含的关键数据类别：**
*   🧠 **推理与 CoT (思维链)：** 例如 `Jackrong/Qwen3.5-reasoning-700`、`Jackrong/Natural-Reasoning-gpt-oss-120B-S1` 以及 `Jackrong/glm-4.7-multiturn-CoT`，旨在改善逐步逻辑及演绎能力。
*   📐 **数学与 STEM：** 诸如 `DeepSeek-v3.1-reasoner-Distilled-math-samples` 等专业数据，以及针对特定领域知识的 `Jackrong/Qwen3-235B-A22B-Instruct-2507-Distilled-chat`。
*   💻 **代码与算法：** 数据集集合如 `Competitive-Programming-python-blend` 与 `qwen3-coder-480b-distill-mini` 用于增强算法竞赛表现及代码生成。
*   💬 **指令与多轮交互：** 如 `Jackrong/LogicMind-Chat-Reasoning-SFT-300K`、`Chinese-Qwen3-235B-Thinking-2507-Distill-100k` 以及 `ShareGPT-gpt-oss-120B-reasoning` 的资源，侧重于人类对齐验证、雅思写作反馈及稳健的连贯对话。

*所有数据集均已在 [HuggingFace Hub](https://huggingface.co/Jackrong) 免费开源。您也可以利用提供的 `download_datasets.py` 脚本，批量下载整个套件以供本地训练使用。*

---

## 🌍 开源承诺与社区影响

展望未来，我在 Hugging Face 上发布的每一个微调模型的完整训练源码，都将在此仓库中完全开源。我的初衷是确保任何人——无论其背景或资源如何——都能自由下载、执行并从这些脚本中学习，从而构建属于自己的 AI 能力。

我非常感谢社区的支持。近期我在 Hugging Face 分享的 Qwen3.5 微调系列模型达到了超过**百万次下载**——这无声地印证了核心知识开放的力量。我真诚地希望，通过公开这些完整的训练管道工作流，能激励更多的开发者开启属于他们自己的微调旅程。

---

## 📝 引用

如果您在学习或研究中觉得此仓库对您有帮助，请考虑引用：

```bibtex
@misc{jackrong-llm-finetuning,
  author = {Jackrong},
  title = {Jackrong-llm-finetuning-guide: An Educational LLM Fine-Tuning Pipeline},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Jackrong/Jackrong-llm-finetuning-guide}}
}
```
