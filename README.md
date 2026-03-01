# 基于SFT的新闻结构化摘要助手（QLoRA微调Qwen3-4B）

基于 **Qwen3 + LLaMA-Factory** 的参数高效微调（PEFT）流水线，面向消费级 GPU 的结构化新闻摘要生成。

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.8+](https://img.shields.io/badge/CUDA-12.8%2B-green.svg)](https://developer.nvidia.com/cuda-downloads)

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 项目结构](#2-项目结构)
- [3. 快速开始](#3-快速开始)
- [4. 端到端流程](#4-端到端流程)
- [5. 评测方法](#5-评测方法)
- [6. 实验结果](#6-实验结果)
- [7. 技术细节](#7-技术细节)
- [8. 常见问题](#8-常见问题)
- [9. 参考文献](#9-参考文献)

---

## 1. 项目概述

本项目实现了一个**端到端的新闻结构化摘要微调系统**，通过 QLoRA（Quantized Low-Rank Adaptation）技术在消费级 GPU 上高效训练大语言模型，使其能够将新闻文本转化为固定 6 字段格式的结构化摘要，适用于客户终端、边缘设备的信息流展示。

**核心技术栈**：
- **基座模型**：Qwen3-4B / Qwen3-8B
- **微调框架**：LLaMA-Factory（支持 QLoRA、4-bit NF4 量化）
- **数据来源**：XL-Sum 多语言新闻数据集
- **标注方案**：DeepSeek API

**输出格式示例**：
```
【一句话摘要】事件核心描述
【核心要点】1. ... 2. ... 3. ...
【事件类别】科技/财经/社会/国际/...
【主要主体】相关组织或个人
【时间信息】事件发生时间
【潜在影响】对行业、市场、社会的影响分析
```

---

### 1.1 核心特性

- **固定字段结构化输出**：固定格式输出，零后处理直接适配 UI 渲染
- **QLoRA 参数高效微调**：4-bit 量化 + LoRA 低秩适配，消费级显卡 16GB 显存可训练
- **异步数据标注流水线**：DeepSeek API 异步并发打标（5 并发 ~50 条/分钟），成本 <￥15/1000 条
- **多维度评测体系**：ROUGE-L、格式合规率、推理延迟三维评测
- **基座/微调对比工具**：内置并排对比模式，量化微调收益

---

### 1.2 系统要求

| 组件 | 规格要求 |
|------|---------|
| **GPU** | NVIDIA RTX 3090 / 4090 / 5060 Ti 及以上（≥16GB 显存） |
| **CUDA** | 12.8+ |
| **Python** | 3.11 |
| **磁盘空间** | ~30GB（模型 + 数据集 + checkpoints） |

---

## 2. 项目结构

```
Qwen3-QLoRA-News/
├── README.md                              # 项目主文档（含实验结果与使用指南）
├── requirements.txt                       # Python 依赖清单（精确版本号）
├── .gitignore                             # 排除大文件：outputs/、data/raw/ 等
│
└── projects/edge_news_summarizer/
    │
    ├── configs/                           # LLaMA-Factory YAML 配置文件
    │   ├── train_qwen3_4b_qlora_news.yaml # Qwen3-4B QLoRA 训练配置（已验证）
    │   ├── train_qwen3_8b_qlora_news.yaml # Qwen3-8B QLoRA 训练配置（备用）
    │   ├── infer_news.yaml                # 微调模型批量推理配置（batch=4）
    │   └── infer_news_base.yaml           # 基座模型批量推理配置（max_new_tokens=2048）
    │
    ├── data/                              # 数据流水线（大文件已 .gitignore）
    │   ├── raw/                           # 原始 XL-Sum 采集数据（未处理）
    │   ├── labeled/                       # DeepSeek API 标注中间产物
    │   ├── cleaned/                       # 最终可用数据集
    │   │   ├── train.json                 # 训练集（3,843 条）
    │   │   ├── val.json                   # 验证集（480 条）
    │   │   └── test.json                  # 测试集（481 条）
    │   ├── prompts/
    │   │   └── label_prompt_news_structured.txt  # DeepSeek 标注 prompt 模板
    │   └── reports/                       # 数据质量检查报告（JSON）
    │
    ├── scripts/                           # 自动化流水线脚本
    │   ├── 01_collect_news.py             # 从 XL-Sum 采集原始新闻数据
    │   ├── 02_generate_labels_api.py      # 调用 DeepSeek API 批量生成结构化标签
    │   ├── 03_validate_and_clean.py       # 格式校验、去重、异常过滤
    │   ├── 04_split_dataset.py            # 按 8:1:1 切分 Train/Val/Test
    │   ├── 05_register_dataset_info.py    # 向 LLaMA-Factory dataset_info.json 注册数据集
    │   ├── 06_eval_rouge_and_format.py    # ROUGE 评测 + 格式合规率统计（支持 --strip_think）
    │   ├── 07_benchmark_latency.py        # 单条推理延迟基准测试
    │   ├── 08_demo_cli.py                 # 交互式命令行演示（基座/微调对比模式）
    │   ├── infer_base_hf.py               # 基座模型 HF 原生推理（断点续推，每条即时保存）
    │   └── infer_base_vllm.py             # 基座模型 vLLM 推理（Linux/WSL2 环境）
    │
    ├── outputs/                           # 训练与评测产物（已 .gitignore）
    │   ├── checkpoints/
    │   │   └── qwen3-4b-qlora-news/       # LoRA adapter 权重（合并前）
    │   ├── merged/                        # 合并后完整模型权重（可选）
    │   ├── eval/                          # 微调模型评测结果
    │   │   ├── generated_predictions.jsonl
    │   │   ├── rouge_report.json
    │   │   ├── format_report.json
    │   │   └── bad_cases.jsonl            # 格式不合规样本（共 2 条）
    │   ├── eval_base/                     # 基座模型评测结果（对比用）
    │   └── logs/                          # 训练日志（trainer_log.jsonl）
    │
    └── docs/                              # 开发文档
        ├── dev_plan.md                    # 项目开发计划与阶段拆解
        ├── labeling_guideline.md          # 标注规范：6 字段格式定义与示例
        └── troubleshooting.md             # 常见问题排查记录
```

---

## 3. 快速开始

### 3.1 环境配置

```bash
# 创建 conda 环境
conda create -n my_sft python=3.11 -y
conda activate my_sft

# 安装 PyTorch（以CUDA 12.8为例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 验证 GPU 可用性
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

### 3.2 安装 LLaMA-Factory

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e ".[torch,metrics]"
```

### 3.3 安装项目依赖

```bash
pip install -U datasets pandas tqdm pydantic python-dotenv openai \
               rouge-score jieba bitsandbytes pyyaml jsonlines
```

### 3.4 配置 API 凭据（以deepseek为例）

```bash
cp projects/edge_news_summarizer/.env.example projects/edge_news_summarizer/.env
# 编辑 .env 填入：
# OPENAI_API_KEY=sk-xxxx          # DeepSeek API Key
# OPENAI_API_BASE=https://api.deepseek.com  
# OPENAI_MODEL=deepseek-chat  
```

### 3.5 下载基座模型

```python
from huggingface_hub import snapshot_download

# Qwen3-4B
snapshot_download("Qwen/Qwen3-4B", local_dir=r"D:\LLM\models\Qwen3-4B")

# Qwen3-8B
snapshot_download("Qwen/Qwen3-8B", local_dir=r"D:\LLM\models\Qwen3-8B")
```

---

## 4. 端到端流程

### 4.1 数据构建流程

构建高质量训练数据分为 5 个步骤，每个步骤对应一个独立脚本。所有脚本可在项目任意目录执行，路径已自动处理。

#### 4.1.1 Step 1: 原始数据采集

从 XL-Sum 数据集（BBC 多语言新闻）采集中英文混合数据，过滤涉政内容。

```bash
python projects/edge_news_summarizer/scripts/01_collect_news.py \
  --source xlsum --lang mixed --max_samples 6000
```

**输出**：`data/raw/news_raw.jsonl` (过滤后约4,800 条记录)

#### 4.1.2 Step 2: 结构化标注

使用 DeepSeek API 异步生成六字段结构化标签。

```bash
python projects/edge_news_summarizer/scripts/02_generate_labels_api.py \
  --max_samples 0 --concurrency 5
```

**输出**：`data/labeled/news_labeled_v1.jsonl` 

#### 4.1.3 Step 3: 校验与清洗

校验字段完整性、去重、检查格式合规性。

```bash
python projects/edge_news_summarizer/scripts/03_validate_and_clean.py
```

**输出**：`data/cleaned/cleaned_all.jsonl`, `data/reports/data_quality_report.md`

#### 4.1.4 Step 4: 数据集划分

按 8:1:1 划分训练集/验证集/测试集。

```bash
python projects/edge_news_summarizer/scripts/04_split_dataset.py \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

**输出**：`train.json` (3,843) / `val.json` (480) / `test.json` (481) / `test_manual_eval.json` (100)

#### 4.1.5 Step 5: 注册到 LLaMA-Factory

将数据集注册到 LLaMA-Factory 的配置文件。

```bash
# 在 LlamaFactory 根目录执行
python projects/edge_news_summarizer/scripts/05_register_dataset_info.py
```

**输出**：数据集 `news_structured_summary` 已注册至 `LlamaFactory/data/dataset_info.json`

**数据流示意**：
```
XL-Sum (BBC) → 筛选+内容过滤 → DeepSeek API → 校验清洗 → 数据划分 → Alpaca 格式
  ~200k           ~4,800      ~4,800        ~4,804  3,843/480/481   train.json
```

### 4.2 模型训练

本项目采用 **QLoRA（Quantized Low-Rank Adaptation）** 微调方法，在 16GB 显存上高效训练 4B/8B 模型。技术细节见 [Technical Details](#技术细节) 章节。

#### 4.2.1 训练命令

在 **LlamaFactory 根目录**执行：

```bash
conda activate my_sft

# Qwen3-4B（推荐先验证，训练约 2.5 小时）
llamafactory-cli train projects/edge_news_summarizer/configs/train_qwen3_4b_qlora_news.yaml

# Qwen3-8B（生产级质量，需调整配置）
llamafactory-cli train projects/edge_news_summarizer/configs/train_qwen3_8b_qlora_news.yaml
```

**训练信息**（4B 模型，RTX 5060 Ti 16GB）：
- **总步数**：651 步（3 epochs × 217 steps/epoch）
- **训练时长**：~2.5 小时（~14 秒/步）
- **显存占用**：~6.6 GB（batch=1 + 梯度累积）
- **Checkpoints**：每 50 步保存一次，保留最近 5 个

#### 4.2.2 合并 LoRA 权重（可选）

训练完成后，可将 LoRA adapter 合并回基座模型用于独立部署：

```bash
llamafactory-cli export \
  --model_name_or_path D:/LLM/models/Qwen3-4B \
  --adapter_name_or_path projects/edge_news_summarizer/outputs/checkpoints/qwen3-4b-qlora-news \
  --template qwen3_nothink \
  --finetuning_type lora \
  --export_dir projects/edge_news_summarizer/outputs/merged/qwen3-4b-news
```



### 4.3 推理与应用

#### 4.3.1 交互式 CLI

```bash
# 仅微调模型
python projects/edge_news_summarizer/scripts/08_demo_cli.py \
  --model_path D:/LLM/models/Qwen3-4B \
  --adapter_path projects/edge_news_summarizer/outputs/checkpoints/qwen3-4b-qlora-news

# 对比模式（基座 vs 微调，推荐）
python projects/edge_news_summarizer/scripts/08_demo_cli.py \
  --model_path D:/LLM/models/Qwen3-4B \
  --adapter_path projects/edge_news_summarizer/outputs/checkpoints/qwen3-4b-qlora-news \
  --compare
```

#### 4.3.2 批量对比推理

从测试集提取样本，生成基座/微调并排对比结果：

```bash
# 交互式对比
python projects/edge_news_summarizer/scripts/08_demo_cli.py \
  --model_path D:/LLM/models/Qwen3-4B \
  --adapter_path projects/edge_news_summarizer/outputs/checkpoints/qwen3-4b-qlora-news \
  --compare

# 批量对比（从测试集取 50 条，保存对比结果）
python projects/edge_news_summarizer/scripts/08_demo_cli.py \
  --model_path D:/LLM/models/Qwen3-4B \
  --adapter_path projects/edge_news_summarizer/outputs/checkpoints/qwen3-4b-qlora-news \
  --compare \
  --input_file projects/edge_news_summarizer/data/cleaned/test.json \
  --output_file projects/edge_news_summarizer/outputs/eval/compare_outputs.jsonl \
  --num_samples 50
```

两个模型均使用**相同的系统提示词**，区别仅在于有无 LoRA adapter，因此对比结果可以纯粹反映微调带来的格式约束与摘要质量提升。

对比输出格式示例：
```
======================================================================
新闻标题：苹果发布 iPhone 16 系列
======================================================================
【基座模型（Base + 系统提示词）】  耗时: 2.341s
----------------------------------------------------------------------
（无固定格式，通常为自由文本）这是一款非常好的手机...
======================================================================
【微调模型（Fine-tuned + 系统提示词）】  耗时: 2.158s
----------------------------------------------------------------------
【一句话摘要】苹果发布 iPhone 16，全系搭载 A18 芯片并支持 Apple Intelligence。
【核心要点】1. ...  2. ...  3. ...
【事件类别】科技
...
======================================================================
```

对比结果 JSONL 中每条记录包含 `base_prediction` 和 `ft_prediction` 两个字段，可直接传入 `06_eval_rouge_and_format.py` 分别评测两组结果。

---

## 5. 评测方法

本项目主要从**文本生成质量**和**推理性能**两个维度进行量化评估。

### 5.1 文本质量评测 (ROUGE + 格式合规)

执行以下命令生成测试集预测结果并计算指标：

```bash
# 生成预测结果
llamafactory-cli eval projects/edge_news_summarizer/configs/infer_news.yaml

# 计算 ROUGE 分数与格式合规率
python projects/edge_news_summarizer/scripts/06_eval_rouge_and_format.py \
  --test projects/edge_news_summarizer/data/cleaned/test.json \
  --predictions projects/edge_news_summarizer/outputs/eval/generated_predictions.jsonl
```

**评估指标说明**：

| 维度 | 指标 | 目标值 | 说明 |
|------|------|--------|------|
| **内容重叠** | ROUGE-1 | > 0.40 | 单词级重叠 (jieba 分词) |
| | ROUGE-2 | > 0.20 | 双词组重叠 (Bigram) |
| | ROUGE-L | > 0.30 | 最长公共子序列 |
| **格式合规** | 字段完整率 | 100% | 6 个预定义字段全部存在 |
| | 类别合规率 | 100% | 事件类别在白名单内 |
| | 要点格式率 | > 95% | 核心要点包含编号列表 |

### 5.2 推理延迟基准测试

在目标部署硬件上评估模型推理速度：

```bash
python projects/edge_news_summarizer/scripts/07_benchmark_latency.py \
  --model_path D:/LLM/models/Qwen3-4B \
  --adapter_path projects/edge_news_summarizer/outputs/checkpoints/qwen3-4b-qlora-news \
  --num_samples 20
```

---

## 6. 实验结果

本节记录 Qwen3-4B QLoRA SFT 的完整量化评测结果，包含**自动评测指标**、**格式合规性**及**基座/微调对比**，遵循业界主流 SFT 评测规范（参考 RLHF Survey, InstructGPT, LLaMA2 等工作的评测方法）。

### 6.1 实验配置

| 项目 | 配置 |
|------|------|
| **基座模型** | Qwen3-4B |
| **微调方法** | QLoRA（4-bit NF4，LoRA rank=8，alpha=16）|
| **训练数据** | 3,843 条（DeepSeek API 标注，来源 XL-Sum）|
| **验证集** | 480 条 |
| **测试集** | 481 条 |
| **训练步数** | 651 steps（3 epochs）|
| **训练时长** | 3h 06m 45s（RTX 5060 Ti 16GB）|
| **评测框架** | LLaMA-Factory predict + rouge-score + jieba 分词 |

### 6.2 训练过程指标

| 指标 | 数值 |
|------|------|
| 最终 Train Loss | 0.948 |
| 最终 Eval Loss | 0.9494 |
| 困惑度（Perplexity）| exp(0.9494) ≈ **2.58** |
| 最优 Checkpoint | Step 400（eval_loss = 0.9299）|

> Eval Loss 在 Step 400 后轻微回升（0.9299 → 0.9494），训练-验证 gap 从 0.038 增至 0.145，Epoch 3 存在轻微过拟合信号，对最终指标影响可忽略。

### 6.3 评测结果（测试集 481 条）

#### 6.3.1 文本质量（ROUGE，jieba 分词）

> **实验条件说明**：基座模型（Prompt Engineering）使用严格格式 prompt，显式给出 `【标签】` 模板、13 类枚举、字数约束并禁用 Markdown，`thinking=False`；微调模型使用原始一句话 instruction，无格式 prompt 辅助。两者均在同一测试集（481 条）上评测，结果反映 **Prompt Engineering 上限 vs SFT** 的对比。

| 指标 | 基座模型（+ Prompt Engineering）| 微调模型（SFT）| 绝对提升（SFT−PE）| 相对提升 |
|------|------------------------------|--------------|-------------------|--------|
| **ROUGE-1** | 0.687 | **0.769** | +0.082 | +11.9% |
| **ROUGE-2** | 0.423 | **0.532** | +0.109 | +25.8% |
| **ROUGE-L** | 0.652 | **0.737** | +0.085 | +13.0% |

#### 6.3.2 格式合规性

| 指标 | 基座模型（+ Prompt Engineering）| 微调模型（SFT）| 说明 |
|------|-------------------------------|--------------|------|
| **必需字段完整率** | 100.00% | **100%** | 严格 prompt 模板逐字列出 6 个标签 |
| **类别合规率** | 98.96% | **100.00%** | SFT 精确内化枚举；基座偶发自造类别（如"法律""治安""音乐"）|
| **要点格式率**（≥3条编号）| 100.00% | **100%** | 两者均高度合规 |
| **时间信息完整率** | 100.00% | **100.00%** | — |
| **平均要点数** | 3.01 条 | **3.53 条** | SFT 输出更丰富的要点内容 |
| **格式错误样本** | 5 / 481 | **2 / 481** | 基座 5 条均为类别超出枚举，非结构字段缺失 |

#### 6.3.3 推理效率（批量推理，batch=4，BF16，RTX 5060 Ti 16GB）

| 指标 | 基座模型（+ Prompt Engineering，thinking=False）| 微调模型（SFT）| 说明 |
|------|-------------------------------|--------------|------|
| **总耗时**（481 条）| ~33 min | **~17 min** | 关闭 thinking 后耗时大幅压缩 |
| **平均速度** | ~4.1 s/条 | **~2.1 s/条** | — |
| **吞吐量** | ~14.6 条/min | **~28.3 条/min** | — |
| **平均输出长度** | ~287 字符 | **~351 字符** | SFT 要点更丰富（3.53 vs 3.01 条）|

> 基座模型使用 `enable_thinking=False`（模板层禁用 `<think>` token）后，单条平均生成 token 从 ~761 降至 ~180，推理速度相比开启 thinking 的初始版本（~7.1 s/条）提升约 1.7×。SFT 模型因训练标签不含 `<think>` 内容，即使使用默认配置也不会自发触发思考链，推理效率更高。

### 6.4 关键观察

1. **SFT 的核心价值是低成本指令遵循**：基座模型在精心设计的严格 prompt 下，格式通过率可达 100%，ROUGE-1 达 0.687。但微调模型仅凭一句话 instruction（"你是专业新闻编辑助手…"）就能稳定输出结构化结果（ROUGE-1 0.769，格式 99.58%）。SFT 的价值在于将复杂格式规范内化为模型权重，**无需 prompt 工程即可可靠工作**。

2. **类别枚举精度是 SFT 独有优势**：基座模型即使在 prompt 中列出 13 个合法类别，仍有约 1% 的样本自造超出枚举的类别（如"法律""治安""音乐"），类别合规率 98.96%；微调模型类别合规率 **100%**，说明 SFT 将枚举精准内化为权重约束，严格程度高于 prompt 软约束。

3. **ROUGE 差距缩小但 SFT 仍领先**：使用严格格式 prompt 后，基座模型 ROUGE-1 从原始 prompt 下的 ~0.48（130 条预览）提升至 0.687，说明格式对齐本身可显著提升 ROUGE。微调模型在同等格式标准下进一步领先（0.769，+11.9%）、ROUGE-2 更高（+25.8%），反映出 SFT 内容质量的真实提升，而非仅格式红利。

4. **思考链抑制大幅提升推理效率**：基座模型使用 `enable_thinking=False`（模板层禁用 `<think>` token）后，单条平均生成 token 从 ~761 降至 ~180，推理速度提升约 1.7×（7.1 s/条→4.1 s/条）；SFT 模型因训练标签不含 `<think>` 内容，无需额外配置即可保持高效。

5. **SFT 输出内容更丰富**：微调模型平均输出 3.53 条核心要点（vs 基座 v3 的 3.01 条），输出长度 ~351 字符（vs 基座 v3 的 ~287 字符），说明微调模型在格式合规的同时，信息密度也更高。

6. **Bad cases 极少**：微调模型 2 条格式缺失（0.4%），基座 v3 的 5 条 bad case 均为类别超出枚举，非结构字段缺失；两组结果均高质量，可直接用于实际部署评估。

---

## 7. 技术细节

### 7.1 微调参数配置

#### 7.1.1 LoRA 与量化设置
| 参数 | 4B 配置 | 8B 配置 | 设计意图 |
|------|---------|---------|----------|
| `quantization_bit` | 4 | 4 | 使用 NF4 格式量化，4B 权重仅约 2.5GB，大幅降低显存门槛 |
| `lora_target` | all | all | 覆盖所有线性层 (Attention + MLP)，最大化适应能力 |
| `lora_rank` | 8 | 16 | 8B 模型参数空间更大，提高 Rank 以保证表达能力 |
| `lora_alpha` | 16 | 32 | 保持 `alpha/rank = 2` 的缩放比例 |

#### 7.1.2 训练动态与显存优化
| 参数 | 4B 配置 | 8B 配置 | 设计意图 |
|------|---------|---------|----------|
| `batch_size` | 1 | 1 | 配合梯度累积使用，将单步显存占用降至最低 |
| `grad_accum` | 16 | 16 | 模拟有效 Batch Size = 16，稳定梯度下降方向 |
| `cutoff_len` | 1024 | 512 | 8B 模型显存紧张，限制序列长度以防 OOM |
| `learning_rate` | 2e-4 | 1e-4 | 8B 模型训练稳定性要求更高，降低 LR |

### 7.2 梯度累积原理

为在 16GB 显存上模拟大 Batch 训练，采用了 `batch_size=1` + `gradient_accumulation_steps=16` 策略：

1. **前向/反向传播**：每次只计算 1 条数据，累加梯度但不更新权重。
2. **权重更新**：每累积 16 次后，进行一次 Optimizer Step 并清空梯度。
3. **效果**：数学上等价于 Batch Size = 16，但显存峰值仅为 Batch Size = 1 的水平。

### 7.3 VRAM 占用分析 (Qwen3-4B)

| 显存占用项 | 预估大小 | 说明 |
|------------|----------|------|
| **Base Model (4-bit)** | ~2.5 GB | 冻结权重 |
| **LoRA Adapters** | ~0.1 GB | 可训练参数 (BF16) |
| **Gradients/Optimizer**| ~1.5 GB | 仅针对 LoRA 参数 |
| **Activation** | ~1.5 GB | 激活值 (Batch=1) |
| **KV Cache/Overhead** | ~1.0 GB | 框架开销 |
| **Total** | **~6.6 GB** | 安全运行于 8GB+ 显卡 |

> **注意**：若增加 Batch Size 至 2，激活值显存占用将翻倍，导致总占用超出 16GB 物理显存（溢出至共享内存），训练速度将下降 ~40%。

---

## 8. 常见问题

Q: 为什么生成的摘要有时候不包含【核心要点】？
A: 可能是训练步数不足或 `cutoff_len` 截断了输出。建议检查 ROUGE 报告中的 format rate。

Q: 如何解决 Windows 下的编码错误？
A: 设置环境变量 `PYTHONUTF8=1`，项目中所有文件读写均已强制指定 `utf-8` 编码。

更多问题请参阅 [docs/troubleshooting.md](docs/troubleshooting.md)。

---

## 9. 参考文献

1. [LLaMA-Factory: Unified Efficient Fine-Tuning of 100+ LLMs](https://github.com/hiyouga/LlamaFactory)
2. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
3. [XL-Sum: Large-Scale Multilingual Abstractive Summarization](https://aclanthology.org/2021.findings-acl.413/)

