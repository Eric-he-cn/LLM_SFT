# Edge-News-Summarizer-FT

基于 **Qwen + LLaMA-Factory** 的新闻结构化摘要微调项目（QLoRA），面向本地消费级 GPU（RTX 5060 Ti 16GB）。

## 项目亮点

- 📰 **结构化输出**：固定 6 字段格式（摘要、要点、类别、主体、时间、影响），适合边缘端 UI 卡片展示
- 🚀 **QLoRA 高效微调**：4-bit 量化 + LoRA，16GB 显存可跑 3B/7B 模型
- 🏭 **完整数据管线**：原始新闻 → API 打标 → 清洗 → Alpaca 格式，端到端可复现
- 📊 **自动评测**：ROUGE + 格式正确率 + 推理延迟，量化微调效果
- 🖥️ **CLI Demo**：本地交互推理，支持微调模型与基座模型对比

---

## 环境要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA RTX 5060 Ti 16GB（或等效） |
| CUDA | 12.x |
| Python | 3.10 |
| OS | Windows 11 / WSL2 Ubuntu |
| 磁盘 | 80GB+ |

---

## 快速开始

### 1. 创建 Conda 环境

```bash
conda create -n news_sft_qwen python=3.10 -y
conda activate news_sft_qwen
```

### 2. 安装 GPU 版 PyTorch

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available())"  # 应输出 True
```

### 3. 安装 LLaMA-Factory

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .
pip install -r requirements/metrics.txt
```

### 4. 安装额外依赖

```bash
pip install -U datasets pandas tqdm pydantic python-dotenv openai \
               rouge-score jieba gradio pyyaml jsonlines
```

### 5. 安装 bitsandbytes（QLoRA 必须）

```bash
# Windows 原生
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.2.post2-py3-none-win_amd64.whl

# Linux / WSL2
pip install bitsandbytes
```

### 6. 配置 API Key

```bash
cp projects/edge_news_summarizer/.env.example projects/edge_news_summarizer/.env
# 编辑 .env，填入 OPENAI_API_KEY
```

---

## 下载模型

```bash
# 3B 模型（推荐先跑）
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir ./models/Qwen2.5-3B-Instruct

# 7B 模型（正式版）
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct
```

---

## 数据构建流程

> 所有脚本在 **LlamaFactory 根目录**执行

### Step 1：收集新闻数据

```bash
# 从 HuggingFace CNN/DailyMail 数据集收集
python projects/edge_news_summarizer/scripts/01_collect_news.py \
  --source hf --dataset cnn_dailymail --split train --max_samples 5000

# 或从本地 JSONL 文件导入
python projects/edge_news_summarizer/scripts/01_collect_news.py \
  --source local --input /path/to/news.jsonl
```

**输出**：`projects/edge_news_summarizer/data/raw/news_raw.jsonl`

### Step 2：API 生成结构化标签

```bash
python projects/edge_news_summarizer/scripts/02_generate_labels_api.py \
  --max_samples 2000
```

**输出**：`projects/edge_news_summarizer/data/labeled/news_labeled_v1.jsonl`

### Step 3：数据校验与清洗

```bash
python projects/edge_news_summarizer/scripts/03_validate_and_clean.py
```

**输出**：`projects/edge_news_summarizer/data/cleaned/cleaned_all.jsonl`
**报告**：`projects/edge_news_summarizer/data/reports/data_quality_report.md`

### Step 4：划分数据集

```bash
python projects/edge_news_summarizer/scripts/04_split_dataset.py
```

**输出**：`data/cleaned/train.json` / `val.json` / `test.json`

### Step 5：注册数据集到 LLaMA-Factory

```bash
python projects/edge_news_summarizer/scripts/05_register_dataset_info.py
```

---

## 训练

### 3B 模型（推荐先跑通）

```bash
llamafactory-cli train projects/edge_news_summarizer/configs/train_qwen25_3b_qlora_news.yaml
```

### 7B 模型（正式版）

```bash
llamafactory-cli train projects/edge_news_summarizer/configs/train_qwen25_7b_qlora_news.yaml
```

### 合并 LoRA 权重

```bash
llamafactory-cli export \
  --model_name_or_path ./models/Qwen2.5-3B-Instruct \
  --adapter_name_or_path projects/edge_news_summarizer/outputs/checkpoints/qwen25-3b-qlora-news \
  --template qwen \
  --finetuning_type lora \
  --export_dir projects/edge_news_summarizer/outputs/merged/qwen25-3b-news
```

---

## 推理 Demo

### 交互式 CLI

```bash
python projects/edge_news_summarizer/scripts/08_demo_cli.py \
  --model_path ./models/Qwen2.5-3B-Instruct \
  --adapter_path projects/edge_news_summarizer/outputs/checkpoints/qwen25-3b-qlora-news
```

### 批量推理

```bash
python projects/edge_news_summarizer/scripts/08_demo_cli.py \
  --model_path ./models/Qwen2.5-3B-Instruct \
  --adapter_path projects/edge_news_summarizer/outputs/checkpoints/qwen25-3b-qlora-news \
  --input_file projects/edge_news_summarizer/data/cleaned/test.json \
  --output_file projects/edge_news_summarizer/outputs/eval/demo_outputs.jsonl \
  --num_samples 50
```

---

## 评测

### ROUGE + 格式正确率

```bash
python projects/edge_news_summarizer/scripts/06_eval_rouge_and_format.py \
  --test projects/edge_news_summarizer/data/cleaned/test.json \
  --predictions projects/edge_news_summarizer/outputs/eval/demo_outputs.jsonl
```

### 推理延迟评测

```bash
python projects/edge_news_summarizer/scripts/07_benchmark_latency.py \
  --model_path ./models/Qwen2.5-3B-Instruct \
  --adapter_path projects/edge_news_summarizer/outputs/checkpoints/qwen25-3b-qlora-news \
  --num_samples 20
```

---

## 输出示例

### 输入

**标题**：苹果发布 iPhone 16 系列，全系支持 Apple Intelligence

**正文**：苹果公司于 2024 年 9 月 9 日举办秋季发布会，推出 iPhone 16 系列四款新机...

### 微调模型输出（FT）

```
【一句话摘要】
苹果于2024年9月9日发布iPhone 16系列，全系搭载A18芯片并支持Apple Intelligence AI功能。

【核心要点】
1. iPhone 16系列共四款机型，起售价799美元
2. 全系搭载苹果A18芯片，支持Apple Intelligence
3. 新增融合相机系统，支持4K 120fps视频录制
4. 中国大陆首批不支持Apple Intelligence功能

【事件类别】
科技

【主要主体】
苹果公司、蒂姆·库克

【时间信息】
2024年9月9日发布，2024年9月20日开售

【潜在影响】
此次发布将加速科技巨头AI功能集成竞争，可能影响三星等竞争对手战略布局；AI功能在部分地区的缺席或影响当地销量。
```

### 基座模型输出（Base，无微调）

```
这款手机非常好，苹果公司做了很多创新...（格式不规范，字段缺失）
```

---

## 项目结构

```
projects/edge_news_summarizer/
├── README.md              # 本文件
├── .env.example           # API 配置模板
├── configs/               # 训练/推理配置
│   ├── train_qwen25_3b_qlora_news.yaml
│   ├── train_qwen25_7b_qlora_news.yaml
│   └── infer_news.yaml
├── data/
│   ├── raw/               # 原始新闻数据
│   ├── labeled/           # API 打标后数据
│   ├── cleaned/           # 清洗后训练数据
│   ├── prompts/           # 标注 prompt 模板
│   └── reports/           # 数据质量报告
├── scripts/               # 各阶段脚本
│   ├── 01_collect_news.py
│   ├── 02_generate_labels_api.py
│   ├── 03_validate_and_clean.py
│   ├── 04_split_dataset.py
│   ├── 05_register_dataset_info.py
│   ├── 06_eval_rouge_and_format.py
│   ├── 07_benchmark_latency.py
│   └── 08_demo_cli.py
├── outputs/               # 训练输出
│   ├── checkpoints/       # LoRA adapter
│   ├── merged/            # 合并后的完整模型
│   ├── logs/              # 训练日志
│   └── eval/              # 评测结果
└── docs/
    ├── dev_plan.md        # 开发计划
    ├── labeling_guideline.md  # 标注指南
    └── troubleshooting.md    # 常见问题
```

---

## 常见问题

见 [docs/troubleshooting.md](docs/troubleshooting.md)

---

## 参考资源

- [LLaMA-Factory GitHub](https://github.com/hiyouga/LlamaFactory)
- [Qwen LLaMA-Factory 文档](https://qwen.readthedocs.io/en/v2.5/training/SFT/llama_factory.html)
- [Qwen2.5-7B-Instruct HuggingFace](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [CNN/DailyMail 数据集](https://huggingface.co/datasets/ccdv/cnn_dailymail)
