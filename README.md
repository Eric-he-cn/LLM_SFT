# LLM_SFT


# 项目开发文档

## 项目名称

**Edge-News-Summarizer-FT：基于 Qwen + LLaMA-Factory 的新闻结构化摘要微调项目（QLoRA）**

## 项目目标

在本地消费级 GPU（RTX 5060 Ti 16GB）上，完成一个**新闻结构化摘要**小模型微调项目，实现：

* 输入：新闻标题 + 正文
* 输出：固定结构化摘要（摘要、要点、类别、主体、时间、影响）
* 训练方式：Qwen + QLoRA（LLaMA-Factory）
* 数据构建：原始新闻 + 高性能模型生成监督标签（SFT 数据）
* 评测：ROUGE + 格式正确率 + 关键信息覆盖率 + 推理延迟
* Demo：CLI 或 Gradio 页面

---

## 0. 方案说明（为什么这样设计）

### 0.1 为什么这个项目适合你的硬件

LLaMA-Factory 的 README 给了一个**硬件需求估计表**：7B 模型做 4-bit QLoRA 的估算显存占用在可行范围内（表里给的是 7B 的 4-bit QLoRA 级别），而你有 16GB 显存，适合做 3B/7B 的 QLoRA 微调（配合小 batch 和梯度累积）。([GitHub][1])

### 0.2 为什么选 LLaMA-Factory + Qwen

* LLaMA-Factory 官方支持 LoRA / QLoRA 等主流微调方式，并提供 CLI 和 YAML 配置范式。([GitHub][1])
* Qwen 官方文档直接给出了用 LLaMA-Factory 做 Qwen SFT 的示例（含 `template qwen`、LoRA 训练、LoRA merge 等）。([qwen.readthedocs.io][2])

### 0.3 为什么做“结构化摘要”而不是纯摘要

结构化输出更适合：

* 手机端 / 边缘端 UI 卡片展示
* 自动评测（字段存在性、格式正确率）
* 体现“微调价值”（输出稳定、字段齐全）

---

# 1. 硬件与软件环境要求

## 1.1 你的硬件信息（写入项目文档）

* **GPU**：NVIDIA RTX 5060 Ti 16GB（你当前设备）
* **显存**：16GB
* **CPU**：任意较新 x86_64（建议 6 核以上）
* **内存**：建议 32GB（最低 16GB 可跑，但数据处理会慢）
* **磁盘**：至少 80GB 可用空间（模型 + 数据 + checkpoint）

## 1.2 软件环境（推荐）

* **OS**：Windows 11（原生）或 WSL2 Ubuntu（更稳）
* **Python**：3.10（LLaMA-Factory 推荐区间内）([GitHub][1])
* **PyTorch**：GPU 版本
* **CUDA**：按本机驱动对应版本（Windows 原生建议走 PyTorch 官方 wheel）

> 说明：LLaMA-Factory README 在 Windows 部分给了 PyTorch CUDA 安装示例（`cu126`），并强调 Windows 需要手动安装 GPU 版 PyTorch。([GitHub][1])

---

# 2. 从零开始搭建环境（Conda + 依赖）

> 默认以下命令在 **PowerShell** 或 **Anaconda Prompt** 执行。
> 如果你用 WSL2，把路径和激活命令换成 Linux 习惯即可。

---

## 2.1 检查 GPU 驱动与 CUDA 可用性

### 命令

```bash
nvidia-smi
```

### 验收

* 能看到 RTX 5060 Ti
* 驱动版本正常
* CUDA Version 有显示（这只是驱动支持版本，不等于本地 toolkit）

---

## 2.2 新建 Conda 环境（Python 3.10）

```bash
conda create -n news_sft_qwen python=3.10 -y
conda activate news_sft_qwen
```

---

## 2.3 安装 GPU 版 PyTorch（Windows）

LLaMA-Factory 的 Windows 安装说明建议手动安装带 CUDA 的 PyTorch（示例是 `cu126`）。你可以先按这个装；如果本机驱动/环境更适合其他 CUDA wheel（如 cu121/cu124/cu126/cu128），按 PyTorch 官网选择即可。([GitHub][1])

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 验收

* `torch.cuda.is_available()` 输出 `True`

---

## 2.4 安装 LLaMA-Factory（源码安装）

LLaMA-Factory 官方 README 提供了源码安装方式：`pip install -e .`，评测相关依赖可装 `requirements/metrics.txt`。([GitHub][1])

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory

pip install -e .
pip install -r requirements/metrics.txt
```

---

## 2.5 安装 QLoRA 所需 bitsandbytes（Windows 重点）

LLaMA-Factory README 明确说明：**Windows 上做 QLoRA 需要安装预编译的 bitsandbytes wheel**，并指出该预编译版本支持 CUDA 11.1~12.2（需要按你的 CUDA 版本选择合适 release）。([GitHub][1])

### Windows 原生（优先尝试）

```bash
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.2.post2-py3-none-win_amd64.whl
```

### 验证

```bash
python -c "import bitsandbytes as bnb; print('bnb ok')"
```

### 如果失败（推荐方案）

由于 Windows 上 `bitsandbytes` 版本/显卡/CUDA 兼容有时会踩坑（尤其新显卡），**建议改用 WSL2 Ubuntu 跑训练**（Windows 保留做编辑与调试）。

> 这不是你项目的问题，是 Windows + QLoRA 的常见环境兼容问题。

---

## 2.6 安装项目额外依赖（数据处理 / 打标 / 评测 / Demo）

在 `LlamaFactory` 根目录下执行：

```bash
pip install -U \
  datasets \
  pandas \
  orjson \
  tqdm \
  pydantic \
  python-dotenv \
  openai \
  rouge-score \
  jieba \
  gradio \
  pyyaml \
  jsonlines
```

> `datasets` 用于加载公开新闻数据（如 CNN/DailyMail），`rouge-score` 用于摘要评测，`openai` 用于调用高性能模型生成监督标签。CNN/DailyMail 数据集在 Hugging Face 上有标准 summarization 数据卡，常用 ROUGE 做指标。([Hugging Face][3])

---

## 2.7 （可选）登录 Hugging Face

LLaMA-Factory README 也建议，如果使用某些需要确认/登录的数据集，先登录 Hugging Face。([GitHub][1])

```bash
pip install "huggingface_hub<1.0.0"
huggingface-cli login
```

---

# 3. 模型选择与下载

## 3.1 推荐模型（第一版）

* **Qwen/Qwen2.5-3B-Instruct**（先跑通）
* **Qwen/Qwen2.5-7B-Instruct**（正式版）

Qwen 的 Hugging Face 模型卡里给了标准加载示例，模型名可以直接用 `Qwen/Qwen2.5-7B-Instruct`。([Hugging Face][4])

## 3.2 本地下载（推荐）

在 `LlamaFactory` 根目录执行（可选）：

```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir ./models/Qwen2.5-3B-Instruct
# 或
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct
```

> 也可以训练时直接用远程模型名，不提前下载。

---

# 4. 项目目录结构（Codex 按这个生成）

> **注意**：下面目录是在 `LlamaFactory` 仓库根目录下创建一个子项目目录（推荐），避免改动官方仓库太散。

```text
LlamaFactory/
├─ projects/
│  └─ edge_news_summarizer/
│     ├─ README.md
│     ├─ .env.example
│     ├─ configs/
│     │  ├─ train_qwen25_3b_qlora_news.yaml
│     │  ├─ train_qwen25_7b_qlora_news.yaml
│     │  └─ infer_news.yaml
│     ├─ data/
│     │  ├─ raw/
│     │  │  ├─ news_raw.jsonl
│     │  │  └─ sample_raw.jsonl
│     │  ├─ labeled/
│     │  │  ├─ news_labeled_v1.jsonl
│     │  │  └─ label_errors.jsonl
│     │  ├─ cleaned/
│     │  │  ├─ train.json
│     │  │  ├─ val.json
│     │  │  ├─ test.json
│     │  │  └─ test_manual_eval.json
│     │  ├─ prompts/
│     │  │  └─ label_prompt_news_structured.txt
│     │  └─ reports/
│     │     ├─ data_quality_report.md
│     │     └─ labeling_stats.json
│     ├─ scripts/
│     │  ├─ 01_collect_news.py
│     │  ├─ 02_generate_labels_api.py
│     │  ├─ 03_validate_and_clean.py
│     │  ├─ 04_split_dataset.py
│     │  ├─ 05_register_dataset_info.py
│     │  ├─ 06_eval_rouge_and_format.py
│     │  ├─ 07_benchmark_latency.py
│     │  └─ 08_demo_cli.py
│     ├─ outputs/
│     │  ├─ checkpoints/
│     │  ├─ merged/
│     │  ├─ logs/
│     │  └─ eval/
│     └─ docs/
│        ├─ labeling_guideline.md
│        ├─ dev_plan.md
│        └─ troubleshooting.md
└─ data/
   └─ dataset_info.json   # LLaMA-Factory 官方文件（会追加你的数据集定义）
```

---

# 5. 任务定义与输出格式（训练目标）

## 5.1 输入格式（新闻）

建议原始数据字段统一为：

```json
{
  "id": "news_000001",
  "title": "新闻标题",
  "content": "新闻正文……",
  "source": "来源（可选）",
  "published_at": "2026-02-25 10:30:00（可选）",
  "lang": "zh"
}
```

## 5.2 结构化摘要输出模板（固定）

> 训练时所有标签都必须严格遵守这个模板

```text
【一句话摘要】
...

【核心要点】
1. ...
2. ...
3. ...

【事件类别】
科技/财经/国际/社会/体育/娱乐/其他

【涉及主体】
- ...
- ...

【时间信息】
...

【影响或后续关注】
...
```

---

# 6. 数据集构建方案（高性能模型打标）

## 6.1 原始新闻来源（建议）

### 方案 A（最快起步）

用 Hugging Face 的新闻摘要数据（如 CNN/DailyMail）先跑通全流程，适合验证训练/评测脚本。CNN/DailyMail 数据集卡明确它是新闻摘要常用数据，并提到 ROUGE 作为常见评测方式。([Hugging Face][3])

### 方案 B（正式版，中文）

自己准备中文新闻原文（公开新闻、RSS、合规抓取或自有数据），再用高性能模型生成结构化标签。

> 也可以参考 LCSTS（中文摘要数据）这类公开中文摘要研究数据思路。LCSTS 是中文摘要领域经典公开数据集（ACL 2015）。([ACL Anthology][5])

---

## 6.2 标注规则（必须先写，Codex 生成 `docs/labeling_guideline.md`）

### 一句话摘要

* 1~2 句
* 必须包含“主体 + 事件 + 结果/影响”
* 禁止主观评价、禁止夸张词

### 核心要点

* 优先输出 3 条（不足可 2 条）
* 每条一句
* 优先保留数字、时间、动作、影响

### 事件类别

* 仅允许：`科技/财经/国际/社会/体育/娱乐/其他`

### 涉及主体

* 提取 1~5 个（组织、公司、人物、国家）
* 不要泛词（如“专家”“记者”）

### 时间信息

* 优先提取文中明确时间
* 若无明确时间，写：`未明确提及`

### 影响或后续关注

* 1 句
* 仅基于原文，不得编造

---

## 6.3 用高性能模型生成监督标签（SFT 数据）

### 6.3.1 `.env.example`

Codex 创建文件：`projects/edge_news_summarizer/.env.example`

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
LABEL_MODEL=gpt-5-mini
```

> 你也可以换成兼容 OpenAI SDK 的其他模型服务（如 DeepSeek/OpenRouter 等兼容接口）。

---

## 6.3.2 打标 Prompt（Codex 创建）

文件：`data/prompts/label_prompt_news_structured.txt`

```text
你是一个新闻结构化摘要助手。请严格基于给定新闻内容生成摘要，禁止编造事实，禁止加入原文不存在的信息。

输出必须严格使用以下模板，不要增加其他字段，不要省略字段：

【一句话摘要】
...

【核心要点】
1. ...
2. ...
3. ...

【事件类别】
科技/财经/国际/社会/体育/娱乐/其他

【涉及主体】
- ...
- ...

【时间信息】
...

【影响或后续关注】
...

要求：
1) 客观、简洁、准确
2) 如果原文没有明确时间，写“未明确提及”
3) 如果主体不止一个，按重要性排序
4) 事件类别只能从给定标签中选一个
5) 不要输出 Markdown 代码块
```

---

## 6.3.3 打标脚本（Codex 实现）

文件：`scripts/02_generate_labels_api.py`

### 功能要求

* 读取 `data/raw/news_raw.jsonl`
* 对每条新闻调用高性能模型 API 生成结构化摘要
* 保存到 `data/labeled/news_labeled_v1.jsonl`
* 失败样本保存到 `data/labeled/label_errors.jsonl`
* 支持断点续跑（跳过已打标 `id`）
* 控制速率（避免 API 限流）
* 打印进度和耗时

### 输出格式（labeled）

```json
{
  "id": "news_000001",
  "title": "...",
  "content": "...",
  "structured_summary": "【一句话摘要】\n...\n\n【核心要点】\n1. ...",
  "label_model": "gpt-5-mini",
  "created_at": "2026-02-25T12:00:00Z"
}
```

---

# 7. 数据清洗与转换（LLaMA-Factory 格式）

LLaMA-Factory 文档说明自定义数据集需要：

1. 数据文件用 Alpaca 或 ShareGPT 格式
2. 在 `data/dataset_info.json` 里注册数据集定义。([llamafactory.readthedocs.io][6])

---

## 7.1 规则校验脚本（Codex 实现）

文件：`scripts/03_validate_and_clean.py`

### 需要做的校验

* 字段完整性（是否包含 6 个标题块）
* 类别是否合法（是否在 7 类标签里）
* 核心要点编号格式是否正确（`1.` / `2.` / `3.`）
* 内容长度过滤（过短、过长）
* 去重（标题+正文 hash）

### 输出

* `data/reports/data_quality_report.md`
* `data/reports/labeling_stats.json`
* 清洗后的中间文件（JSONL）

---

## 7.2 划分训练/验证/测试集（Codex 实现）

文件：`scripts/04_split_dataset.py`

### 目标比例

* train: 90%
* val: 5%
* test: 5%

### 约束

* 随机种子固定（如 42）
* 尽量按类别分层抽样（如果类别字段可用）

---

## 7.3 转换为 Alpaca 格式（LLaMA-Factory SFT）

文件：`data/cleaned/train.json` 等

LLaMA-Factory 的 Alpaca SFT 格式字段是 `instruction`、`input`、`output`（`system` 可选）。文档里给了明确样例。([llamafactory.readthedocs.io][6])

### 目标格式（示例）

```json
[
  {
    "instruction": "请对以下新闻进行结构化总结，按指定格式输出：一句话摘要、核心要点、事件类别、涉及主体、时间信息、影响或后续关注。",
    "input": "标题：某科技公司发布新芯片\n正文：......",
    "output": "【一句话摘要】\n...\n\n【核心要点】\n1. ...\n2. ...\n3. ...\n\n【事件类别】\n科技\n\n【涉及主体】\n- ...",
    "system": "你是一个新闻摘要助手，要求客观、准确、简洁，禁止编造事实。"
  }
]
```

---

## 7.4 注册 `dataset_info.json`（Codex 自动追加）

文件：`LlamaFactory/data/dataset_info.json`

Qwen 文档和 LLaMA-Factory 数据文档都给了 Alpaca 格式的字段映射定义。([qwen.readthedocs.io][2])

### 追加条目（示例）

```json
{
  "edge_news_structured_train": {
    "file_name": "projects/edge_news_summarizer/data/cleaned/train.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "system": "system"
    }
  },
  "edge_news_structured_val": {
    "file_name": "projects/edge_news_summarizer/data/cleaned/val.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "system": "system"
    }
  }
}
```

> 注意：`dataset_info.json` 是大文件，Codex 要做“安全追加”，不要覆盖已有内容。

---

# 8. 训练配置（LLaMA-Factory YAML）

## 8.1 训练模板说明

* 使用 `llamafactory-cli train`（官方文档示例）([llamafactory.readthedocs.io][7])
* Qwen 官方示例训练命令使用 `--template qwen`，并提示 `cutoff_len` 过大可能 OOM。([qwen.readthedocs.io][2])
* LLaMA-Factory README 也提醒训练和推理要用同一个 template。([GitHub][1])

---

## 8.2 `configs/train_qwen25_3b_qlora_news.yaml`（Codex 创建）

> 下面是建议配置（先跑通 3B）

```yaml
### model
model_name_or_path: ./models/Qwen2.5-3B-Instruct
template: qwen
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
quantization_bit: 4
double_quantization: true
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05

### dataset
dataset: edge_news_structured_train
eval_dataset: edge_news_structured_val
cutoff_len: 1024
max_samples: 3000
overwrite_cache: true
preprocessing_num_workers: 4

### output
output_dir: ./projects/edge_news_summarizer/outputs/checkpoints/qwen25_3b_news_qlora
logging_steps: 10
save_steps: 200
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
ddp_timeout: 180000

### eval
val_size: 0.0
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 200

### misc
report_to: none
dataloader_num_workers: 0  # Windows常见兼容设置
```

> `dataloader_num_workers: 0` 是 Windows 上常见兼容设置，LLaMA-Factory README 也提到 Windows 下可以这么处理某些报错。([GitHub][1])

---

## 8.3 `configs/train_qwen25_7b_qlora_news.yaml`（Codex 创建）

> 正式版 7B（你 16GB 显存可尝试，参数更保守）

```yaml
### model
model_name_or_path: ./models/Qwen2.5-7B-Instruct
template: qwen
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
quantization_bit: 4
double_quantization: true
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05

### dataset
dataset: edge_news_structured_train
eval_dataset: edge_news_structured_val
cutoff_len: 768
max_samples: 3000
overwrite_cache: true
preprocessing_num_workers: 2

### output
output_dir: ./projects/edge_news_summarizer/outputs/checkpoints/qwen25_7b_news_qlora
logging_steps: 10
save_steps: 200
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 8.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
ddp_timeout: 180000

### eval
val_size: 0.0
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 200

### misc
report_to: none
dataloader_num_workers: 0
```

> 如果 OOM，优先降 `cutoff_len`（Qwen 文档也特别提醒这个参数和 OOM 关系很大）。([qwen.readthedocs.io][2])

---

# 9. 训练与推理命令（可直接执行）

## 9.1 开始训练（3B 先跑通）

LLaMA-Factory SFT 文档和 README 都给了 `llamafactory-cli train` 的标准用法。([llamafactory.readthedocs.io][7])

```bash
cd LlamaFactory
llamafactory-cli train ./projects/edge_news_summarizer/configs/train_qwen25_3b_qlora_news.yaml
```

## 9.2 训练正式版（7B）

```bash
llamafactory-cli train ./projects/edge_news_summarizer/configs/train_qwen25_7b_qlora_news.yaml
```

---

## 9.3 推理（LoRA Adapter）

LLaMA-Factory README 提供了 `chat` / `export` 的快速命令。([GitHub][1])

### CLI 聊天测试（可改成固定新闻输入）

```bash
llamafactory-cli chat ./projects/edge_news_summarizer/configs/infer_news.yaml
```

---

## 9.4 合并 LoRA（导出）

Qwen 文档给了 LoRA merge 的 `llamafactory-cli export` 示例（`template qwen`）。([qwen.readthedocs.io][2])

```bash
llamafactory-cli export \
  --model_name_or_path ./models/Qwen2.5-7B-Instruct \
  --adapter_name_or_path ./projects/edge_news_summarizer/outputs/checkpoints/qwen25_7b_news_qlora \
  --template qwen \
  --finetuning_type lora \
  --export_dir ./projects/edge_news_summarizer/outputs/merged/qwen25_7b_news_merged \
  --export_size 2 \
  --export_legacy_format False
```

---

# 10. 评测设计（Codex 必做）

## 10.1 自动评测指标（脚本：`06_eval_rouge_and_format.py`）

### A. ROUGE（摘要质量）

* ROUGE-1
* ROUGE-2
* ROUGE-L

> 新闻摘要任务常用 ROUGE，CNN/DailyMail 数据卡也明确提到这一点。([Hugging Face][3])

### B. 格式正确率（重点）

统计以下项目的通过率：

* 是否包含 6 个字段标题
* 核心要点是否为编号列表
* 类别标签是否在允许集合中
* “时间信息”字段是否存在

**输出示例**

```json
{
  "format_pass_rate": 0.93,
  "missing_field_rate": 0.04,
  "invalid_category_rate": 0.02
}
```

---

## 10.2 人工评测（小样本 100 条）

Codex 生成 `data/cleaned/test_manual_eval.json`，并在 `docs/labeling_guideline.md` 增加评分标准。

### 建议评分项（0/1）

* 是否提到关键主体
* 是否提到核心事件
* 是否提到关键时间（若原文有）
* 是否有事实错误
* 是否有明显幻觉

---

## 10.3 推理性能评测（脚本：`07_benchmark_latency.py`）

统计：

* 平均推理时延（单篇新闻）
* P50 / P95 延迟
* 显存占用（可用 `torch.cuda.max_memory_allocated()`）

---

# 11. Demo（CLI 优先，Gradio 可选）

## 11.1 CLI Demo（Codex 实现 `08_demo_cli.py`）

功能：

* 输入新闻标题和正文（或读取 txt/json）
* 调用本地模型（base / finetuned）
* 输出结构化摘要
* 可选保存结果到 `outputs/eval/demo_outputs.jsonl`

## 11.2 Gradio Demo（可选）

简单页面：

* 左侧输入：标题、正文
* 右侧输出：结构化摘要
* 复选项：`使用微调模型 / 使用基座模型`
* 显示推理耗时

---

# 12. 给 Codex 的任务清单（按顺序执行）

> 这一段你可以原封不动发给 Codex。

## Phase 1：项目骨架与环境说明

1. 在 `LlamaFactory/projects/edge_news_summarizer` 下创建目录结构（见本文）
2. 生成 `README.md`（简版项目介绍）
3. 生成 `.env.example`
4. 生成 `docs/dev_plan.md`（拷贝本开发文档精简版）

## Phase 2：数据处理与打标

5. 实现 `scripts/01_collect_news.py`

   * 支持从本地 JSONL / HuggingFace dataset 导入新闻
   * 输出统一格式 `data/raw/news_raw.jsonl`
6. 实现 `scripts/02_generate_labels_api.py`

   * 使用 OpenAI 兼容 SDK
   * 支持断点续跑、重试、失败日志
7. 实现 `scripts/03_validate_and_clean.py`

   * 模板校验、类别校验、去重、长度过滤
8. 实现 `scripts/04_split_dataset.py`

   * 生成 `train/val/test`
9. 实现 `scripts/05_register_dataset_info.py`

   * 自动向 `LlamaFactory/data/dataset_info.json` 追加配置（避免覆盖）

## Phase 3：训练配置与训练

10. 生成 `configs/train_qwen25_3b_qlora_news.yaml`
11. 生成 `configs/train_qwen25_7b_qlora_news.yaml`
12. 生成 `configs/infer_news.yaml`
13. 在 `README.md` 里写清训练命令

## Phase 4：评测与 Demo

14. 实现 `scripts/06_eval_rouge_and_format.py`
15. 实现 `scripts/07_benchmark_latency.py`
16. 实现 `scripts/08_demo_cli.py`
17. （可选）增加 `demo_gradio.py`

## Phase 5：文档与结果展示

18. 生成 `docs/labeling_guideline.md`
19. 生成 `docs/troubleshooting.md`
20. 在 README 中补充：

* 环境安装步骤
* 数据构建步骤
* 训练步骤
* 评测步骤
* 常见问题

---

# 13. 关键脚本实现要求（让 Codex 别偷懒）

## 13.1 `02_generate_labels_api.py` 实现细则

* 使用 `python-dotenv` 读取 `.env`
* 使用 `OpenAI` 客户端（兼容接口）
* 每次请求超时 + 重试（指数退避）
* 解析返回文本，做初步模板检查
* 每处理 N 条自动 flush 到磁盘（防中断丢数据）

## 13.2 `03_validate_and_clean.py` 实现细则

* 用正则提取 6 个字段
* 检查“事件类别”是否在白名单
* 检查“核心要点”行数与编号
* 记录每类错误数量
* 输出清洗统计

## 13.3 `06_eval_rouge_and_format.py` 实现细则

* 输入：测试集 + 模型推理结果
* 输出：

  * `rouge_report.json`
  * `format_report.json`
  * `bad_cases.jsonl`（格式失败样本）

---

# 14. 环境与训练常见问题（先写进 `docs/troubleshooting.md`）

## 14.1 `torch.cuda.is_available() == False`

* 检查 `nvidia-smi`
* 确认装的是 GPU 版 PyTorch（不是 CPU 版）
* 重装 PyTorch CUDA wheel（如 `cu126`）

## 14.2 `bitsandbytes` 导入失败（Windows 常见）

* 先确认 wheel 安装成功
* 若显卡/CUDA 过新导致兼容问题，改用 **WSL2 Ubuntu** 跑训练
* 第一版可先改为 LoRA（非 QLoRA）验证流程，但显存压力会更大

## 14.3 训练 OOM

按顺序降：

1. `cutoff_len`
2. `per_device_train_batch_size`
3. `lora_rank`
4. 换 3B 模型先跑通

> Qwen 官方 LLaMA-Factory 文档也提示 `cutoff_len` 是避免 OOM 的关键参数。([qwen.readthedocs.io][2])

## 14.4 模板不一致导致输出怪异

* 训练和推理都要用同一个 `template`
* LLaMA-Factory README 明确提醒训练/推理模板要一致。([GitHub][1])

---

# 15. 里程碑与验收标准（项目管理视角）

## M1：数据管线跑通（Day 1~3）

**验收**

* 能生成 `news_raw.jsonl`
* 能调用 API 生成 100 条结构化标签
* 清洗脚本能产出 `train/val/test`

## M2：训练跑通（Day 3~5）

**验收**

* 3B QLoRA 能完成 1 个 epoch
* 能保存 LoRA adapter
* 能用 CLI 输出结构化摘要

## M3：评测与对比（Day 5~7）

**验收**

* 有 Base vs FT 对比样例
* 有 ROUGE 和格式正确率报告
* 有至少 20 条 bad case 分析

## M4：Demo 与文档（Day 7+）

**验收**

* CLI 或 Gradio 可演示
* README 完整（安装、训练、评测）
* 可用于简历/面试展示

---

# 16. README（Codex 最终生成内容建议）

README 至少包含：

1. 项目背景（边缘端新闻总结）
2. 项目亮点（结构化摘要、QLoRA、本地部署）
3. 环境安装（Conda + PyTorch + LLaMA-Factory）
4. 数据构建（原始新闻 → API 打标 → 清洗 → Alpaca）
5. 训练命令
6. 推理命令
7. 评测命令
8. 常见问题
9. 结果示例（Base vs FT）

---

# 17. 可选升级（第二阶段，先不做）

* 多篇新闻聚合总结（同一事件聚合）
* 个性化摘要风格（极简版/专业版）
* 新闻 + 邮件摘要迁移（你最终目标）
* DPO 偏好优化（摘要更简洁/更客观）

---


[1]: https://github.com/hiyouga/LlamaFactory "GitHub - hiyouga/LlamaFactory: Unified Efficient Fine-Tuning of 100+ LLMs & VLMs (ACL 2024)"
[2]: https://qwen.readthedocs.io/en/v2.5/training/SFT/llama_factory.html "LLaMA-Factory - Qwen"
[3]: https://huggingface.co/datasets/ccdv/cnn_dailymail "ccdv/cnn_dailymail · Datasets at Hugging Face"
[4]: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct "Qwen/Qwen2.5-7B-Instruct · Hugging Face"
[5]: https://aclanthology.org/D15-1229/?utm_source=chatgpt.com "LCSTS: A Large Scale Chinese Short Text Summarization ..."
[6]: https://llamafactory.readthedocs.io/en/latest/getting_started/data_preparation.html "Data Preparation - LLaMA Factory"
[7]: https://llamafactory.readthedocs.io/en/latest/getting_started/sft.html "Supervised Fine-tuning - LLaMA Factory"
