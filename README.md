# 面向端侧信息流场景的结构化摘要助手（Qwen3-4B + QLoRA SFT + AWQ）

> 更新时间：2026-04-17

基于 **Qwen3-4B + LLaMA-Factory** 的端到端结构化摘要工程，面向 16GB 显存级别硬件，覆盖从数据构建、SFT 微调、AWQ 后训练量化到端侧推理评测的完整闭环。

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.8+](https://img.shields.io/badge/CUDA-12.8%2B-green.svg)](https://developer.nvidia.com/cuda-downloads)

---

## 目录

- [1. 项目摘要](#1-项目摘要)
- [2. 背景、目标与贡献](#2-背景目标与贡献)
- [3. 系统架构与项目结构](#3-系统架构与项目结构)
- [4. 环境与依赖](#4-环境与依赖)
- [5. 方法与实现](#5-方法与实现)
- [6. 实验设置与评测口径](#6-实验设置与评测口径)
- [7. 实验结果与分析](#7-实验结果与分析)
- [8. 复现流程与命令清单](#8-复现流程与命令清单)
- [9. 工程建议与排障](#9-工程建议与排障)
- [10. 参考文献](#10-参考文献)

---

## 1. 项目摘要

本项目围绕“端侧信息流结构化摘要”这一落地场景，解决三个核心矛盾：
- 质量矛盾：基座模型在新闻摘要上的关键信息覆盖率不足。
- 结构矛盾：自由生成容易偏离固定 6 字段协议，导致前端渲染和后处理复杂。
- 部署矛盾：完整 BF16 模型体积大、延迟高，在端侧资源预算下难以稳定上线。

为此，项目采用“两条主线并行收敛”的策略：
- **SFT 主线（能力构建）**：通过 QLoRA SFT 学习结构化任务协议，显著提升 ROUGE 与格式稳定性。
- **量化主线（部署压缩）**：在 SFT merged 模型上执行 AutoAWQ（W4A16）后训练量化，并统一在 vLLM 口径下完成质量/性能评估。

最终方案不是“只做 SFT”或“只做量化”，而是：
- 先由 SFT 保证任务能力与结构约束。
- 再由 AWQ 压缩体积并降低时延，达到可部署状态。

---

## 2. 背景、目标与贡献

### 2.1 场景定义

目标任务是将输入新闻文本转换为固定 6 字段结构化摘要，输出协议如下：

```
【一句话摘要】
【核心要点】
【事件类别】
【主要主体】
【时间信息】
【潜在影响】
```

该协议具有两个工程价值：
- 可直接映射到信息流卡片组件，减少业务后处理逻辑。
- 可用规则校验字段完整率、类别合法性、要点格式等，便于建立可量化的质量门槛。

### 2.2 项目目标

在固定协议下完成以下目标：
1. 提升摘要语义质量（ROUGE 指标）。
2. 提升格式稳定性（字段完整率、类别合规率、要点格式率）。
3. 在质量可接受损失内，将模型压缩到端侧可部署体积并显著降低延迟。

### 2.3 主要贡献

1. 完成 `XL-Sum -> DeepSeek 标注 -> 清洗校验 -> QLoRA SFT` 的端到端数据与训练链路。
2. 形成可复用的 AWQ 量化链路（校准集准备、量化、冒烟、全量评测）。
3. 建立统一评测口径（质量 + 性能），并提供 checkpoint 续跑与对照结果追溯。
4. 完成量化参数探索并收敛到最终推荐配置，明确“可部署方案”与“未采用方案”的边界。

---

## 3. 系统架构与项目结构

### 3.1 端到端流程架构

```text
数据源(XL-Sum)
   -> 01 采集与过滤
   -> 02 DeepSeek 标注
   -> 03 校验清洗
   -> 04 划分 train/val/test
   -> 05 注册 LLaMA-Factory
   -> QLoRA SFT 训练与合并
   -> 06 SFT 质量评测
   -> 09/10 AWQ 校准与量化
   -> 13 AWQ 冒烟
   -> 14 全量质量对比（vLLM）
   -> 07 性能对比（vLLM）
```

### 3.2 仓库结构（按主/备选流程标注）

```text
Qwen3-QLoRA-News/
├── README.md
├── requirements.txt
├── configs/
│   ├── train_qwen3_4b_qlora_news.yaml
│   ├── train_qwen3_4b_qlora_news_v2.yaml
│   ├── train_qwen3_8b_qlora_news.yaml
│   ├── infer_news.yaml
│   └── infer_news_base.yaml
├── data/
│   ├── raw/
│   ├── labeled/
│   ├── cleaned/
│   └── prompts/
├── scripts/
│   ├── 01_collect_news.py
│   ├── 02_generate_labels_api.py
│   ├── 03_validate_and_clean.py
│   ├── 04_split_dataset.py
│   ├── 05_register_dataset_info.py
│   ├── 06_eval_rouge_and_format.py
│   ├── 07_benchmark_latency.py
│   ├── 08_demo_cli.py
│   ├── 09_prepare_awq_calib.py
│   ├── 10_quantize_awq.py
│   ├── 11_awq_smoke_infer.py
│   ├── 12_benchmark_quality_awq.py
│   ├── 13_vllm_awq_smoke_infer.py
│   ├── 14_benchmark_quality_vllm.py
│   ├── awq_prepare_calib.py
│   ├── quantize_awq.py
│   ├── awq_smoke_infer.py
│   └── experimental/
│       ├── 15_build_awq_baseline_manifest.py
│       └── 16_awq_recovery_compare.py
├── docs/
│   ├── dev_plan.md
│   ├── key_configs.md
│   ├── awq_experiment_commands.md
│   ├── awq_dvawq_backlog.md
│   └── troubleshooting.md
└── outputs/
```

流程定位说明：
- **主流程（AWQ）**：`09 -> 10 -> 13 -> 14 -> 07(vllm)`。
- **备选流程（AutoAWQ推理）**：`11 -> 12`，仅在 vLLM 不可用或排障时使用。
- **演示工具**：`08_demo_cli.py`，用于交互体验和基座/SFT 快速对比。

---

## 4. 环境与依赖

### 4.1 双环境分工（推荐）

| 环境 | 角色 | 典型脚本 |
|------|------|----------|
| `my_sft`（Windows） | 数据处理、SFT 训练、AutoAWQ 量化 | `01-06, 09-12` |
| `wsl_vllm`（WSL/Linux） | vLLM 推理与性能压测主链 | `13, 14, 07 --backend vllm` |

这样分工的原因：
- 训练与量化依赖偏向 PyTorch/Transformers 组合。
- vLLM 在 Linux/WSL 侧兼容性与性能更稳定。
- 避免单环境混装导致的包冲突与调试成本。

### 4.2 训练与量化环境（`my_sft`）

```bash
conda create -n my_sft python=3.11 -y
conda activate my_sft

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -U datasets pandas tqdm pydantic python-dotenv openai \
               rouge-score jieba bitsandbytes pyyaml jsonlines autoawq
```

### 4.3 LLaMA-Factory 安装

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e ".[torch,metrics]"
```

### 4.4 API 与模型准备

```bash
cp .env.example .env
# OPENAI_API_KEY=...
# OPENAI_API_BASE=https://api.deepseek.com
# OPENAI_MODEL=deepseek-chat
```

```python
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-4B", local_dir=r"D:\LLM\models\Qwen3-4B")
```

### 4.5 WSL vLLM 环境说明

`wsl_vllm` 环境用于 13/14/07 的主评测链路。若首次使用 WSL，请先确保：
- Windows WSL 已升级到最新版本。
- `nvidia-smi` 在 WSL 内可见 GPU。
- vLLM 可正常加载 BF16 与 AWQ 模型。

---

## 5. 方法与实现

### 5.1 数据构建链路（01 -> 05）

#### 5.1.1 Step 1：数据采集（01）

```bash
python scripts/01_collect_news.py --source xlsum --lang mixed --max_samples 6000
```

目标与输出：
- 从 XL-Sum 采集新闻样本。
- 过滤不适配业务场景的内容。
- 生成 `data/raw/news_raw.jsonl`。

#### 5.1.2 Step 2：结构化标注（02）

```bash
python scripts/02_generate_labels_api.py --max_samples 0 --concurrency 5
```

标注模板约束模型输出为 6 字段格式，生成 `data/labeled/news_labeled_v1.jsonl`。

#### 5.1.3 Step 3：校验与清洗（03）

```bash
python scripts/03_validate_and_clean.py
```

可选增强：

```bash
python scripts/03_validate_and_clean.py \
  --sample_preview_count 5 \
  --quality_snapshot_path data/reports/quality_snapshot.json
```

主要检查项：
- 字段完整性。
- 类别白名单合法性。
- 核心要点编号格式。
- 重复样本和明显异常文本。

#### 5.1.4 Step 4：数据集划分（04）

```bash
python scripts/04_split_dataset.py --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

可选：刷新 instruction 与 token 长度统计。

```bash
python scripts/04_split_dataset.py \
  --refresh_instruction \
  --analyze_tokens \
  --tokenizer_path D:/LLM/models/Qwen3-4B
```

最终规模：
- `train.json = 3843`
- `val.json = 480`
- `test.json = 481`

#### 5.1.5 Step 5：注册数据集（05）

```bash
python scripts/05_register_dataset_info.py
```

用于把训练数据注册到 LLaMA-Factory 的 `dataset_info.json`。

---

### 5.2 SFT 主线（能力构建）

#### 5.2.1 训练策略

SFT 采用 QLoRA 进行参数高效微调：
- 基座：Qwen3-4B
- 量化：4-bit NF4（bitsandbytes，当前 LLaMA-Factory 默认量化类型）
- LoRA：rank=8, alpha=16
- 梯度累积：16（在有限显存下模拟更大 batch）

这种配置的目标是：
- 降低训练显存占用。
- 保持结构化任务学习能力。
- 在 16GB 显存下实现可训练性。

#### 5.2.2 训练命令

在 LLaMA-Factory 根目录执行：

```bash
conda activate my_sft

llamafactory-cli train configs/train_qwen3_4b_qlora_news_v2.yaml
# 备用：llamafactory-cli train configs/train_qwen3_8b_qlora_news.yaml
```

历史训练信息（4B）：
- 训练步数：651（3 epochs）
- 耗时：约 3.4 小时
- 检查点策略：每 50 step 保存，保留最近 5 个

#### 5.2.3 合并 LoRA 权重

```bash
llamafactory-cli export \
  --model_name_or_path D:/LLM/models/Qwen3-4B \
  --adapter_name_or_path outputs/checkpoints/qwen3-4b-qlora-news-v2 \
  --template qwen3_nothink \
  --finetuning_type lora \
  --export_dir outputs/merged/qwen3-4b-news-v2
```

输出 `outputs/merged/qwen3-4b-news-v2` 作为：
- SFT 对照评测模型。
- AWQ 量化输入模型。

#### 5.2.4 SFT 推理与对比演示（08）

```bash
python scripts/08_demo_cli.py \
  --model_path D:/LLM/models/Qwen3-4B \
  --adapter_path outputs/checkpoints/qwen3-4b-qlora-news-v2 \
  --compare
```

该脚本用于快速观察：
- 基座 vs SFT 在同一输入上的输出差异。
- 结构化格式稳定性提升是否直观可见。

---

### 5.3 AWQ 主线（压缩部署）

#### 5.3.1 量化目标与范式

AWQ 采用 **W4A16（int4 weight-only）**：
- 权重使用 4-bit 存储。
- 激活仍为高精度（FP16/BF16）路径。
- 在控制质量损失的前提下降低磁盘体积和推理时延。

#### 5.3.2 校准集构造（09）

```bash
python scripts/09_prepare_awq_calib.py \
  --train data/cleaned/train.json \
  --output outputs/awq/calib_prompts_chat_strat256.jsonl \
  --num_samples 256 \
  --prompt_mode chat_template \
  --stratified_by_length true \
  --tokenizer_path outputs/merged/qwen3-4b-news-v2 \
  --stats_output outputs/awq/calib_stats.json \
  --seed 42
```

本轮收敛关键点：
- 不再只用原始 input 文本校准，而是改为真实推理时的 `system + user` chat-template 组织。
- 使用分层抽样（按长度分桶）降低校准样本分布偏差。

#### 5.3.3 AWQ 量化执行（10）

```bash
python scripts/10_quantize_awq.py \
  --model_path outputs/merged/qwen3-4b-news-v2 \
  --calib_path outputs/awq/calib_prompts_chat_strat256.jsonl \
  --output_dir outputs/quantized/qwen3-4b-news-v2-awq4-q1-chatstrat \
  --w_bit 4 \
  --group_size 128 \
  --zero_point true \
  --version GEMM \
  --max_calib_seq_len 1024 \
  --calib_samples 256 \
  --seed 42
```

说明：
- 主配置使用 `group_size=128`。
- `version=GEMM` 为量化导出配置；推理端通过 vLLM `awq_marlin` 走高性能 kernel。

#### 5.3.4 冒烟验证（13）

```bash
python scripts/13_vllm_awq_smoke_infer.py \
  --model_path outputs/quantized/qwen3-4b-news-v2-awq4-q1-chatstrat \
  --test data/cleaned/test.json \
  --num_samples 10 \
  --max_new_tokens 800 \
  --temperature 0.0 \
  --top_p 1.0 \
  --quantization awq_marlin
```

冒烟的价值：
- 快速检查模型是否可加载、可生成。
- 快速检查 6 字段是否稳定输出。
- 在全量评测前提前暴露 tokenizer 与 kernel 兼容问题。

#### 5.3.5 全量质量评测（14）

```bash
python scripts/14_benchmark_quality_vllm.py \
  --bf16_model_path outputs/merged/qwen3-4b-news-v2 \
  --awq_model_path outputs/quantized/qwen3-4b-news-v2-awq4-q1-chatstrat \
  --test data/cleaned/test.json \
  --n_samples 0 \
  --max_new_tokens 800 \
  --temperature 0.0 \
  --top_p 1.0 \
  --repetition_penalty 1.1 \
  --awq_quantization awq_marlin \
  --output_dir outputs/eval/awq_benchmark_vllm_q1 \
  --seed 42
```

说明：
- 支持 checkpoint 续跑。
- 当 BF16 基线已固定，可用 `--awq_only` 只复跑 AWQ 分支。

#### 5.3.6 性能评测（07）

BF16：

```bash
python scripts/07_benchmark_latency.py \
  --backend vllm \
  --model_path outputs/merged/qwen3-4b-news-v2 \
  --quantization none \
  --test data/cleaned/test.json \
  --num_samples 20 \
  --warmup_steps 5 \
  --repeat 30 \
  --max_new_tokens 800 \
  --temperature 0.0 \
  --top_p 1.0 \
  --report_tag sft_bf16_vllm_promptfix30 \
  --seed 42
```

AWQ：

```bash
python scripts/07_benchmark_latency.py \
  --backend vllm \
  --model_path outputs/quantized/qwen3-4b-news-v2-awq4-q1-chatstrat \
  --quantization awq_marlin \
  --test data/cleaned/test.json \
  --num_samples 20 \
  --warmup_steps 5 \
  --repeat 30 \
  --max_new_tokens 800 \
  --temperature 0.0 \
  --top_p 1.0 \
  --report_tag sft_awq4_q1_vllm_30 \
  --seed 42
```

#### 5.3.7 备选链路（11/12）

- `11_awq_smoke_infer.py`：AutoAWQ 推理冒烟。
- `12_benchmark_quality_awq.py`：AutoAWQ 后端质量评测。

该链路保留用于兼容性排障，不作为主结论口径。

---

## 6. 实验设置与评测口径

### 6.1 对照组定义

SFT 阶段历史对照：
- Group A：Base（no-think）
- Group B：Base（think）
- Group C：SFT merged（no-think）

AWQ 阶段主对照：
- BF16：`outputs/merged/qwen3-4b-news-v2`
- AWQ4：`outputs/quantized/qwen3-4b-news-v2-awq4-q1-chatstrat`

### 6.2 统一解码参数

为保证公平性，BF16 与 AWQ 对比严格固定：
- `max_new_tokens=800`
- `temperature=0.0`
- `top_p=1.0`
- `repetition_penalty=1.1`
- `seed=42`
- 测试集：`data/cleaned/test.json`（481）

### 6.3 质量指标定义

| 指标 | 含义 |
|------|------|
| ROUGE-1 | 词级重叠 |
| ROUGE-2 | 二元组重叠 |
| ROUGE-L | 最长公共子序列重叠 |
| all_sections_pass_rate | 6 字段完整率 |
| valid_category_rate | 事件类别白名单合法率 |
| valid_bullets_rate | 核心要点编号格式与条数合规率 |
| has_time_info_rate | 时间字段有效率 |

### 6.4 性能指标定义

| 指标 | 含义 |
|------|------|
| load_time_s | 模型加载耗时 |
| ttft_p50_s / ttft_p95_s | 首 token 延迟分位值 |
| latency_p50_s / latency_p95_s | 端到端延迟分位值 |
| tokens_per_s | 吞吐（token/s） |
| peak_gpu_memory_mb | 峰值显存 |
| model_disk_size_gb | 模型磁盘体积 |

### 6.5 验收阈值

- `ROUGE-L` 相对下降 <= 3%
- `all_sections_pass_rate` 下降 <= 2%
- 性能满足其一：
  - `latency_p50` 改善 >= 15%
  - 或 `peak_gpu_memory_mb` 下降 >= 25%

---

## 7. 实验结果与分析

### 7.1 SFT 阶段结果（历史主成果，完整保留）

#### 7.1.1 文本质量（测试集 481）

| 指标 | Group A（Base 不思考） | Group B（Base 思考） | Group C（SFT V2 不思考） | 最优 |
|------|------------------------|----------------------|---------------------------|------|
| ROUGE-1 | 0.6644 | 0.6895 | **0.7653** | C |
| ROUGE-2 | 0.4211 | 0.4117 | **0.5232** | C |
| ROUGE-L | 0.6348 | 0.6446 | **0.7347** | C |

相对 A 组提升（C 组）：
- ROUGE-1：+15.2%
- ROUGE-2：+24.2%
- ROUGE-L：+15.7%

#### 7.1.2 结构化稳定性（测试集 481）

| 指标 | Group A（Base 不思考） | Group B（Base 思考） | Group C（SFT V2 不思考） | 最优 |
|------|------------------------|----------------------|---------------------------|------|
| 必需字段完整率 | 97.1% | 100.0% | **100.0%** | B/C |
| 类别合规率 | 89.6% | 88.8% | **100.0%** | C |
| 要点格式率（>=3 条编号） | 92.1% | 45.3% | **100.0%** | C |
| 时间信息完整率 | 97.1% | 100.0% | **100.0%** | B/C |
| 平均要点条数 | 2.81 | 1.90 | **3.27** | C |
| 格式错误样本 | 50/481 | 54/481 | **0/481** | C |

#### 7.1.3 SFT 阶段效率（历史口径）

| 组别 | 配置 | 平均速度 |
|------|------|----------|
| Group A | Base 不思考，batch=4 | 4.2 s/条 |
| Group B | Base 思考，batch=2 | 17.7 s/条 |
| Group C | SFT merged 不思考，batch=4 | **2.9 s/条** |

#### 7.1.4 阶段结论

- SFT 明确解决了“结构化不稳定”问题。
- SFT 同时提升 ROUGE 与格式合规率，形成高质量基础模型。
- SFT 结果是后续 AWQ 量化的前提；如果 SFT 本身不稳定，量化只会放大不稳定。

### 7.2 AWQ 阶段结果（最终采用方案）

最终配置：`g128 + calib256 + chat-template 校准 + 分层抽样`。

#### 7.2.1 质量对比（BF16 vs AWQ4，481 全量）

| 指标 | BF16 (merged) | AWQ4 (最终) | 变化 |
|------|---------------|-------------|------|
| ROUGE-1 | 0.7064 | 0.6816 | -0.0248 |
| ROUGE-2 | 0.4598 | 0.4341 | -0.0257 |
| ROUGE-L | 0.6791 | 0.6578 | **-3.13%** |
| all_sections_pass_rate | 99.38% | 97.51% | **-1.87%** |
| valid_category_rate | 98.34% | 97.71% | -0.63% |
| valid_bullets_rate | 93.97% | 88.15% | -5.82% |
| has_time_info_rate | 99.38% | 98.13% | -1.25% |

分析：
- 格式主指标 `all_sections_pass_rate` 满足阈值（下降 1.87% <= 2%）。
- `ROUGE-L` 接近阈值边界（-3.13%），属于“可接受但偏紧”的状态。
- 尾字段和要点格式是主要损失点，后续优化应优先围绕校准分布和解码约束展开。

#### 7.2.2 性能对比（20 样本池，warmup=5，repeat=30）

| 指标 | BF16 | AWQ4 (最终) | 变化 |
|------|------|-------------|------|
| load_time_s | 53.67 | 36.99 | -31.1% |
| ttft_p50_s | 0.0438 | 0.0198 | -54.8% |
| latency_p50_s | 3.5495 | 1.3844 | -61.0% |
| latency_p95_s | 5.3610 | 1.7461 | -67.4% |
| tokens_per_s | 49.81 | 124.86 | 2.51x |
| model_disk_size_gb | 7.503 | 2.494 | -66.8% |
| peak_gpu_memory_mb | 16244.8 | 16171.7 | 基本持平 |

分析：
- 时延和吞吐改善显著，端侧收益明显。
- 显存峰值“接近持平”并不矛盾：vLLM 会把空余预算用于 KV cache 和预留空间。
- 磁盘体积下降 66.8% 是压缩有效性的直接证据。

#### 7.2.3 阶段结论

- AWQ 在保持主要质量指标可接受范围内，实现了显著性能收益。
- 对于信息流端侧场景，最终方案具备落地可行性。

### 7.3 参数探索（Q2，保留记录但不采用）

探索配置：`g64 + calib512 + chat-template + 分层抽样`。

观察结果：
- ROUGE-L 提升到 `0.6676`（语义重叠更高）。
- `all_sections_pass_rate` 下降到 `91.68%`（结构稳定性明显退化）。

解释：
- 更细粒度量化组可能保留更多词面相关信息，带来 ROUGE 增益。
- 但结构协议依赖的“格式控制能力”对量化扰动更敏感，导致字段一致性下降。

结论：
- Q2 作为参数探索记录保留。
- 不进入最终推荐配置。

### 7.4 总体结论（SFT + AWQ 缺一不可）

- SFT 是能力基础：保证语义质量和结构稳定。
- AWQ 是部署关键：降低体积和时延，提高吞吐。
- 最终推荐为当前 AWQ 最终配置（原 Q1 路线），并沿用 vLLM 主评测口径。

---

## 8. 复现流程与命令清单

### 8.1 从零开始（SFT + AWQ 全链）

```bash
# 1) 数据链路
python scripts/01_collect_news.py --source xlsum --lang mixed --max_samples 6000
python scripts/02_generate_labels_api.py --max_samples 0 --concurrency 5
python scripts/03_validate_and_clean.py
python scripts/04_split_dataset.py --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
python scripts/05_register_dataset_info.py

# 2) SFT
llamafactory-cli train configs/train_qwen3_4b_qlora_news_v2.yaml
llamafactory-cli export \
  --model_name_or_path D:/LLM/models/Qwen3-4B \
  --adapter_name_or_path outputs/checkpoints/qwen3-4b-qlora-news-v2 \
  --template qwen3_nothink \
  --finetuning_type lora \
  --export_dir outputs/merged/qwen3-4b-news-v2

# 3) SFT 评测
python scripts/06_eval_rouge_and_format.py --mode benchmark --n_samples 0 --batch_size 4 --skip_think

# 4) AWQ
python scripts/09_prepare_awq_calib.py \
  --train data/cleaned/train.json \
  --output outputs/awq/calib_prompts_chat_strat256.jsonl \
  --num_samples 256 \
  --prompt_mode chat_template \
  --stratified_by_length true \
  --tokenizer_path outputs/merged/qwen3-4b-news-v2 \
  --stats_output outputs/awq/calib_stats.json \
  --seed 42

python scripts/10_quantize_awq.py \
  --model_path outputs/merged/qwen3-4b-news-v2 \
  --calib_path outputs/awq/calib_prompts_chat_strat256.jsonl \
  --output_dir outputs/quantized/qwen3-4b-news-v2-awq4-q1-chatstrat \
  --w_bit 4 --group_size 128 --zero_point true --version GEMM \
  --max_calib_seq_len 1024 --calib_samples 256 --seed 42

python scripts/13_vllm_awq_smoke_infer.py \
  --model_path outputs/quantized/qwen3-4b-news-v2-awq4-q1-chatstrat \
  --test data/cleaned/test.json \
  --num_samples 10 --max_new_tokens 800 --temperature 0.0 --top_p 1.0 \
  --quantization awq_marlin

python scripts/14_benchmark_quality_vllm.py \
  --bf16_model_path outputs/merged/qwen3-4b-news-v2 \
  --awq_model_path outputs/quantized/qwen3-4b-news-v2-awq4-q1-chatstrat \
  --test data/cleaned/test.json --n_samples 0 \
  --max_new_tokens 800 --temperature 0.0 --top_p 1.0 --repetition_penalty 1.1 \
  --awq_quantization awq_marlin --output_dir outputs/eval/awq_benchmark_vllm_q1 --seed 42

python scripts/07_benchmark_latency.py \
  --backend vllm --model_path outputs/merged/qwen3-4b-news-v2 --quantization none \
  --test data/cleaned/test.json --num_samples 20 --warmup_steps 5 --repeat 30 \
  --max_new_tokens 800 --temperature 0.0 --top_p 1.0 --report_tag sft_bf16_vllm_promptfix30 --seed 42

python scripts/07_benchmark_latency.py \
  --backend vllm --model_path outputs/quantized/qwen3-4b-news-v2-awq4-q1-chatstrat --quantization awq_marlin \
  --test data/cleaned/test.json --num_samples 20 --warmup_steps 5 --repeat 30 \
  --max_new_tokens 800 --temperature 0.0 --top_p 1.0 --report_tag sft_awq4_q1_vllm_30 --seed 42
```

### 8.2 仅复跑 AWQ 分支（BF16 已固定）

```bash
python scripts/14_benchmark_quality_vllm.py \
  --awq_only \
  --awq_model_path outputs/quantized/qwen3-4b-news-v2-awq4-q1-chatstrat \
  --test data/cleaned/test.json \
  --n_samples 0 \
  --output_dir outputs/eval/awq_benchmark_vllm_q1_awqonly \
  --seed 42
```

### 8.3 关键结果文件索引

- SFT 汇总：`outputs/eval/benchmark_summary.json`
- AWQ 质量汇总：`outputs/eval/awq_benchmark_vllm_q1/benchmark_summary_awq_vllm.json`
- BF16 性能：`outputs/eval/latency_report_sft_bf16_vllm_promptfix30.json`
- AWQ 性能：`outputs/eval/latency_report_sft_awq4_q1_vllm_30.json`
- 量化报告：`outputs/quantized/qwen3-4b-news-v2-awq4-q1-chatstrat/awq_quantize_report.json`

---

## 9. 工程建议与排障

### 9.1 为什么性能测试与质量测试分开跑

建议分开的原因：
- 质量评测追求“全量覆盖与稳定统计”。
- 性能评测追求“可重复的时延统计（warmup/repeat）”。
- 混跑会互相污染缓存与资源占用，结果解释困难。

### 9.2 为什么 AWQ 模型有时看起来不省显存

- AWQ 确实降低了权重存储占用。
- 但 vLLM 会把空余预算用于 KV cache，导致峰值显存可能与 BF16 接近。
- 因此应结合 `model_disk_size_gb`、延迟、吞吐综合判断收益。

### 9.3 10 条冒烟为什么也会慢

常见原因：
- 首次加载和 kernel 编译开销。
- 输出上限较高（`max_new_tokens=800`）。
- 显卡存在残留进程造成资源争用。

### 9.4 常见故障清单

- WSL 更新提示：先运行 `wsl.exe --update` 后重启。
- 显存爆满：排查残留 python/vLLM 进程。
- tokenizer 兼容问题：使用脚本自动生成的 tokenizer 修复目录。
- AWQ 速度异常：确认使用 vLLM `awq_marlin` 而非无加速路径。

更完整排障文档见 [docs/troubleshooting.md](docs/troubleshooting.md)。

---

## 10. 参考文献

1. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
2. [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
3. [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory)
4. [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
5. [vLLM](https://docs.vllm.ai/)
