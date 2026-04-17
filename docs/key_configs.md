# 项目关键配置总览（面试速查版）

> 更新时间：2026-04-17

本文档汇总项目中最常被面试官追问的关键参数，覆盖模型架构、数据集统计、QLoRA SFT、AWQ 量化、推理解码、评测口径与最终实验结果。

适用项目：
- `Qwen3-4B + QLoRA SFT + AWQ`
- 目标任务：端侧信息流场景的 6 字段结构化新闻摘要

---

## 1. 一句话总览

- 基座模型：`Qwen3-4B`
- SFT 方法：`QLoRA (4-bit NF4) + LoRA`
- 量化方法：`AWQ, W4A16 (int4 weight-only)`
- 主推理后端：`vLLM`
- 最终推荐量化配置：`g128 + calib256 + chat-template + 分层抽样`

---

## 2. 模型架构参数

| 项目 | 数值 |
|------|------|
| 模型类型 | `Qwen3ForCausalLM` |
| 基座模型 | `Qwen3-4B` |
| `d_model / hidden_size` | `2560` |
| Transformer 层数 | `36` |
| Attention heads | `32` |
| KV heads | `8` |
| `head_dim` | `128` |
| `intermediate_size` | `9728` |
| 词表大小 `vocab_size` | `151936` |
| 最大位置长度 | `40960` |
| 默认 dtype | `bfloat16` |
| 词嵌入类型 | 模型内置可学习 Token Embedding |

补充说明：
- 这个项目没有单独训练外部词向量模型，词嵌入直接来自 Qwen3-4B 自带 embedding 层。
- `d_model` 在面试中可直接回答为 `2560`。

---

## 3. 数据集与数据统计

### 3.1 数据流程

```text
XL-Sum -> DeepSeek API 标注 -> 格式校验/清洗 -> train/val/test 划分
```

### 3.2 样本数量

| 项目 | 数量 |
|------|------|
| 原始标注后样本 | `4806` |
| 清洗通过样本 | `4804` |
| 去重丢弃 | `2` |
| 训练集 | `3843` |
| 验证集 | `480` |
| 测试集 | `481` |

### 3.3 长度统计

单位说明：
- `input/output` 长度为字符数统计
- AWQ 校准长度为 chat-template token 统计

| 数据集 | input 平均长度 | input P50 | input P95 | output 平均长度 |
|--------|----------------|-----------|-----------|-----------------|
| `cleaned_all` | `1735.72` | `1835` | `3072` | `358.93` |
| `train` | `1723.70` | `1772` | `3072` | `358.71` |
| `val` | `1820.20` | `2197` | `3074` | `361.10` |
| `test` | `1747.47` | `1855` | `3073` | `358.46` |

### 3.4 标签与类别约束

- 输出协议固定为 6 个字段：
  - `【一句话摘要】`
  - `【核心要点】`
  - `【事件类别】`
  - `【主要主体】`
  - `【时间信息】`
  - `【潜在影响】`
- 类别白名单共 `23` 个合法值
- 核心中文类别主要为：
  - `政治、经济、科技、文化、社会、军事、体育、健康、环境、国际、历史、旅游、财经`
- 同时兼容若干英文类别字符串，避免模型中英混输时被误判为错误

---

## 4. SFT（QLoRA）训练配置

### 4.1 训练目标

- 任务：学习新闻结构化摘要协议
- 重点：提升 ROUGE、类别合规率、字段完整率、要点编号稳定性
- 模板：`qwen3_nothink`
- `enable_thinking=false`

### 4.2 QLoRA / LoRA 配置

| 项目 | 数值 |
|------|------|
| `stage` | `sft` |
| `finetuning_type` | `lora` |
| `lora_target` | `all` |
| `quantization_bit` | `4` |
| `quantization_method` | `bitsandbytes` |
| `lora_rank` | `8` |
| `lora_alpha` | `16` |
| `lora_dropout` | `0.05` |

### 4.3 训练超参数

| 项目 | 数值 |
|------|------|
| `cutoff_len` | `1024` |
| `max_samples` | `10000` |
| `per_device_train_batch_size` | `1` |
| `gradient_accumulation_steps` | `16` |
| 有效 batch size | 约 `16` |
| `learning_rate` | `2e-4` |
| `num_train_epochs` | `3.0` |
| `lr_scheduler_type` | `cosine` |
| `warmup_ratio` | `0.1` |
| `bf16` | `true` |
| `fp16` | `false` |
| `logging_steps` | `10` |
| `save_steps` | `50` |
| `save_total_limit` | `5` |
| `per_device_eval_batch_size` | `1` |
| `eval_strategy` | `steps` |
| `eval_steps` | `50` |

### 4.4 训练结果

| 项目 | 数值 |
|------|------|
| 总步数 | `651` |
| epoch | `3.0` |
| 最终 `train_loss` | `0.9530` |
| 最终 `eval_loss` | `0.9662` |
| 困惑度 `exp(eval_loss)` | 约 `2.63` |
| 训练时长 | `12293.6s`（约 `3h25m`） |

补充说明：
- 训练时使用 `batch=1 + grad_accum=16`，核心目的是在 16GB 显存下稳定训练。
- 训练后导出 merged 模型，作为 AWQ 量化输入。

---

## 5. SFT 推理与历史对照口径

### 5.1 历史 A/B/C 三组定义

| 组别 | 含义 |
|------|------|
| Group A | Base，不思考 |
| Group B | Base，思考 |
| Group C | SFT merged，不思考 |

### 5.2 历史推理配置

#### SFT 侧配置

| 项目 | 数值 |
|------|------|
| 模板 | `qwen3_nothink` |
| `cutoff_len` | `1024` |
| `per_device_eval_batch_size` | `4` |
| `max_new_tokens` | `512` |
| `temperature` | `0.1` |
| `top_p` | `0.9` |
| `bf16` | `true` |

#### Base 侧配置

| 项目 | 数值 |
|------|------|
| 模板 | `qwen3` |
| `enable_thinking` | `true` |
| `cutoff_len` | `4096` |
| `per_device_eval_batch_size` | `4` |
| `max_new_tokens` | `2048` |
| `temperature` | `0.1` |
| `top_p` | `0.9` |
| `repetition_penalty` | `1.1` |

补充说明：
- 早期 Base 推理之所以保留 thinking，是因为 Qwen3 基座模型会自发生成 think 块。
- 后续结构化任务主线统一关闭 thinking，以减少冗余推理链和时延开销。

---

## 6. AWQ 量化配置

### 6.1 量化目标

- 输入模型：`outputs/merged/qwen3-4b-news-v2`
- 量化方式：`AWQ`
- 量化范式：`W4A16`
- 含义：权重 4bit，激活保持高精度路径

### 6.2 最终采用方案（推荐）

| 项目 | 数值 |
|------|------|
| `w_bit` | `4` |
| `group_size` | `128` |
| `zero_point` | `true` |
| `version` | `GEMM` |
| `max_calib_seq_len` | `1024` |
| `calib_samples` | `256` |
| `seed` | `42` |
| 校准 prompt 组织方式 | `chat_template(system+user)` |
| 抽样方式 | `分层抽样（按长度分桶）` |

### 6.3 校准集统计（最终方案）

| 项目 | 数值 |
|------|------|
| 候选样本数 | `3843` |
| 选中样本数 | `256` |
| 候选 token 平均长度 | `797.21` |
| 候选 token P50 | `734` |
| 候选 token P95 | `2056` |
| 选中 token 平均长度 | `793.41` |
| 选中 token P50 | `727` |
| 选中 token P95 | `2044` |

分桶阈值与抽样分布：
- `short_max=284`
- `medium_max=805`
- 候选数量：`short=1284`, `medium=1279`, `long=1280`
- 选中数量：`short=86`, `medium=85`, `long=85`

### 6.4 Q2 探索方案（未采用）

| 项目 | 数值 |
|------|------|
| `w_bit` | `4` |
| `group_size` | `64` |
| `calib_samples` | `512` |
| `max_calib_seq_len` | `1024` |
| prompt 组织方式 | `chat_template` |
| 抽样方式 | `分层抽样` |

Q2 结论：
- ROUGE 更高一点
- 但格式合规率明显变差
- 因此不采用

---

## 7. vLLM 推理与统一解码参数

### 7.1 主评测统一口径

| 项目 | 数值 |
|------|------|
| 后端 | `vLLM` |
| `max_new_tokens` | `800` |
| `temperature` | `0.0` |
| `top_p` | `1.0` |
| `repetition_penalty` | `1.1` |
| `seed` | `42` |
| 测试集 | `481` 条 |

### 7.2 性能压测设置

| 项目 | 数值 |
|------|------|
| 样本池大小 | `20` |
| `warmup_steps` | `5` |
| `repeat` | `30` |
| BF16 量化模式 | `none` |
| AWQ 量化模式 | `awq_marlin` |

### 7.3 vLLM AWQ 冒烟设置

| 项目 | 数值 |
|------|------|
| 冒烟样本数 | `5~10` |
| `gpu_memory_utilization` | `0.85` |
| `max_model_len` | `4096` |
| 判定阈值 | `all_sections_pass_rate >= 0.90` |

---

## 8. 评测指标与判定规则

### 8.1 质量指标

| 指标 | 含义 |
|------|------|
| `ROUGE-1` | 单词级重叠 |
| `ROUGE-2` | Bigram 重叠 |
| `ROUGE-L` | 最长公共子序列 |
| `all_sections_pass_rate` | 6 字段完整率 |
| `valid_category_rate` | 类别白名单合法率 |
| `valid_bullets_rate` | 核心要点编号合规率（>=3条） |
| `has_time_info_rate` | 时间字段有效率 |
| `empty_output_rate` | 空输出比例 |

### 8.2 bad case 判定规则

任一条件触发即判为 bad case：
- 缺少任意必需字段
- `【事件类别】` 不在白名单
- `【核心要点】` 编号条数小于 `3`

### 8.3 性能指标

| 指标 | 含义 |
|------|------|
| `load_time_s` | 模型加载耗时 |
| `ttft_p50_s / ttft_p95_s` | 首 token 延迟分位值 |
| `latency_p50_s / latency_p95_s` | 端到端时延分位值 |
| `tokens_per_s` | 吞吐 |
| `peak_gpu_memory_mb` | 峰值显存 |
| `model_disk_size_gb` | 模型磁盘体积 |

### 8.4 验收阈值

- `ROUGE-L` 相对下降不超过 `3%`
- `all_sections_pass_rate` 下降不超过 `2%`
- 性能满足其一：
  - `latency_p50` 改善 >= `15%`
  - 或 `peak_gpu_memory_mb` 下降 >= `25%`

---

## 9. 核心实验结果

### 9.1 SFT 阶段（Base vs SFT）

| 指标 | Group A Base | Group B Base+Think | Group C SFT |
|------|--------------|--------------------|-------------|
| ROUGE-1 | `0.6644` | `0.6895` | `0.7653` |
| ROUGE-2 | `0.4211` | `0.4117` | `0.5232` |
| ROUGE-L | `0.6348` | `0.6446` | `0.7347` |
| 字段完整率 | `97.09%` | `100.0%` | `100.0%` |
| 类别合规率 | `89.6%` | `88.77%` | `100.0%` |
| 要点格式率 | `92.1%` | `45.32%` | `100.0%` |
| 时间信息率 | `97.09%` | `100.0%` | `100.0%` |
| bad cases | `50` | `54` | `0` |

SFT 相对 Base（A 组）提升：
- ROUGE-1：`+15.2%`
- ROUGE-2：`+24.2%`
- ROUGE-L：`+15.7%`

### 9.2 最终 AWQ 方案（BF16 vs AWQ4-Q1）

#### 质量（481 全量）

| 指标 | BF16 | AWQ4-Q1 | 变化 |
|------|------|---------|------|
| ROUGE-1 | `0.7064` | `0.6816` | `-0.0248` |
| ROUGE-2 | `0.4598` | `0.4341` | `-0.0257` |
| ROUGE-L | `0.6791` | `0.6578` | `-3.13%` |
| all_sections_pass_rate | `99.38%` | `97.51%` | `-1.87%` |
| valid_category_rate | `98.34%` | `97.71%` | `-0.63%` |
| valid_bullets_rate | `93.97%` | `88.15%` | `-5.82%` |
| has_time_info_rate | `99.38%` | `98.13%` | `-1.25%` |

#### 性能（warmup=5, repeat=30）

| 指标 | BF16 | AWQ4-Q1 | 变化 |
|------|------|---------|------|
| load_time_s | `53.67` | `36.99` | `-31.1%` |
| ttft_p50_s | `0.0438` | `0.0198` | `-54.8%` |
| latency_p50_s | `3.5495` | `1.3844` | `-61.0%` |
| latency_p95_s | `5.3610` | `1.7461` | `-67.4%` |
| tokens_per_s | `49.81` | `124.86` | `2.51x` |
| model_disk_size_gb | `7.503` | `2.494` | `-66.8%` |
| peak_gpu_memory_mb | `16244.8` | `16171.7` | 基本持平 |

### 9.3 Q2 探索结果

| 指标 | AWQ4-Q2 |
|------|---------|
| ROUGE-1 | `0.6942` |
| ROUGE-2 | `0.4460` |
| ROUGE-L | `0.6676` |
| all_sections_pass_rate | `91.68%` |
| valid_category_rate | `96.88%` |
| valid_bullets_rate | `94.39%` |
| has_time_info_rate | `91.68%` |
| latency_p50_s | `1.4881` |
| tokens_per_s | `126.33` |

最终结论：
- `Q1` 是最终采用方案
- `Q2` 虽然 ROUGE 略高，但格式稳定性明显下降

---

## 10. 面试时可直接说的结论

- 这个项目不是只做微调，也不是只做量化，而是 `SFT + AWQ` 两阶段闭环。
- SFT 负责把结构化摘要任务学稳，AWQ 负责把模型压到端侧可部署。
- SFT 后模型把 ROUGE-L 提升到了 `0.7347`，同时把类别合规率和要点格式率都拉到了 `100%`。
- 在最终 AWQ 配置下，模型体积从 `7.5GB` 压到 `2.49GB`，`P50` 时延下降约 `61%`，吞吐提升到 `2.5x`。
- 量化不是盲目压缩，我们专门做了 `chat-template` 校准和长度分层抽样，最后选的是 `group_size=128, calib=256` 这条更稳的路线。

---

## 11. 重要文件索引

- 训练配置：`configs/train_qwen3_4b_qlora_news_v2.yaml`
- SFT 结果汇总：`outputs/eval/benchmark_summary.json`
- AWQ 最终质量汇总：`outputs/eval/awq_benchmark_vllm_q1/benchmark_summary_awq_vllm.json`
- BF16 性能报告：`outputs/eval/latency_report_sft_bf16_vllm_promptfix30.json`
- AWQ Q1 性能报告：`outputs/eval/latency_report_sft_awq4_q1_vllm_30.json`
- AWQ Q2 性能报告：`outputs/eval/latency_report_sft_awq4_q2_vllm_30.json`
- Q1 量化报告：`outputs/quantized/qwen3-4b-news-v2-awq4-q1-chatstrat/awq_quantize_report.json`
- 模型结构配置：`outputs/merged/qwen3-4b-news-v2/config.json`
