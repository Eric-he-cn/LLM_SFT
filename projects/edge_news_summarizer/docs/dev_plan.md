# Edge-News-Summarizer-FT 开发计划

## 项目概述

基于 Qwen + LLaMA-Factory 的新闻结构化摘要微调项目（QLoRA），目标在本地消费级 GPU（RTX 5060 Ti 16GB）上完成端到端的 SFT 流程。

---

## 技术路线

```
原始新闻 → API 打标（GPT-4o-mini）→ 清洗 → Alpaca 格式
    ↓
LLaMA-Factory QLoRA 微调（Qwen2.5-3B/7B）
    ↓
ROUGE + 格式正确率评测 + CLI Demo
```

---

## 里程碑

### M1：数据管线（第 1-3 天）

**目标**：能产出可用的训练数据集

- [ ] 收集 5000+ 条新闻（CNN/DailyMail 或本地数据）
- [ ] 调用 API 生成 2000+ 条结构化标签
- [ ] 清洗后得到 1500+ 条高质量数据
- [ ] 划分 train/val/test（8:1:1）

**验收标准**：
- `data/cleaned/train.json` 存在且有效
- 格式校验通过率 > 90%

### M2：训练跑通（第 3-5 天）

**目标**：3B QLoRA 能完成 1 个 epoch，输出可读摘要

- [ ] 注册数据集到 LLaMA-Factory
- [ ] 运行 3B QLoRA 训练（`train_qwen25_3b_qlora_news.yaml`）
- [ ] 保存 LoRA adapter
- [ ] 用 CLI demo 验证输出质量

**验收标准**：
- 训练 loss 能降到 1.5 以下
- 模型能输出包含 6 个字段的结构化摘要

### M3：评测与对比（第 5-7 天）

**目标**：量化微调效果，与基座模型对比

- [ ] 运行 ROUGE 评测（Base vs FT）
- [ ] 格式正确率对比
- [ ] 生成至少 20 条 bad case 分析
- [ ] 延迟评测（P50/P95）

**验收标准**：
- FT 模型格式正确率 > 90%（Base 通常 < 60%）
- ROUGE-L 相对提升 > 10%

### M4：Demo 与文档（第 7 天+）

**目标**：可演示、可展示

- [ ] CLI demo 可交互运行
- [ ] README 包含完整安装/训练/评测步骤
- [ ] 包含 Base vs FT 输出对比示例

---

## 数据规格

| 字段 | 规格 |
|------|------|
| 训练集 | 1200+ 条 |
| 验证集 | 150+ 条 |
| 测试集 | 150+ 条 |
| 输入长度 | 50-4000 字符 |
| 输出长度 | 100-2000 字符 |

---

## 资源需求

| 资源 | 需求 |
|------|------|
| GPU 显存 | 16GB（RTX 5060 Ti） |
| 磁盘空间 | 80GB（模型 + 数据 + checkpoints） |
| API 费用估算 | ~$5-15（2000 条 × GPT-4o-mini） |
| 训练时间（3B, 3 epoch） | ~2-4 小时 |

---

## 关键配置参数

### 3B QLoRA（推荐首选）

```yaml
lora_rank: 8
lora_alpha: 16
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2.0e-4
cutoff_len: 1024
```

### 7B QLoRA（显存 16GB 边界）

```yaml
lora_rank: 16
lora_alpha: 32
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 1.0e-4
cutoff_len: 1024
```

---

## 风险与应对

| 风险 | 概率 | 应对方案 |
|------|------|----------|
| bitsandbytes 在 Windows 不兼容 | 中 | 改用 WSL2 跑训练 |
| API 标注格式合格率低 | 低 | 换更强模型（gpt-4o） |
| 7B 模型 OOM | 中 | 先用 3B 跑通，再升级 |
| CNN/DM 数据集下载慢 | 中 | 设置 HF 镜像或用本地数据 |
