# 常见问题排查指南（Qwen3-4B + QLoRA + AWQ + vLLM）

## 1. 环境与安装

### 1.1 CUDA 不可用

症状：`torch.cuda.is_available() == False`

处理：

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"
```

### 1.2 AutoAWQ/Transformers 版本兼容

症状：量化或加载报错（kernel / API 变更）。

处理：
- 固定同一套环境复现（建议 `my_sft`）。
- 先运行 `10_quantize_awq.py` + `13_vllm_awq_smoke_infer.py` 做链路验证。

---

## 2. 量化与推理

### 2.1 GEMM 内核不可用

症状：`GEMM kernel not found` 或类似错误。

处理：
- `10_quantize_awq.py` 会自动尝试回退到 GEMV。
- 检查量化报告中的 `used_version` 与 `fallback_to_gemv`。

### 2.2 tokenizer_config 兼容问题

症状：vLLM/HF 加载 tokenizer 失败（`extra_special_tokens` 结构不兼容）。

处理：
- 使用脚本自动生成 `*_tokenizerfix` 目录。
- 在报告里确认 `tokenizer_fix_applied=true`。

### 2.3 WSL 下显卡被占满（中断后）

症状：任务已中断但显存持续高占用。

处理：

```bash
# Windows
nvidia-smi

# WSL 内查看残留
ps -eo pid,cmd | grep -E 'python|vllm' | grep -v grep

# 杀掉残留 PID
kill -9 <pid>
```

说明：vLLM 中断后若进程未退出，会持续占用 GPU。

---

## 3. 评测口径与脚本

### 3.1 为什么质量和性能要分开跑

- 质量评测关注输出文本稳定性，通常跑全量（481）。
- 性能评测关注统计稳定性，使用固定样本池 + warmup/repeat。
- 混跑会让指标互相污染，不利于可复现。

### 3.2 只想复跑 AWQ，不重跑 BF16

```bash
python scripts/14_benchmark_quality_vllm.py \
  --awq_only \
  --awq_model_path <awq_model_path> \
  --test data/cleaned/test.json \
  --n_samples 0
```

### 3.3 为什么 peak GPU memory 变化不明显

AWQ 降低的是权重体积；在 vLLM 下，释放出来的显存常被 KV cache 预留占用，因此峰值显存可能接近持平。

---

## 4. 格式合规率异常下降

优先检查：
- 是否出现替代字段（如 `【一句话总结】`）
- 尾字段是否缺失（`【主要主体】/【时间信息】/【潜在影响】`）
- 是否触发 `max_new_tokens` 截断

建议：
- 先做 10 条冒烟，再跑全量。
- 统一系统提示词、解码参数、seed，再比较 AWQ/BF16。

---

## 5. 主链与备选链

- 主链：`09 -> 10 -> 13 -> 14 -> 07(vllm)`
- 备选链：`11/12`（AutoAWQ 推理）

遇到主链故障时，先用备选链定位问题，再回到主链复测。
