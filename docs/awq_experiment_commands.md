# AWQ 实验命令清单（vLLM 主链 + AutoAWQ 备选）

本清单用于复现 `SFT merged(BF16)` vs `SFT merged AWQ4`。
统一口径：`seed=42`，`max_new_tokens=800`，`temperature=0.0`，`top_p=1.0`。

---

## A. 主链（推荐）vLLM

### A1) 准备校准集（chat-template + 分层抽样）

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

### A2) 量化（W4A16）

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

### A3) vLLM 冒烟（10 条）

```bash
python scripts/13_vllm_awq_smoke_infer.py \
  --model_path outputs/quantized/qwen3-4b-news-v2-awq4-q1-chatstrat \
  --test data/cleaned/test.json \
  --num_samples 10 \
  --max_new_tokens 800 \
  --quantization awq_marlin
```

### A4) vLLM 全量质量（481）

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

仅复跑 AWQ 分支时：

```bash
python scripts/14_benchmark_quality_vllm.py \
  --awq_only \
  --awq_model_path outputs/quantized/qwen3-4b-news-v2-awq4-q1-chatstrat \
  --test data/cleaned/test.json \
  --n_samples 0 \
  --output_dir outputs/eval/awq_benchmark_vllm_q1_awqonly \
  --seed 42
```

### A5) vLLM 性能（30 次）

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

---

## B. 备选链（兼容）AutoAWQ 推理

仅在 vLLM 不可用或排障时使用：

```bash
python scripts/11_awq_smoke_infer.py \
  --model_path outputs/quantized/qwen3-4b-news-v2-awq4-q1-chatstrat \
  --test data/cleaned/test.json \
  --num_samples 10 \
  --max_new_tokens 800

python scripts/12_benchmark_quality_awq.py \
  --bf16_model_path outputs/merged/qwen3-4b-news-v2 \
  --awq_model_path outputs/quantized/qwen3-4b-news-v2-awq4-q1-chatstrat \
  --test data/cleaned/test.json \
  --n_samples 0 \
  --max_new_tokens 800 \
  --output_dir outputs/eval/awq_benchmark_awqbackend \
  --seed 42
```

---

## C. 核心结果文件

- 量化报告：`outputs/quantized/qwen3-4b-news-v2-awq4-q1-chatstrat/awq_quantize_report.json`
- 质量汇总（主链）：`outputs/eval/awq_benchmark_vllm_q1/benchmark_summary_awq_vllm.json`
- 性能汇总（BF16）：`outputs/eval/latency_report_sft_bf16_vllm_promptfix30.json`
- 性能汇总（AWQ）：`outputs/eval/latency_report_sft_awq4_q1_vllm_30.json`

参数探索（Q2）结果可参考：`outputs/eval/awq_benchmark_vllm_q2_awqonly/`。
