#!/usr/bin/env python3
"""
11_awq_smoke_infer.py
对 AWQ 量化模型做 5-10 条冒烟推理，验证可加载、可生成、格式稳定。

示例：
  python scripts/11_awq_smoke_infer.py \
    --model_path outputs/quantized/qwen3-4b-news-v2-awq4 \
    --num_samples 10
"""

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
EVAL_DIR = PROJECT_DIR / "outputs" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TEST = PROJECT_DIR / "data" / "cleaned" / "test.json"
DEFAULT_OUTPUT = EVAL_DIR / "awq_smoke_predictions.jsonl"
DEFAULT_REPORT = EVAL_DIR / "awq_smoke_report.json"

SYSTEM_PROMPT = (
    "你是专业的新闻编辑助手。请对新闻内容进行结构化摘要，严格按以下6个标签顺序输出，禁止使用 Markdown：\n"
    "【一句话摘要】【核心要点】【事件类别】【主要主体】【时间信息】【潜在影响】\n"
    "其中【核心要点】用阿拉伯数字编号列出至少3条。"
)

REQUIRED_SECTIONS = [
    "【一句话摘要】",
    "【核心要点】",
    "【事件类别】",
    "【主要主体】",
    "【时间信息】",
    "【潜在影响】",
]


def load_samples(path: Path, num_samples: int) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    if num_samples <= 0:
        return data
    return data[:num_samples]


def has_all_sections(text: str) -> bool:
    return all(tag in text for tag in REQUIRED_SECTIONS)


def clean_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def build_input_ids(tokenizer, prompt: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )
    except Exception:
        return tokenizer(f"{SYSTEM_PROMPT}\n\n{prompt}", return_tensors="pt").input_ids


def normalize_model_inputs(raw_inputs):
    """统一为 generate 可接受的 kwargs 输入。"""
    if isinstance(raw_inputs, dict):
        result = {}
        if "input_ids" in raw_inputs:
            result["input_ids"] = raw_inputs["input_ids"]
        if "attention_mask" in raw_inputs:
            result["attention_mask"] = raw_inputs["attention_mask"]
        return result

    input_ids = getattr(raw_inputs, "input_ids", None)
    attention_mask = getattr(raw_inputs, "attention_mask", None)
    if input_ids is not None:
        out = {"input_ids": input_ids}
        if attention_mask is not None:
            out["attention_mask"] = attention_mask
        return out

    return {"input_ids": raw_inputs}


def main() -> None:
    parser = argparse.ArgumentParser(description="AWQ 模型冒烟推理")
    parser.add_argument("--model_path", type=str, required=True, help="AWQ 量化模型目录")
    parser.add_argument("--test", type=str, default=str(DEFAULT_TEST), help="测试集路径")
    parser.add_argument("--num_samples", type=int, default=10, help="冒烟样本数")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成 token")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="预测输出 JSONL")
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT), help="报告输出 JSON")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    test_path = Path(args.test)
    if not model_path.exists():
        print(f"[ERROR] 模型目录不存在: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not test_path.exists():
        print(f"[ERROR] 测试集不存在: {test_path}", file=sys.stderr)
        sys.exit(1)

    try:
        import torch
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        print("[ERROR] 缺少依赖：pip install autoawq transformers torch", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 加载 AWQ 模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    kwargs_candidates = [
        {
            "quant_path": str(model_path),
            "trust_remote_code": True,
            "fuse_layers": True,
            "device_map": "auto",
            "safetensors": True,
        },
        {
            "quant_path": str(model_path),
            "trust_remote_code": True,
            "fuse_layers": False,
            "device_map": "auto",
            "safetensors": True,
        },
        {
            "quant_path": str(model_path),
            "trust_remote_code": True,
            "device_map": "auto",
        },
    ]

    model = None
    last_err = None
    for kwargs in kwargs_candidates:
        try:
            model = AutoAWQForCausalLM.from_quantized(**kwargs)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc

    if model is None:
        print(f"[ERROR] AWQ 模型加载失败: {last_err}", file=sys.stderr)
        sys.exit(1)
    model.eval()

    samples = load_samples(test_path, args.num_samples)
    if not samples:
        print("[ERROR] 没有可用样本", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    pass_flags = []  # type: List[bool]
    latencies = []  # type: List[float]
    rows = []

    for i, sample in enumerate(samples, start=1):
        prompt = str(sample.get("input", "")).strip()
        if not prompt:
            continue
        raw_inputs = build_input_ids(tokenizer, prompt)
        model_inputs = normalize_model_inputs(raw_inputs)
        try:
            device = next(model.parameters()).device
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        except Exception:
            pass

        t0 = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency_s = time.perf_counter() - t0
        latencies.append(latency_s)

        input_len = int(model_inputs["input_ids"].shape[-1])
        output_text = tokenizer.decode(generated[0][input_len:], skip_special_tokens=True)
        output_text = clean_think(output_text)
        passed = has_all_sections(output_text)
        pass_flags.append(passed)

        row = {
            "index": i - 1,
            "input": prompt,
            "reference": sample.get("output", ""),
            "prediction": output_text,
            "all_sections_pass": passed,
            "latency_s": latency_s,
        }
        rows.append(row)
        print(f"[{i:2d}/{len(samples)}] pass={passed} latency={latency_s:.3f}s")

    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(rows)
    pass_rate = (sum(pass_flags) / total) if total else 0.0
    report = {
        "model_path": str(model_path),
        "num_samples": total,
        "all_sections_pass_rate": pass_rate,
        "avg_latency_s": (statistics.mean(latencies) if latencies else 0.0),
        "p50_latency_s": (statistics.median(latencies) if latencies else 0.0),
        "p95_latency_s": (sorted(latencies)[min(int(len(latencies) * 0.95), len(latencies) - 1)] if latencies else 0.0),
        "threshold_pass": pass_rate >= 0.90,
        "threshold": "all_sections_pass_rate >= 0.90",
        "prediction_file": str(output_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== AWQ 冒烟结果 =====")
    print(f"样本数:       {total}")
    print(f"字段完整率:   {pass_rate:.2%}")
    print(f"平均耗时:     {report['avg_latency_s']:.3f}s")
    print(f"阈值通过:     {report['threshold_pass']}")
    print(f"预测文件:     {output_path}")
    print(f"报告文件:     {report_path}")


if __name__ == "__main__":
    main()
