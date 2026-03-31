#!/usr/bin/env python3
"""
13_vllm_awq_smoke_infer.py
使用 vLLM 加载 AWQ 量化模型做 5-10 条冒烟推理，验证可加载、可生成、格式稳定。

示例：
  python scripts/13_vllm_awq_smoke_infer.py \
    --model_path /mnt/d/LLM/Qwen3-QLoRA-News/outputs/quantized/qwen3-4b-news-v2-awq4 \
    --num_samples 5
"""

import argparse
import json
import re
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
EVAL_DIR = PROJECT_DIR / "outputs" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TEST = PROJECT_DIR / "data" / "cleaned" / "test.json"
DEFAULT_OUTPUT = EVAL_DIR / "vllm_awq_smoke_predictions.jsonl"
DEFAULT_REPORT = EVAL_DIR / "vllm_awq_smoke_report.json"

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


def build_prompt(tokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return f"{SYSTEM_PROMPT}\n\n{user_prompt}"


def p95(values: List[float]) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    idx = min(int(len(values_sorted) * 0.95), len(values_sorted) - 1)
    return values_sorted[idx]


def maybe_prepare_tokenizer_fix(model_path: Path, tokenizer_path: Path) -> Path:
    """兼容 transformers 对 extra_special_tokens=list 的报错，生成一次修复副本。"""
    cfg_path = tokenizer_path / "tokenizer_config.json"
    if not cfg_path.exists():
        return tokenizer_path
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return tokenizer_path

    extra = cfg.get("extra_special_tokens")
    if not isinstance(extra, list):
        return tokenizer_path

    fixed_dir = model_path.parent / f"{model_path.name}_tokenizerfix"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"]:
        src = tokenizer_path / name
        if src.exists():
            shutil.copy2(src, fixed_dir / name)

    fixed_cfg_path = fixed_dir / "tokenizer_config.json"
    fixed_cfg = json.loads(fixed_cfg_path.read_text(encoding="utf-8"))
    fixed_map = {}
    for i, tok in enumerate(extra):
        key = re.sub(r"[^0-9a-zA-Z_]+", "_", str(tok).strip("<>|")).strip("_").lower()
        if not key:
            key = f"tok_{i}"
        if key in fixed_map:
            key = f"{key}_{i}"
        fixed_map[key] = tok
    fixed_cfg["extra_special_tokens"] = fixed_map
    fixed_cfg["tokenizer_class"] = "Qwen2TokenizerFast"
    fixed_cfg_path.write_text(json.dumps(fixed_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] 检测到 tokenizer_config 兼容问题，已生成修复目录: {fixed_dir}")
    return fixed_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM + AWQ 冒烟推理")
    parser.add_argument("--model_path", type=str, required=True, help="AWQ 量化模型目录")
    parser.add_argument("--test", type=str, default=str(DEFAULT_TEST), help="测试集路径")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="可选 tokenizer 路径（用于兼容修复目录）")
    parser.add_argument("--num_samples", type=int, default=5, help="冒烟样本数（建议 5-10）")
    parser.add_argument("--max_new_tokens", type=int, default=800, help="最大生成 token（建议与 BF16 保持一致）")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="top_p")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="vLLM 显存占用比例")
    parser.add_argument("--max_model_len", type=int, default=4096, help="最大上下文长度")
    parser.add_argument(
        "--tokenizer_mode",
        type=str,
        default="auto",
        choices=["auto", "slow"],
        help="tokenizer 模式，Qwen3 在部分版本下建议 slow",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="awq_marlin",
        choices=["awq", "awq_marlin"],
        help="AWQ 推理内核类型，awq_marlin 通常更快",
    )
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
        from vllm import LLM, SamplingParams
    except ImportError:
        print("[ERROR] 缺少依赖：pip install vllm", file=sys.stderr)
        sys.exit(1)

    tokenizer_path = Path(args.tokenizer_path) if args.tokenizer_path else model_path
    if not tokenizer_path.exists():
        print(f"[ERROR] tokenizer 路径不存在: {tokenizer_path}", file=sys.stderr)
        sys.exit(1)
    tokenizer_path = maybe_prepare_tokenizer_fix(model_path=model_path, tokenizer_path=tokenizer_path)

    print(f"[INFO] 加载 vLLM AWQ 模型: {model_path}")
    print(f"[INFO] 使用 tokenizer: {tokenizer_path}")
    t_load0 = time.perf_counter()
    llm = LLM(
        model=str(model_path),
        tokenizer=str(tokenizer_path),
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=True,
        quantization=args.quantization,
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    load_time_s = time.perf_counter() - t_load0
    tokenizer = llm.get_tokenizer()

    samples = load_samples(test_path, args.num_samples)
    if not samples:
        print("[ERROR] 没有可用样本", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        repetition_penalty=1.1,
    )

    pass_flags: List[bool] = []
    latencies: List[float] = []
    rows: List[Dict] = []

    for i, sample in enumerate(tqdm(samples, desc="vLLM smoke", unit="sample"), start=1):
        prompt = str(sample.get("input", "")).strip()
        if not prompt:
            continue
        final_prompt = build_prompt(tokenizer, prompt)

        t0 = time.perf_counter()
        outputs = llm.generate([final_prompt], sp, use_tqdm=False)
        latency_s = time.perf_counter() - t0
        latencies.append(latency_s)

        pred = ""
        if outputs and outputs[0].outputs:
            pred = outputs[0].outputs[0].text
        pred = clean_think(pred)

        passed = has_all_sections(pred)
        pass_flags.append(passed)

        row = {
            "index": i - 1,
            "input": prompt,
            "reference": sample.get("output", ""),
            "prediction": pred,
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
        "backend": "vllm",
        "model_path": str(model_path),
        "num_samples": total,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "all_sections_pass_rate": pass_rate,
        "avg_latency_s": (statistics.mean(latencies) if latencies else 0.0),
        "p50_latency_s": (statistics.median(latencies) if latencies else 0.0),
        "p95_latency_s": p95(latencies),
        "load_time_s": load_time_s,
        "threshold_pass": pass_rate >= 0.90,
        "threshold": "all_sections_pass_rate >= 0.90",
        "prediction_file": str(output_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== vLLM AWQ 冒烟结果 =====")
    print(f"样本数:       {total}")
    print(f"字段完整率:   {pass_rate:.2%}")
    print(f"加载耗时:     {load_time_s:.2f}s")
    print(f"平均耗时:     {report['avg_latency_s']:.3f}s")
    print(f"阈值通过:     {report['threshold_pass']}")
    print(f"预测文件:     {output_path}")
    print(f"报告文件:     {report_path}")


if __name__ == "__main__":
    main()
