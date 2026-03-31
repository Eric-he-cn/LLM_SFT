#!/usr/bin/env python3
"""
14_benchmark_quality_vllm.py
统一使用 vLLM 对比 SFT merged(BF16) 与 AWQ4 的全量质量表现，支持断点续跑。

核心特性：
1) 两组串行执行（同卡避免并行污染）。
2) 每条样本即时写 checkpoint（JSONL），中断后自动续跑。
3) 输出统一质量指标与阈值判断。

示例：
  python scripts/14_benchmark_quality_vllm.py \
    --bf16_model_path outputs/merged/qwen3-4b-news-v2 \
    --awq_model_path outputs/quantized/qwen3-4b-news-v2-awq4 \
    --n_samples 0 \
    --output_dir outputs/eval/awq_benchmark_vllm
"""

import argparse
import gc
import importlib.util
import json
import random
import re
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DEFAULT_TEST = PROJECT_DIR / "data" / "cleaned" / "test.json"
DEFAULT_BF16 = PROJECT_DIR / "outputs" / "merged" / "qwen3-4b-news-v2"
DEFAULT_AWQ = PROJECT_DIR / "outputs" / "quantized" / "qwen3-4b-news-v2-awq4"
DEFAULT_OUTDIR = PROJECT_DIR / "outputs" / "eval" / "awq_benchmark_vllm"


def load_eval_module():
    eval_script = SCRIPT_DIR / "06_eval_rouge_and_format.py"
    spec = importlib.util.spec_from_file_location("eval_mod", eval_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载评测模块: {eval_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_test_data(path: Path, n_samples: int) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    if n_samples <= 0:
        return data
    return data[:n_samples]


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    idx = int((p / 100.0) * (len(values_sorted) - 1))
    idx = max(0, min(idx, len(values_sorted) - 1))
    return values_sorted[idx]


def _clear_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def maybe_prepare_tokenizer_fix(model_path: Path, tokenizer_path: Path) -> Tuple[Path, bool]:
    """兼容 transformers 对 extra_special_tokens=list 的报错，生成修复副本。"""
    cfg_path = tokenizer_path / "tokenizer_config.json"
    if not cfg_path.exists():
        return tokenizer_path, False
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return tokenizer_path, False

    extra = cfg.get("extra_special_tokens")
    if not isinstance(extra, list):
        return tokenizer_path, False

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
    return fixed_dir, True


def build_prompt_text(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
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
            return f"{system_prompt}\n\n{user_prompt}"


def load_checkpoint(checkpoint_path: Path, total: int) -> Tuple[List[Optional[str]], List[Optional[float]], int]:
    preds: List[Optional[str]] = [None] * total
    latencies: List[Optional[float]] = [None] * total
    recovered = 0
    if not checkpoint_path.exists():
        return preds, latencies, recovered

    with open(checkpoint_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = int(row.get("index", -1))
            if 0 <= idx < total:
                if preds[idx] is None:
                    recovered += 1
                preds[idx] = str(row.get("raw", ""))
                lat = row.get("latency_s")
                if isinstance(lat, (int, float)):
                    latencies[idx] = float(lat)
    return preds, latencies, recovered


def save_group_outputs(
    output_dir: Path,
    group_name: str,
    test_data: List[Dict],
    preds_raw: List[str],
    per_sample_latency: List[float],
    eval_mod,
) -> Dict:
    refs = [str(x.get("output", "")) for x in test_data]
    clean_preds = [eval_mod.strip_think_block(p) for p in preds_raw]
    rouge = eval_mod.compute_rouge(refs, clean_preds, use_jieba=True)
    fmt_report, bad_indices = eval_mod.evaluate_format(clean_preds)

    empty_output_rate = (
        sum(1 for p in clean_preds if not str(p).strip()) / len(clean_preds)
        if clean_preds
        else 0.0
    )

    group_dir = output_dir / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    pred_file = group_dir / "predictions_raw.jsonl"

    with open(pred_file, "w", encoding="utf-8") as f:
        for i, (row, raw, clean) in enumerate(zip(test_data, preds_raw, clean_preds)):
            check_row = eval_mod.check_format(clean)
            f.write(
                json.dumps(
                    {
                        "index": i,
                        "input": row.get("input", ""),
                        "reference": row.get("output", ""),
                        "raw": raw,
                        "clean": clean,
                        "latency_s": per_sample_latency[i] if i < len(per_sample_latency) else None,
                        "all_sections_present": bool(check_row.get("all_sections_present", False)),
                        "valid_category": bool(check_row.get("valid_category", False)),
                        "valid_bullets": bool(check_row.get("valid_bullets", False)),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    (group_dir / "rouge_report.json").write_text(json.dumps(rouge, ensure_ascii=False, indent=2), encoding="utf-8")
    (group_dir / "format_report.json").write_text(json.dumps(fmt_report, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(group_dir / "bad_cases.jsonl", "w", encoding="utf-8") as f:
        for i in bad_indices:
            f.write(json.dumps({"index": i, "clean": clean_preds[i]}, ensure_ascii=False) + "\n")

    valid_lat = [x for x in per_sample_latency if isinstance(x, (int, float)) and x > 0]
    return {
        "group": group_name,
        "n_samples": len(test_data),
        "rouge1": float(rouge["rouge1"]),
        "rouge2": float(rouge["rouge2"]),
        "rougeL": float(rouge["rougeL"]),
        "all_sections_pass_rate": float(fmt_report["all_sections_pass_rate"]),
        "valid_category_rate": float(fmt_report["valid_category_rate"]),
        "valid_bullets_rate": float(fmt_report["valid_bullets_rate"]),
        "has_time_info_rate": float(fmt_report["has_time_info_rate"]),
        "avg_bullet_count": float(fmt_report["avg_bullet_count"]),
        "empty_output_rate": float(empty_output_rate),
        "bad_cases": int(len(bad_indices)),
        "avg_latency_s": (statistics.mean(valid_lat) if valid_lat else 0.0),
        "latency_p50_s": percentile(valid_lat, 50),
        "latency_p95_s": percentile(valid_lat, 95),
    }


def infer_group_vllm(
    group_name: str,
    model_path: Path,
    tokenizer_path: Optional[Path],
    quantization: str,
    test_data: List[Dict],
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    output_dir: Path,
    eval_mod,
) -> Dict:
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("[ERROR] 缺少 vllm：pip install vllm", file=sys.stderr)
        sys.exit(1)

    group_dir = output_dir / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = group_dir / "infer_checkpoint.jsonl"

    n = len(test_data)
    preds, latencies, recovered = load_checkpoint(checkpoint_path, n)
    if recovered > 0:
        print(f"[{group_name}] 检测到 checkpoint，已恢复 {recovered}/{n} 条。")

    unresolved = [i for i, p in enumerate(preds) if p is None]

    raw_tokenizer_path = tokenizer_path if tokenizer_path is not None else model_path
    if not raw_tokenizer_path.exists():
        print(f"[ERROR] tokenizer 路径不存在: {raw_tokenizer_path}", file=sys.stderr)
        sys.exit(1)
    resolved_tokenizer_path, tokenizer_fix_applied = maybe_prepare_tokenizer_fix(
        model_path=model_path,
        tokenizer_path=raw_tokenizer_path,
    )
    if tokenizer_fix_applied:
        print(f"[{group_name}] tokenizer 自动修复目录: {resolved_tokenizer_path}")

    q = None if quantization == "none" else quantization
    load_t0 = time.perf_counter()
    llm = LLM(
        model=str(model_path),
        tokenizer=str(resolved_tokenizer_path),
        tokenizer_mode="auto",
        trust_remote_code=True,
        quantization=q,
        dtype="float16",
        tensor_parallel_size=1,
    )
    load_time_s = time.perf_counter() - load_t0

    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )

    if unresolved:
        ckpt_f = open(checkpoint_path, "a", encoding="utf-8")
        pbar = None
        if tqdm is not None:
            pbar = tqdm(total=n, initial=n - len(unresolved), desc=group_name, unit="sample", dynamic_ncols=True)
        else:
            print(f"[{group_name}] 将推理 {len(unresolved)} 条（总计 {n}）。")

        try:
            for idx in unresolved:
                prompt = str(test_data[idx].get("input", "")).strip()
                if not prompt:
                    preds[idx] = ""
                    latencies[idx] = 0.0
                else:
                    prompt_text = build_prompt_text(tokenizer, system_prompt, prompt)
                    t0 = time.perf_counter()
                    outputs = llm.generate([prompt_text], sp, use_tqdm=False)
                    elapsed = time.perf_counter() - t0

                    pred = ""
                    if outputs and outputs[0].outputs:
                        pred = outputs[0].outputs[0].text
                    preds[idx] = pred
                    latencies[idx] = elapsed

                ckpt_f.write(
                    json.dumps(
                        {
                            "index": idx,
                            "raw": preds[idx],
                            "latency_s": latencies[idx],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                ckpt_f.flush()
                if pbar is not None:
                    pbar.update(1)
        finally:
            if pbar is not None:
                pbar.close()
            ckpt_f.close()
    else:
        print(f"[{group_name}] checkpoint 已覆盖全部样本，跳过推理。")

    del llm
    _clear_cuda()

    preds_final = [p if p is not None else "" for p in preds]
    lat_final = [float(x) if isinstance(x, (int, float)) else 0.0 for x in latencies]
    summary = save_group_outputs(
        output_dir=output_dir,
        group_name=group_name,
        test_data=test_data,
        preds_raw=preds_final,
        per_sample_latency=lat_final,
        eval_mod=eval_mod,
    )
    summary["load_time_s"] = load_time_s
    summary["quantization"] = quantization
    summary["tokenizer_path"] = str(resolved_tokenizer_path)
    summary["tokenizer_fix_applied"] = tokenizer_fix_applied
    summary["checkpoint_file"] = str(checkpoint_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM 全量质量对比（BF16 vs AWQ4，支持断点续跑）")
    parser.add_argument("--test", type=str, default=str(DEFAULT_TEST))
    parser.add_argument("--bf16_model_path", type=str, default=str(DEFAULT_BF16))
    parser.add_argument("--awq_model_path", type=str, default=str(DEFAULT_AWQ))
    parser.add_argument("--bf16_tokenizer_path", type=str, default="")
    parser.add_argument("--awq_tokenizer_path", type=str, default="")
    parser.add_argument("--n_samples", type=int, default=0, help="0 表示全量")
    parser.add_argument("--max_new_tokens", type=int, default=800)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--awq_quantization", type=str, default="awq_marlin", choices=["awq", "awq_marlin"])
    parser.add_argument("--awq_only", action="store_true", help="仅运行 AWQ 组，不运行 BF16 组")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTDIR))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    test_path = Path(args.test)
    bf16_path = Path(args.bf16_model_path)
    awq_path = Path(args.awq_model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_paths = [test_path, awq_path]
    if not args.awq_only:
        required_paths.append(bf16_path)

    for p in required_paths:
        if not p.exists():
            print(f"[ERROR] 路径不存在: {p}", file=sys.stderr)
            sys.exit(1)

    eval_mod = load_eval_module()
    system_prompt = getattr(eval_mod, "MEDIUM_SYSTEM_PROMPT", "你是专业的新闻编辑助手。")
    test_data = load_test_data(test_path, args.n_samples)
    if not test_data:
        print("[ERROR] 测试集为空", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 样本数: {len(test_data)}")
    bf16_summary = None
    if args.awq_only:
        print("[INFO] awq_only=true，跳过 group_bf16。")
    else:
        print("[INFO] 开始 group_bf16 ...")
        bf16_summary = infer_group_vllm(
            group_name="group_bf16",
            model_path=bf16_path,
            tokenizer_path=(Path(args.bf16_tokenizer_path) if args.bf16_tokenizer_path else None),
            quantization="none",
            test_data=test_data,
            system_prompt=system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            output_dir=output_dir,
            eval_mod=eval_mod,
        )

    print("[INFO] 开始 group_awq4 ...")
    awq_summary = infer_group_vllm(
        group_name="group_awq4",
        model_path=awq_path,
        tokenizer_path=(Path(args.awq_tokenizer_path) if args.awq_tokenizer_path else None),
        quantization=args.awq_quantization,
        test_data=test_data,
        system_prompt=system_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        output_dir=output_dir,
        eval_mod=eval_mod,
    )

    if bf16_summary is not None:
        rougeL_drop = (
            (bf16_summary["rougeL"] - awq_summary["rougeL"]) / bf16_summary["rougeL"]
            if bf16_summary["rougeL"] > 0
            else 0.0
        )
        section_drop = bf16_summary["all_sections_pass_rate"] - awq_summary["all_sections_pass_rate"]
        comparison = {
            "rougeL_drop_ratio": rougeL_drop,
            "all_sections_pass_drop": section_drop,
            "thresholds": {
                "rougeL_drop_ratio_le_0_03": rougeL_drop <= 0.03,
                "all_sections_pass_drop_le_0_02": section_drop <= 0.02,
            },
        }
        variants = [bf16_summary, awq_summary]
    else:
        rougeL_drop = None
        section_drop = None
        comparison = None
        variants = [awq_summary]

    summary_report = {
        "task": "bf16_vs_awq4_quality_vllm",
        "backend": "vllm",
        "n_samples": len(test_data),
        "seed": args.seed,
        "decode_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
        },
        "awq_only": args.awq_only,
        "variants": variants,
        "comparison": comparison,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = output_dir / "benchmark_summary_awq_vllm.json"
    summary_path.write_text(json.dumps(summary_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== vLLM 质量评测汇总 =====")
    if bf16_summary is not None:
        print(f"BF16  RougeL: {bf16_summary['rougeL']:.4f} | 字段完整率: {bf16_summary['all_sections_pass_rate']:.2%}")
        print(f"AWQ4  RougeL: {awq_summary['rougeL']:.4f} | 字段完整率: {awq_summary['all_sections_pass_rate']:.2%}")
        print(f"RougeL 下降: {rougeL_drop:.2%} (阈值<=3%)")
        print(f"字段完整率下降: {section_drop:.2%} (阈值<=2%)")
    else:
        print(f"AWQ4  RougeL: {awq_summary['rougeL']:.4f} | 字段完整率: {awq_summary['all_sections_pass_rate']:.2%}")
        print("BF16 对照未运行（awq_only=true）。")
    print(f"报告输出: {summary_path}")


if __name__ == "__main__":
    main()
