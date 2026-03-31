#!/usr/bin/env python3
"""
12_benchmark_quality_awq.py
对比 BF16 merged 模型与 AWQ4 模型的质量表现，并输出统一报告：
- ROUGE-1 / ROUGE-2 / ROUGE-L
- 格式合规率（复用 06_eval_rouge_and_format.py 口径）
- 阈值校验（ROUGE-L 下降 <=3%，字段完整率下降 <=2%）

示例：
  python scripts/12_benchmark_quality_awq.py \
    --bf16_model_path outputs/merged/qwen3-4b-news-v2 \
    --awq_model_path outputs/quantized/qwen3-4b-news-v2-awq4 \
    --n_samples 0 \
    --output_dir outputs/eval/awq_benchmark
"""

import argparse
import gc
import importlib.util
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_TEST = PROJECT_DIR / "data" / "cleaned" / "test.json"
DEFAULT_BF16 = PROJECT_DIR / "outputs" / "merged" / "qwen3-4b-news-v2"
DEFAULT_AWQ = PROJECT_DIR / "outputs" / "quantized" / "qwen3-4b-news-v2-awq4"
DEFAULT_OUTDIR = PROJECT_DIR / "outputs" / "eval" / "awq_benchmark"


def load_eval_module():
    """动态加载 06 脚本中的评测函数，确保口径一致。"""
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


def sync_cuda(torch_mod) -> None:
    try:
        if torch_mod.cuda.is_available():
            torch_mod.cuda.synchronize()
    except Exception:
        pass


def safe_generate(model, tokenizer, model_inputs: Dict, max_new_tokens: int):
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0.1,
        "do_sample": False,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id,
    }
    try:
        return model.generate(**model_inputs, **kwargs)
    except TypeError:
        kwargs.pop("repetition_penalty", None)
        return model.generate(**model_inputs, **kwargs)


def build_input_ids(tokenizer, system_prompt: str, user_prompt: str):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
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
        return tokenizer(f"{system_prompt}\n\n{user_prompt}", return_tensors="pt").input_ids


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


def infer_predictions(model, tokenizer, test_data: List[Dict], system_prompt: str, max_new_tokens: int, tag: str) -> Tuple[List[str], float]:
    try:
        import torch
    except ImportError:
        print("[ERROR] 缺少 torch", file=sys.stderr)
        sys.exit(1)

    preds = []  # type: List[str]
    latencies = []  # type: List[float]

    for idx, row in enumerate(test_data, start=1):
        prompt = str(row.get("input", "")).strip()
        if not prompt:
            preds.append("")
            continue
        raw_inputs = build_input_ids(tokenizer, system_prompt, prompt)
        model_inputs = normalize_model_inputs(raw_inputs)
        try:
            device = next(model.parameters()).device
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        except Exception:
            pass

        sync_cuda(torch)
        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = safe_generate(model, tokenizer, model_inputs, max_new_tokens=max_new_tokens)
        sync_cuda(torch)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        input_len = int(model_inputs["input_ids"].shape[-1])
        pred = tokenizer.decode(out_ids[0][input_len:], skip_special_tokens=True)
        preds.append(pred)
        print(f"[{tag}] {idx:3d}/{len(test_data)} latency={elapsed:.3f}s")

    avg_latency = statistics.mean(latencies) if latencies else 0.0
    return preds, avg_latency


def save_group_outputs(output_dir: Path, group_name: str, test_data: List[Dict], preds_raw: List[str], eval_mod) -> Dict:
    refs = [str(x.get("output", "")) for x in test_data]
    clean_preds = [eval_mod.strip_think_block(p) for p in preds_raw]
    rouge = eval_mod.compute_rouge(refs, clean_preds, use_jieba=True)
    fmt_report, bad_indices = eval_mod.evaluate_format(clean_preds)

    group_dir = output_dir / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    pred_file = group_dir / "predictions_raw.jsonl"
    with open(pred_file, "w", encoding="utf-8") as f:
        for row, raw, clean in zip(test_data, preds_raw, clean_preds):
            f.write(
                json.dumps(
                    {
                        "input": row.get("input", ""),
                        "reference": row.get("output", ""),
                        "raw": raw,
                        "clean": clean,
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

    return {
        "group": group_name,
        "n_samples": len(test_data),
        "rouge1": float(rouge["rouge1"]),
        "rouge2": float(rouge["rouge2"]),
        "rougeL": float(rouge["rougeL"]),
        "all_sections_pass_rate": float(fmt_report["all_sections_pass_rate"]),
        "valid_category_rate": float(fmt_report["valid_category_rate"]),
        "valid_bullets_rate": float(fmt_report["valid_bullets_rate"]),
        "bad_cases": int(len(bad_indices)),
    }


def _clear_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description="BF16 vs AWQ4 质量对比评测")
    parser.add_argument("--test", type=str, default=str(DEFAULT_TEST))
    parser.add_argument("--bf16_model_path", type=str, default=str(DEFAULT_BF16))
    parser.add_argument("--awq_model_path", type=str, default=str(DEFAULT_AWQ))
    parser.add_argument("--n_samples", type=int, default=0, help="0 表示全量")
    parser.add_argument("--max_new_tokens", type=int, default=800)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTDIR))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    test_path = Path(args.test)
    bf16_path = Path(args.bf16_model_path)
    awq_path = Path(args.awq_model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for p in [test_path, bf16_path, awq_path]:
        if not p.exists():
            print(f"[ERROR] 路径不存在: {p}", file=sys.stderr)
            sys.exit(1)

    eval_mod = load_eval_module()
    system_prompt = getattr(eval_mod, "MEDIUM_SYSTEM_PROMPT", "你是专业的新闻编辑助手。")
    test_data = load_test_data(test_path, args.n_samples)
    if not test_data:
        print("[ERROR] 测试集为空", file=sys.stderr)
        sys.exit(1)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("[ERROR] 缺少 transformers/torch 依赖", file=sys.stderr)
        sys.exit(1)

    # --- BF16 ---
    print(f"[INFO] 加载 BF16 模型: {bf16_path}")
    bf16_tok = AutoTokenizer.from_pretrained(str(bf16_path), trust_remote_code=True)
    bf16_model = AutoModelForCausalLM.from_pretrained(
        str(bf16_path),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    bf16_model.eval()
    bf16_preds, bf16_latency = infer_predictions(
        model=bf16_model,
        tokenizer=bf16_tok,
        test_data=test_data,
        system_prompt=system_prompt,
        max_new_tokens=args.max_new_tokens,
        tag="BF16",
    )
    bf16_summary = save_group_outputs(output_dir, "group_bf16", test_data, bf16_preds, eval_mod)
    bf16_summary["per_sample_s"] = bf16_latency

    del bf16_model
    _clear_cuda()

    # --- AWQ4 ---
    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        print("[ERROR] 缺少 autoawq：pip install autoawq", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 加载 AWQ 模型: {awq_path}")
    awq_tok = AutoTokenizer.from_pretrained(str(awq_path), trust_remote_code=True)

    kwargs_candidates = [
        {
            "quant_path": str(awq_path),
            "trust_remote_code": True,
            "fuse_layers": True,
            "device_map": "auto",
            "safetensors": True,
        },
        {
            "quant_path": str(awq_path),
            "trust_remote_code": True,
            "fuse_layers": False,
            "device_map": "auto",
            "safetensors": True,
        },
        {
            "quant_path": str(awq_path),
            "trust_remote_code": True,
            "device_map": "auto",
        },
    ]

    awq_model = None
    last_err = None
    for kwargs in kwargs_candidates:
        try:
            awq_model = AutoAWQForCausalLM.from_quantized(**kwargs)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    if awq_model is None:
        print(f"[ERROR] AWQ 模型加载失败: {last_err}", file=sys.stderr)
        sys.exit(1)
    awq_model.eval()
    awq_preds, awq_latency = infer_predictions(
        model=awq_model,
        tokenizer=awq_tok,
        test_data=test_data,
        system_prompt=system_prompt,
        max_new_tokens=args.max_new_tokens,
        tag="AWQ4",
    )
    awq_summary = save_group_outputs(output_dir, "group_awq4", test_data, awq_preds, eval_mod)
    awq_summary["per_sample_s"] = awq_latency

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

    summary_report = {
        "task": "bf16_vs_awq4_quality",
        "n_samples": len(test_data),
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "variants": [bf16_summary, awq_summary],
        "comparison": comparison,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = output_dir / "benchmark_summary_awq.json"
    summary_path.write_text(json.dumps(summary_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 质量对比汇总 =====")
    print(f"BF16  RougeL: {bf16_summary['rougeL']:.4f} | 字段完整率: {bf16_summary['all_sections_pass_rate']:.2%}")
    print(f"AWQ4  RougeL: {awq_summary['rougeL']:.4f} | 字段完整率: {awq_summary['all_sections_pass_rate']:.2%}")
    print(f"RougeL 下降: {rougeL_drop:.2%} (阈值<=3%)")
    print(f"字段完整率下降: {section_drop:.2%} (阈值<=2%)")
    print(f"报告输出: {summary_path}")


if __name__ == "__main__":
    main()
