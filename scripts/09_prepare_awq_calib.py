#!/usr/bin/env python3
"""
09_prepare_awq_calib.py
从训练集抽取 AWQ 校准样本，生成 jsonl 文本文件。

示例：
  python scripts/09_prepare_awq_calib.py \
    --train data/cleaned/train.json \
    --output outputs/awq/calib_prompts.jsonl \
    --num_samples 256 \
    --seed 42
"""

import argparse
import importlib.util
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_TRAIN = PROJECT_DIR / "data" / "cleaned" / "train.json"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "awq" / "calib_prompts.jsonl"
DEFAULT_TOKENIZER = PROJECT_DIR / "outputs" / "merged" / "qwen3-4b-news-v2"


def _extract_user_prompt(record: Dict) -> str:
    """优先使用 input，兜底拼接 instruction + input。"""
    input_text = str(record.get("input", "")).strip()
    if input_text:
        return input_text

    instruction = str(record.get("instruction", "")).strip()
    if instruction:
        return f"{instruction}\n\n{input_text}".strip()
    return ""


def load_records(path: Path) -> List[Dict]:
    """加载 json 数据并标准化为 list[dict]。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    return [row for row in data if isinstance(row, dict)]


def load_medium_system_prompt() -> str:
    """优先读取主评测脚本的系统提示词，保证口径一致。"""
    eval_script = SCRIPT_DIR / "06_eval_rouge_and_format.py"
    if not eval_script.exists():
        return "你是专业的新闻编辑助手。"
    try:
        spec = importlib.util.spec_from_file_location("eval_mod", eval_script)
        if spec is None or spec.loader is None:
            return "你是专业的新闻编辑助手。"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return str(getattr(module, "MEDIUM_SYSTEM_PROMPT", "你是专业的新闻编辑助手。"))
    except Exception:
        return "你是专业的新闻编辑助手。"


def parse_bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "yes", "y", "on"}


def percentile(values: List[int], p: float) -> int:
    if not values:
        return 0
    arr = sorted(values)
    idx = int((p / 100.0) * (len(arr) - 1))
    idx = max(0, min(idx, len(arr) - 1))
    return int(arr[idx])


def build_prompt_chat_template(tokenizer, system_prompt: str, user_prompt: str) -> str:
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


def sample_stratified_by_length(
    items: List[Dict],
    n_samples: int,
    rng: random.Random,
) -> Tuple[List[Dict], Dict]:
    """按 token 长度分桶（短/中/长）进行分层抽样。"""
    if not items:
        return [], {
            "thresholds": {"short_max": 0, "medium_max": 0},
            "candidate_counts": {"short": 0, "medium": 0, "long": 0},
            "selected_counts": {"short": 0, "medium": 0, "long": 0},
        }

    lengths = sorted(int(it["token_len"]) for it in items)
    short_max = lengths[len(lengths) // 3]
    medium_max = lengths[(len(lengths) * 2) // 3]

    bins = {"short": [], "medium": [], "long": []}
    for it in items:
        tl = int(it["token_len"])
        if tl <= short_max:
            bins["short"].append(it)
        elif tl <= medium_max:
            bins["medium"].append(it)
        else:
            bins["long"].append(it)

    for key in bins:
        rng.shuffle(bins[key])

    base = n_samples // 3
    targets = {"short": base, "medium": base, "long": base}
    rem = n_samples - base * 3
    order = sorted(["short", "medium", "long"], key=lambda k: len(bins[k]), reverse=True)
    for i in range(rem):
        targets[order[i % 3]] += 1

    selected = []
    for key in ["short", "medium", "long"]:
        take_n = min(targets[key], len(bins[key]))
        selected.extend(bins[key][:take_n])

    if len(selected) < n_samples:
        selected_ids = {id(x) for x in selected}
        leftovers = [it for it in items if id(it) not in selected_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: (n_samples - len(selected))])

    selected = selected[:n_samples]

    selected_counts = {"short": 0, "medium": 0, "long": 0}
    for it in selected:
        tl = int(it["token_len"])
        if tl <= short_max:
            selected_counts["short"] += 1
        elif tl <= medium_max:
            selected_counts["medium"] += 1
        else:
            selected_counts["long"] += 1

    meta = {
        "thresholds": {"short_max": int(short_max), "medium_max": int(medium_max)},
        "candidate_counts": {k: len(v) for k, v in bins.items()},
        "selected_counts": selected_counts,
    }
    return selected, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="准备 AWQ 校准集")
    parser.add_argument("--train", type=str, default=str(DEFAULT_TRAIN), help="训练集路径（JSON）")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="输出路径（JSONL）")
    parser.add_argument("--num_samples", type=int, default=256, help="抽样数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--min_chars", type=int, default=64, help="最小文本长度")
    parser.add_argument("--max_chars", type=int, default=4000, help="最大文本长度（超出截断）")
    parser.add_argument("--prompt_mode", type=str, default="input", choices=["input", "chat_template"])
    parser.add_argument("--system_prompt_source", type=str, default="medium", choices=["medium", "custom"])
    parser.add_argument("--system_prompt_text", type=str, default="", help="当 source=custom 时生效")
    parser.add_argument("--stratified_by_length", type=parse_bool, default=False, help="是否按长度分桶抽样")
    parser.add_argument("--tokenizer_path", type=str, default=str(DEFAULT_TOKENIZER), help="tokenizer 路径")
    parser.add_argument("--stats_output", type=str, default="", help="统计信息输出路径（JSON）")
    args = parser.parse_args()

    train_path = Path(args.train)
    if not train_path.exists():
        print(f"[ERROR] 训练集不存在: {train_path}", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    rows = load_records(train_path)
    rng.shuffle(rows)

    tokenizer = None
    tokenizer_path = Path(args.tokenizer_path) if args.tokenizer_path else None
    if args.prompt_mode == "chat_template" or args.stratified_by_length:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            print("[ERROR] 缺少 transformers：pip install transformers", file=sys.stderr)
            sys.exit(1)
        if tokenizer_path is None or not tokenizer_path.exists():
            print(f"[ERROR] tokenizer 路径不存在: {tokenizer_path}", file=sys.stderr)
            sys.exit(1)
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)

    if args.system_prompt_source == "medium":
        system_prompt = load_medium_system_prompt()
    else:
        if not args.system_prompt_text.strip():
            print("[ERROR] source=custom 时必须提供 --system_prompt_text", file=sys.stderr)
            sys.exit(1)
        system_prompt = args.system_prompt_text.strip()

    candidates: List[Dict] = []
    for row in rows:
        user_prompt = _extract_user_prompt(row)
        if not user_prompt:
            continue
        if len(user_prompt) < args.min_chars:
            continue
        if len(user_prompt) > args.max_chars:
            user_prompt = user_prompt[: args.max_chars]

        if args.prompt_mode == "chat_template":
            assert tokenizer is not None
            prompt = build_prompt_chat_template(tokenizer, system_prompt, user_prompt)
        else:
            prompt = user_prompt

        if tokenizer is not None:
            token_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        else:
            token_len = len(prompt)

        candidates.append(
            {
                "id": row.get("id", ""),
                "prompt": prompt,
                "user_prompt": user_prompt,
                "token_len": int(token_len),
            }
        )

    if not candidates:
        print("[ERROR] 未抽取到有效校准文本", file=sys.stderr)
        sys.exit(1)

    num_samples = args.num_samples if args.num_samples > 0 else len(candidates)
    target_n = min(num_samples, len(candidates))

    strat_meta = {
        "thresholds": {"short_max": 0, "medium_max": 0},
        "candidate_counts": {"short": 0, "medium": 0, "long": 0},
        "selected_counts": {"short": 0, "medium": 0, "long": 0},
    }
    if args.stratified_by_length:
        selected, strat_meta = sample_stratified_by_length(candidates, target_n, rng)
    else:
        rng.shuffle(candidates)
        selected = candidates[:target_n]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(selected):
            row = {
                "id": item.get("id", idx),
                "prompt": item["prompt"],
                "token_len": int(item["token_len"]),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lengths_all = [int(x["token_len"]) for x in candidates]
    lengths_sel = [int(x["token_len"]) for x in selected]
    stats = {
        "train_path": str(train_path),
        "output_path": str(output_path),
        "seed": args.seed,
        "prompt_mode": args.prompt_mode,
        "system_prompt_source": args.system_prompt_source,
        "system_prompt_preview": system_prompt[:120],
        "stratified_by_length": bool(args.stratified_by_length),
        "num_candidates": len(candidates),
        "num_selected": len(selected),
        "tokenizer_path": str(tokenizer_path) if tokenizer_path else "",
        "candidate_token_stats": {
            "avg": round(statistics.mean(lengths_all), 2),
            "p50": percentile(lengths_all, 50),
            "p95": percentile(lengths_all, 95),
            "min": min(lengths_all),
            "max": max(lengths_all),
        },
        "selected_token_stats": {
            "avg": round(statistics.mean(lengths_sel), 2),
            "p50": percentile(lengths_sel, 50),
            "p95": percentile(lengths_sel, 95),
            "min": min(lengths_sel),
            "max": max(lengths_sel),
        },
        "stratified_meta": strat_meta,
    }
    stats_path = Path(args.stats_output) if args.stats_output else (output_path.parent / "calib_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] 原始样本: {len(rows)}")
    print(f"[INFO] 有效文本: {len(candidates)}")
    print(f"[INFO] 写出样本: {len(selected)} | mode={args.prompt_mode} | stratified={args.stratified_by_length}")
    print(f"[INFO] 输出文件: {output_path}")
    print(f"[INFO] 统计文件: {stats_path}")


if __name__ == "__main__":
    main()
