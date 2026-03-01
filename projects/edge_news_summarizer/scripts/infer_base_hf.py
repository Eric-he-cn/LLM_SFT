#!/usr/bin/env python3
"""
infer_base_hf.py
使用 HuggingFace Transformers 对 Qwen3-4B 进行批量推理，支持基座模型和微调模型（LoRA adapter）。

特性：
  - 通过 --adapter 加载 LoRA adapter，无需单独脚本
  - 每处理 SAVE_EVERY 条自动保存一次，防止长时间运行后丢失结果
  - 支持断点续推：已存在输出文件时自动跳过已完成的条目，从上次中断处继续
  - 输出格式与 LlamaFactory generated_predictions.jsonl 完全一致

运行环境：my_sft

用法（基座模型，v3 严格 prompt）：
  cd D:\LLM\MySFT\LLM_SFT\projects\edge_news_summarizer
  python scripts/infer_base_hf.py

用法（微调模型，v3 严格 prompt）：
  python scripts/infer_base_hf.py \
    --adapter "D:/LLM/LlamaFactory/projects/edge_news_summarizer/outputs/checkpoints/qwen3-4b-qlora-news" \
    --output_dir outputs/eval_ft_v3

  python scripts/infer_base_hf.py --restart           # 强制从头重跑（忽略已有结果）

评测（运行完成后）：
  python scripts/06_eval_rouge_and_format.py \
    --predictions outputs/eval_ft_v3/generated_predictions.jsonl \
    --output_dir  outputs/eval_ft_v3
"""

import argparse
import json
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DEFAULT_MODEL      = "D:/LLM/models/Qwen3-4B"
DEFAULT_ADAPTER    = None   # 设置后加载 LoRA adapter（微调模型推理）
DEFAULT_TEST       = PROJECT_DIR / "data" / "cleaned" / "test.json"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "eval_base_v3"
SAVE_EVERY         = 20          # 每处理 20 条保存一次
MAX_NEW_TOKENS     = 800         # 无思考链时足够（严格 prompt 关闭 thinking）
TEMPERATURE        = 0.1
TOP_P              = 0.9
REPETITION_PENALTY = 1.1
BATCH_SIZE         = 4           # 批量推理大小（16GB 显存下 4B 模型安全值）

# 严格格式化系统提示词（v3 默认使用，用于提升基座模型格式合规率）
# 原始 test.json 里的 instruction 只说「做结构化摘要」，没有给出任何格式规范，
# 导致基座模型按预训练习惯输出 Markdown（###、**、-），格式合规率 0%。
# 此 prompt 给出完整模板、合法类别枚举、禁止 Markdown 等约束。
STRICT_SYSTEM_PROMPT = """\
你是一位专业的新闻编辑助手。请严格按照下面的【格式模板】对新闻进行结构化摘要输出。
直接输出摘要内容，不输出任何分析思考过程。

【格式模板】（必须完全遵守，不得增删标签，不得使用 Markdown、#、*、- 等符号）

【一句话摘要】
用一句话概括新闻核心事件，不超过60字。

【核心要点】
1. 第一个关键信息点（每条不超过80字）
2. 第二个关键信息点（每条不超过80字）
3. 第三个关键信息点（每条不超过80字）
（至少3条，最多6条，每条以阿拉伯数字加点号开头）

【事件类别】
从以下选项中选取1-2个，用顿号分隔：政治、经济、科技、文化、社会、军事、体育、健康、环境、国际、历史、旅游、财经

【主要主体】
涉及的主要人物、组织或地区，用顿号分隔，不超过80字。

【时间信息】
涉及的所有时间节点，用顿号分隔，不超过50字。

【潜在影响】
分析此事件的潜在影响或意义，1-3句话，不超过100字。

无论新闻原文是中文还是英文，请始终用中文输出以上所有字段。
"""  # noqa: E501


# ─────────────────────────── 工具函数 ─────────────────────────────────────────

def load_test_data(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def load_done_indices(out_path: Path) -> set[int]:
    """读取已有输出文件，返回已完成的样本序号集合。"""
    done = set()
    if not out_path.exists():
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "_idx" in obj:
                    done.add(obj["_idx"])
            except json.JSONDecodeError:
                pass
    return done


def append_result(out_path: Path, idx: int, predict: str, label: str) -> None:
    """追加单条结果（含 _idx 字段用于断点定位，评测脚本会忽略多余字段）。"""
    line = json.dumps({"_idx": idx, "predict": predict, "label": label},
                      ensure_ascii=False)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def reorder_output(out_path: Path, total: int) -> None:
    """推理结束后将结果按原始顺序排列，去掉 _idx 辅助字段，覆写文件。"""
    records: dict[int, dict] = {}
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records[obj["_idx"]] = obj

    ordered = [records[i] for i in range(total) if i in records]
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in ordered:
            clean = {"predict": obj["predict"], "label": obj["label"]}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")
    print(f"[INFO] 结果已排序并写入: {out_path}")


def build_prompt(tokenizer, sample: dict, use_strict_prompt: bool = True) -> str:
    """构建 Qwen3 chat prompt。

    Args:
        use_strict_prompt: True（默认）= 使用内置严格格式 prompt，同时关闭 thinking
                           （eval_base_v3，输出更短、更干净，不含 <think> 块）；
                           False = 使用 test.json 原始 instruction，保留 thinking
                           （eval_base_v2 兼容，评测时需加 --strip_think）。
    """
    system_content = STRICT_SYSTEM_PROMPT if use_strict_prompt else sample["instruction"]
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": sample["input"]},
    ]
    # 严格模式下关闭 thinking：模板层面抑制 <think> token，比 prompt 里说"别思考"更可靠；
    # 同时 max_new_tokens 可从 2048 降到 800，推理速度明显提升。
    enable_thinking = not use_strict_prompt
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


# ─────────────────────────── 主流程 ───────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-4B HF 批量推理（支持断点续推，支持 LoRA adapter）")
    parser.add_argument("--model",       type=str, default=DEFAULT_MODEL)
    parser.add_argument("--adapter",     type=str, default=DEFAULT_ADAPTER,
                        help="LoRA adapter 目录（不传则使用基座模型）")
    parser.add_argument("--test",        type=str, default=str(DEFAULT_TEST))
    parser.add_argument("--output_dir",  type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--loose_prompt", action="store_true",
                        help="使用 test.json 原始 instruction（默认使用严格格式 prompt）")
    parser.add_argument("--top_p",       type=float, default=TOP_P)
    parser.add_argument("--repetition_penalty", type=float, default=REPETITION_PENALTY)
    parser.add_argument("--batch_size",  type=int, default=BATCH_SIZE,
                        help="批量推理大小，默认 4")
    parser.add_argument("--save_every",  type=int, default=SAVE_EVERY,
                        help="每处理 N 条打印一次进度")
    parser.add_argument("--restart",     action="store_true",
                        help="忽略已有结果，从头重新推理")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "generated_predictions.jsonl"

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    test_data = load_test_data(Path(args.test))
    total = len(test_data)
    print(f"[INFO] 测试集: {total} 条")

    # ── 断点续推：跳过已完成的样本 ────────────────────────────────────────────
    if args.restart and out_path.exists():
        out_path.unlink()
        print("[INFO] --restart: 已删除旧结果，从头开始")

    done_indices = load_done_indices(out_path)
    remaining = [i for i in range(total) if i not in done_indices]

    if done_indices:
        print(f"[INFO] 断点续推: 已完成 {len(done_indices)} 条，"
              f"剩余 {len(remaining)} 条")
    else:
        print(f"[INFO] 首次运行，共需处理 {total} 条")

    if not remaining:
        print("[INFO] 所有样本已完成，直接进行排序整理")
        reorder_output(out_path, total)
        return

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    print(f"[INFO] 加载模型: {args.model} (BF16)")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.adapter:
        from peft import PeftModel
        print(f"[INFO] 加载 LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()   # 合并权重，推理速度与基座模型完全一致
        print("[INFO] LoRA 已合并到基座模型")

    model.eval()
    print(f"[INFO] 模型加载完成，设备: {next(model.parameters()).device}")

    # 批量推理需要左填充（decoder-only 模型标准做法）
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 批量推理 ──────────────────────────────────────────────────────────────
    t_start    = time.time()
    t_interval = time.time()
    done_count = 0  # 本次运行已完成条数（不含断点续推跳过的）

    use_strict = not args.loose_prompt
    bs = args.batch_size
    model_tag = f"微调模型（adapter={args.adapter}）" if args.adapter else "基座模型"

    print(f"[INFO] 开始推理（batch_size={bs}，每完成 {args.save_every} 条打印进度）...")
    print(f"       max_new_tokens={args.max_new_tokens}, "
          f"temperature={args.temperature}, "
          f"repetition_penalty={args.repetition_penalty}")
    print(f"[INFO] 模型类型: {model_tag}")
    print(f"[INFO] System prompt 模式: {'严格格式 prompt (v3)' if use_strict else '原始 instruction'}")
    print()

    for batch_start in range(0, len(remaining), bs):
        batch_indices = remaining[batch_start: batch_start + bs]
        batch_samples = [test_data[i] for i in batch_indices]
        prompts = [build_prompt(tokenizer, s, use_strict_prompt=use_strict)
                   for s in batch_samples]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]  # 左填充后所有样本同长

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                do_sample=(args.temperature > 0),
                pad_token_id=tokenizer.pad_token_id,
            )

        # 逐条解码并立即保存
        for i, (idx, sample) in enumerate(zip(batch_indices, batch_samples)):
            new_ids = output_ids[i][prompt_len:]
            predict = tokenizer.decode(new_ids, skip_special_tokens=True)
            append_result(out_path, idx, predict, sample["output"])

        done_count += len(batch_indices)
        done_total  = len(done_indices) + done_count

        # 进度打印
        if done_count % args.save_every < bs or done_count == len(remaining):
            elapsed = time.time() - t_start
            intv_t  = time.time() - t_interval
            eta_s   = (elapsed / done_count) * (len(remaining) - done_count)
            print(f"  [{done_total:4d}/{total}] "
                  f"最近批次耗时 {intv_t:.1f}s | "
                  f"总耗时 {elapsed/60:.1f}min | "
                  f"预计剩余 {eta_s/60:.1f}min")
            t_interval = time.time()

    # ── 排序并整理输出文件 ─────────────────────────────────────────────────────
    total_elapsed = time.time() - t_start
    print(f"\n[INFO] 推理完成！共耗时 {total_elapsed/60:.1f} 分钟")
    reorder_output(out_path, total)

    print()
    print("=" * 60)
    print("评测命令（在 my_sft 环境中运行）：")
    print(f"  cd D:\\LLM\\MySFT\\LLM_SFT\\projects\\edge_news_summarizer")
    print(f"  conda activate my_sft")
    needs_strip = not use_strict
    print(f"  python scripts/06_eval_rouge_and_format.py \\")
    print(f"    --predictions {out_path} \\")
    print(f"    --output_dir  {output_dir}" + (" \\" if needs_strip else ""))
    if needs_strip:
        print(f"    --strip_think")


if __name__ == "__main__":
    main()
