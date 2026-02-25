#!/usr/bin/env python3
"""
02_generate_labels_api.py
使用 OpenAI 兼容 API 为原始新闻批量生成结构化摘要标签（SFT 数据）。

特性：
  - 使用 python-dotenv 读取 .env 配置
  - 支持断点续跑（已处理的 ID 自动跳过）
  - 指数退避重试
  - 初步模板校验
  - 每 N 条自动 flush 到磁盘防止中断丢数据
  - 失败记录写入 label_errors.jsonl

输入：data/raw/news_raw.jsonl
输出：data/labeled/news_labeled_v1.jsonl
      data/labeled/label_errors.jsonl

用法：
  python scripts/02_generate_labels_api.py
  python scripts/02_generate_labels_api.py --input data/raw/sample_raw.jsonl --max_samples 50
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# 加载 .env（从项目目录或 LlamaFactory 根目录）
try:
    from dotenv import load_dotenv
    env_path = PROJECT_DIR / ".env"
    if not env_path.exists():
        env_path = PROJECT_DIR.parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    print("[WARN] python-dotenv 未安装，跳过 .env 加载，将从环境变量读取配置。")

DATA_LABELED_DIR = PROJECT_DIR / "data" / "labeled"
DATA_LABELED_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_INPUT = PROJECT_DIR / "data" / "raw" / "news_raw.jsonl"
DEFAULT_OUTPUT = DATA_LABELED_DIR / "news_labeled_v1.jsonl"
ERROR_LOG = DATA_LABELED_DIR / "label_errors.jsonl"
PROMPT_TEMPLATE_FILE = PROJECT_DIR / "data" / "prompts" / "label_prompt_news_structured.txt"

REQUIRED_SECTIONS = [
    "【一句话摘要】",
    "【核心要点】",
    "【事件类别】",
    "【主要主体】",
    "【时间信息】",
    "【潜在影响】",
]


def load_prompt_template() -> str:
    if PROMPT_TEMPLATE_FILE.exists():
        return PROMPT_TEMPLATE_FILE.read_text(encoding="utf-8")
    # 内置兜底模板
    return (
        "你是一位专业的新闻编辑助手，请对以下新闻进行结构化摘要分析。\n\n"
        "新闻标题：{title}\n\n新闻正文：\n{content}\n\n"
        "请严格按照以下格式输出：\n"
        "【一句话摘要】\n（摘要）\n\n"
        "【核心要点】\n1. \n2. \n3. \n\n"
        "【事件类别】\n（类别）\n\n"
        "【主要主体】\n（主体）\n\n"
        "【时间信息】\n（时间）\n\n"
        "【潜在影响】\n（影响）\n"
    )


def validate_label(text: str) -> tuple[bool, list[str]]:
    """检查输出是否包含所有必需的结构字段。"""
    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    return len(missing) == 0, missing


def load_done_ids(output_path: Path) -> set[str]:
    """加载已处理的 ID，用于断点续跑。"""
    done = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "id" in obj:
                        done.add(obj["id"])
                except json.JSONDecodeError:
                    pass
    return done


def call_api_with_retry(client, model: str, messages: list[dict],
                        max_retries: int = 5, base_delay: float = 1.0,
                        timeout: float = 60.0) -> str | None:
    """带指数退避重试的 API 调用。"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=800,
                timeout=timeout,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait = base_delay * (2 ** attempt)
            print(f"[WARN] API 调用失败（第 {attempt + 1} 次）: {e}，{wait:.1f}s 后重试...")
            if attempt < max_retries - 1:
                time.sleep(wait)
    return None


def process_records(records: list[dict], client, model: str, prompt_template: str,
                    output_path: Path, flush_every: int = 10,
                    max_retries: int = 5) -> tuple[int, int]:
    """批量处理新闻记录，生成结构化标签。"""
    success_count = 0
    error_count = 0

    out_f = open(output_path, "a", encoding="utf-8")
    err_f = open(ERROR_LOG, "a", encoding="utf-8")

    try:
        for i, record in enumerate(records, 1):
            record_id = record.get("id", f"unknown_{i}")
            title = record.get("title", "").strip()
            content = record.get("content", "").strip()

            if not content:
                print(f"[WARN] [{i}/{len(records)}] 跳过空正文: {record_id}")
                continue

            # 构建 prompt（截断过长正文）
            content_truncated = content[:3000] if len(content) > 3000 else content
            prompt = prompt_template.format(title=title, content=content_truncated)

            messages = [{"role": "user", "content": prompt}]

            print(f"[INFO] [{i}/{len(records)}] 处理: {record_id} (标题: {title[:40]}...)")

            label_text = call_api_with_retry(client, model, messages, max_retries=max_retries)

            if label_text is None:
                print(f"[ERROR] [{i}/{len(records)}] API 失败: {record_id}")
                err_record = {**record, "error": "api_failed", "attempt": max_retries}
                err_f.write(json.dumps(err_record, ensure_ascii=False) + "\n")
                error_count += 1
                continue

            # 模板校验
            is_valid, missing = validate_label(label_text)
            if not is_valid:
                print(f"[WARN] [{i}/{len(records)}] 格式缺失字段 {missing}: {record_id}")
                err_record = {**record, "error": "format_invalid", "missing": missing,
                              "raw_label": label_text}
                err_f.write(json.dumps(err_record, ensure_ascii=False) + "\n")
                error_count += 1
                continue

            # 构建 SFT 格式（Alpaca instruction 格式）
            output_record = {
                "id": record_id,
                "instruction": prompt_template.split("\n\n新闻标题")[0],  # 系统提示
                "input": f"新闻标题：{title}\n\n新闻正文：\n{content_truncated}",
                "output": label_text,
                "source": record.get("source", ""),
                "date": record.get("date", ""),
            }
            out_f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            success_count += 1

            # 定期 flush
            if i % flush_every == 0:
                out_f.flush()
                err_f.flush()
                print(f"[INFO] Checkpoint: {success_count} 成功 / {error_count} 失败")

    finally:
        out_f.flush()
        err_f.flush()
        out_f.close()
        err_f.close()

    return success_count, error_count


def main():
    parser = argparse.ArgumentParser(description="使用 API 生成结构化新闻摘要标签")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                        help=f"输入 JSONL 文件（默认: {DEFAULT_INPUT}）")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help=f"输出 JSONL 文件（默认: {DEFAULT_OUTPUT}）")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="最大处理样本数（0 表示全部）")
    parser.add_argument("--flush_every", type=int, default=10,
                        help="每 N 条写入磁盘（默认: 10）")
    parser.add_argument("--max_retries", type=int, default=5,
                        help="API 失败重试次数（默认: 5）")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="断点续跑（跳过已处理 ID，默认开启）")
    parser.add_argument("--no_resume", dest="resume", action="store_false",
                        help="不使用断点续跑，重新处理所有记录")
    args = parser.parse_args()

    # 读取 API 配置
    api_key = os.environ.get("OPENAI_API_KEY", "")
    api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        print("[ERROR] 未找到 OPENAI_API_KEY，请在 .env 文件或环境变量中设置。", file=sys.stderr)
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("[ERROR] 请先安装 openai: pip install openai", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=api_base)
    print(f"[INFO] 使用模型: {model} (base_url: {api_base})")

    # 读取输入数据
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 输入文件不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.max_samples > 0:
        records = records[:args.max_samples]

    print(f"[INFO] 读取 {len(records)} 条原始新闻")

    # 断点续跑
    output_path = Path(args.output)
    if args.resume:
        done_ids = load_done_ids(output_path)
        if done_ids:
            print(f"[INFO] 断点续跑：跳过已处理 {len(done_ids)} 条")
            records = [r for r in records if r.get("id") not in done_ids]
            print(f"[INFO] 剩余待处理: {len(records)} 条")

    if not records:
        print("[INFO] 所有记录已处理完毕。")
        return

    # 加载 prompt 模板
    prompt_template = load_prompt_template()
    print(f"[INFO] 加载 prompt 模板（{len(prompt_template)} 字符）")

    # 处理
    success, errors = process_records(
        records, client, model, prompt_template, output_path,
        flush_every=args.flush_every, max_retries=args.max_retries
    )

    print(f"\n[INFO] 完成！成功: {success}，失败: {errors}")
    print(f"[INFO] 输出文件: {output_path}")
    print(f"[INFO] 错误日志: {ERROR_LOG}")


if __name__ == "__main__":
    main()
