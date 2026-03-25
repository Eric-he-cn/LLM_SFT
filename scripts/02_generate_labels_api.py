#!/usr/bin/env python3
"""
02_generate_labels_api.py
使用 OpenAI 兼容 API 为原始新闻批量生成结构化摘要标签（SFT 数据）。

特性：
  - asyncio 并发请求（--concurrency 控制，默认 5）
  - 使用 python-dotenv 读取 .env 配置
  - 支持断点续跑（已处理的 ID 自动跳过）
  - Content Exists Risk 直接跳过，不重试
  - 指数退避重试（其他错误）
  - 初步模板校验
  - asyncio.Lock 保护文件写入

输入：data/raw/news_raw.jsonl
输出：data/labeled/news_labeled_v1.jsonl
      data/labeled/label_errors.jsonl

用法：
  python scripts/02_generate_labels_api.py
  python scripts/02_generate_labels_api.py --max_samples 2000 --concurrency 5
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

try:
    from dotenv import load_dotenv
    env_path = PROJECT_DIR / ".env"
    if not env_path.exists():
        env_path = PROJECT_DIR.parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    print("[WARN] python-dotenv 未安装，跳过 .env 加载。")

DATA_LABELED_DIR = PROJECT_DIR / "data" / "labeled"
DATA_LABELED_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_INPUT  = PROJECT_DIR / "data" / "raw" / "news_raw.jsonl"
DEFAULT_OUTPUT = DATA_LABELED_DIR / "news_labeled_v1.jsonl"
ERROR_LOG      = DATA_LABELED_DIR / "label_errors.jsonl"
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
    return (
        "你是一位专业的新闻编辑助手，请对以下新闻进行结构化摘要分析。\n\n"
        "新闻标题：{title}\n\n新闻正文：\n{content}\n\n"
        "请严格按照以下格式输出（无论输入是中文还是英文，请始终用中文输出）：\n"
        "【一句话摘要】\n（摘要）\n\n【核心要点】\n1. \n2. \n3. \n\n"
        "【事件类别】\n（类别）\n\n【主要主体】\n（主体）\n\n"
        "【时间信息】\n（时间）\n\n【潜在影响】\n（影响）\n"
    )


def validate_label(text: str) -> tuple[bool, list[str]]:
    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    return len(missing) == 0, missing


def load_done_ids(output_path: Path) -> set[str]:
    done = set()
    if not output_path.exists():
        return done
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


async def call_api_async(client, model: str, messages: list[dict],
                         max_retries: int = 3, base_delay: float = 1.0,
                         timeout: float = 60.0) -> str | None:
    """异步 API 调用，Content Exists Risk 直接跳过。"""
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=800,
                timeout=timeout,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e)
            if "Content Exists Risk" in err_str or ("400" in err_str and "content" in err_str.lower()):
                print(f"[SKIP] 内容审查拒绝，跳过")
                return None
            wait = base_delay * (2 ** attempt)
            print(f"[WARN] API 失败（第 {attempt + 1} 次）: {e}，{wait:.1f}s 后重试...")
            if attempt < max_retries - 1:
                await asyncio.sleep(wait)
    return None


async def process_records_async(
    records: list[dict],
    api_key: str,
    api_base: str,
    model: str,
    prompt_template: str,
    output_path: Path,
    concurrency: int = 5,
    max_retries: int = 3,
) -> tuple[int, int]:
    """并发处理新闻记录。"""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    sem        = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    total      = len(records)
    counters   = {"success": 0, "error": 0, "done": 0}

    instr = prompt_template.split("\n\n新闻标题")[0]

    out_f = open(output_path, "a", encoding="utf-8")
    err_f = open(ERROR_LOG,   "a", encoding="utf-8")

    async def process_one(i: int, record: dict) -> None:
        record_id = record.get("id", f"unknown_{i}")
        title     = record.get("title", "").strip()
        content   = record.get("content", "").strip()

        if not content:
            return

        content_truncated = content[:3000]
        prompt   = prompt_template.format(title=title, content=content_truncated)
        messages = [{"role": "user", "content": prompt}]

        async with sem:
            label_text = await call_api_async(client, model, messages, max_retries=max_retries)

        async with write_lock:
            counters["done"] += 1
            idx = counters["done"]
            print(f"[INFO] [{idx}/{total}] {record_id}  {title[:38]}...")

            if label_text is None:
                counters["error"] += 1
                err_f.write(json.dumps({**record, "error": "api_failed"}, ensure_ascii=False) + "\n")
                err_f.flush()
                return

            is_valid, missing = validate_label(label_text)
            if not is_valid:
                print(f"[WARN] 格式缺失 {missing}: {record_id}")
                counters["error"] += 1
                err_f.write(json.dumps({**record, "error": "format_invalid",
                                        "missing": missing, "raw_label": label_text},
                                       ensure_ascii=False) + "\n")
                err_f.flush()
                return

            output_record = {
                "id":          record_id,
                "instruction": instr,
                "input":       f"新闻标题：{title}\n\n新闻正文：\n{content_truncated}",
                "output":      label_text,
                "source":      record.get("source", ""),
                "date":        record.get("date", ""),
                "lang":        record.get("lang", ""),
            }
            out_f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            out_f.flush()
            counters["success"] += 1

            if counters["done"] % 50 == 0:
                print(f"[INFO] === Checkpoint {counters['done']}/{total}："
                      f" {counters['success']} 成功 / {counters['error']} 失败 ===")

    try:
        await asyncio.gather(*[process_one(i + 1, r) for i, r in enumerate(records)])
    finally:
        out_f.flush(); out_f.close()
        err_f.flush(); err_f.close()

    return counters["success"], counters["error"]


def main():
    parser = argparse.ArgumentParser(description="并发 API 标注结构化新闻摘要")
    parser.add_argument("--input",       type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output",      type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max_samples", type=int, default=0,  help="0=全部")
    parser.add_argument("--concurrency", type=int, default=5,  help="并发请求数（默认 5）")
    parser.add_argument("--max_retries", type=int, default=3,  help="重试次数（默认 3）")
    parser.add_argument("--resume",    action="store_true",  default=True)
    parser.add_argument("--no_resume", action="store_false", dest="resume")
    args = parser.parse_args()

    api_key  = os.environ.get("OPENAI_API_KEY", "")
    api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    model    = os.environ.get("OPENAI_MODEL",    "gpt-4o-mini")

    if not api_key:
        print("[ERROR] 未找到 OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    try:
        import openai  # noqa: F401
    except ImportError:
        print("[ERROR] 请先安装 openai: pip install openai", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 模型: {model}  并发: {args.concurrency}  base_url: {api_base}")

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

    output_path = Path(args.output)
    if args.resume:
        done_ids = load_done_ids(output_path)
        if done_ids:
            records = [r for r in records if r.get("id") not in done_ids]
            print(f"[INFO] 断点续跑：跳过已处理 {len(done_ids)} 条，剩余 {len(records)} 条")

    if not records:
        print("[INFO] 所有记录已处理完毕。")
        return

    prompt_template = load_prompt_template()
    print(f"[INFO] Prompt 模板 {len(prompt_template)} 字符")

    t0 = time.time()
    success, errors = asyncio.run(process_records_async(
        records, api_key, api_base, model, prompt_template, output_path,
        concurrency=args.concurrency, max_retries=args.max_retries,
    ))
    elapsed = time.time() - t0

    print(f"\n[INFO] 完成！成功: {success}，失败: {errors}，耗时: {elapsed/60:.1f} 分钟")
    print(f"[INFO] 输出: {output_path}")
    print(f"[INFO] 错误日志: {ERROR_LOG}")


if __name__ == "__main__":
    main()
