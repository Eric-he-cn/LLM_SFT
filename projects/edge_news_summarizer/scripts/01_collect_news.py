#!/usr/bin/env python3
"""
01_collect_news.py
新闻数据收集脚本：支持从本地 JSONL 文件或 HuggingFace Dataset 导入新闻，
输出统一格式到 data/raw/news_raw.jsonl。

统一输出格式（每行一个 JSON）：
{
  "id": "唯一ID",
  "title": "新闻标题",
  "content": "新闻正文",
  "source": "数据来源",
  "date": "发布日期（可选）"
}

用法：
  # 从本地 JSONL 导入
  python scripts/01_collect_news.py --source local --input path/to/news.jsonl

  # 从 HuggingFace 导入（CNN/DailyMail）
  python scripts/01_collect_news.py --source hf --dataset cnn_dailymail --split train --max_samples 5000

  # 查看样本
  python scripts/01_collect_news.py --source hf --dataset cnn_dailymail --split train --max_samples 100 --preview
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_RAW_DIR = PROJECT_DIR / "data" / "raw"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_RAW_DIR / "news_raw.jsonl"
SAMPLE_FILE = DATA_RAW_DIR / "sample_raw.jsonl"


def make_id(prefix: str = "news") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def collect_from_local(input_path: str, title_key: str, content_key: str,
                       date_key: str, max_samples: int) -> list[dict]:
    """从本地 JSONL 文件收集新闻。"""
    records = []
    path = Path(input_path)
    if not path.exists():
        print(f"[ERROR] 文件不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    suffix = path.suffix.lower()
    with open(path, encoding="utf-8") as f:
        if suffix == ".jsonl":
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                records.append(obj)
                if max_samples and len(records) >= max_samples:
                    break
        elif suffix == ".json":
            data = json.load(f)
            if isinstance(data, list):
                records = data[:max_samples] if max_samples else data
            else:
                records = [data]
        else:
            print(f"[ERROR] 不支持的文件格式: {suffix}（支持 .jsonl / .json）", file=sys.stderr)
            sys.exit(1)

    normalized = []
    for i, obj in enumerate(records):
        title = obj.get(title_key, obj.get("title", "")).strip()
        content = obj.get(content_key, obj.get("content", obj.get("text", obj.get("body", "")))).strip()
        date = obj.get(date_key, obj.get("date", obj.get("publish_date", ""))).strip() if date_key else ""
        if not title and not content:
            continue
        normalized.append({
            "id": obj.get("id", make_id("local")),
            "title": title,
            "content": content,
            "source": "local",
            "date": date,
        })

    return normalized


def collect_from_hf(dataset_name: str, split: str, config: str,
                    max_samples: int) -> list[dict]:
    """从 HuggingFace 数据集收集新闻。"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] 请先安装 datasets: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 加载 HuggingFace 数据集: {dataset_name} (split={split}, config={config})")
    kwargs = {}
    if config:
        kwargs["name"] = config
    if max_samples:
        kwargs["split"] = f"{split}[:{max_samples}]"
    else:
        kwargs["split"] = split

    ds = load_dataset(dataset_name, **kwargs, trust_remote_code=True)

    records = []
    # 自动探测字段名
    col_names = ds.column_names
    print(f"[INFO] 数据集字段: {col_names}")

    # CNN/DailyMail 字段映射
    if "article" in col_names and "highlights" in col_names:
        for i, row in enumerate(ds):
            records.append({
                "id": make_id("hf"),
                "title": row.get("id", f"article_{i}"),  # CNN/DM 没有标题，用 id 代替
                "content": row["article"].strip(),
                "source": dataset_name,
                "date": "",
            })
    elif "title" in col_names and "text" in col_names:
        for row in ds:
            records.append({
                "id": make_id("hf"),
                "title": row["title"].strip(),
                "content": row["text"].strip(),
                "source": dataset_name,
                "date": row.get("date", ""),
            })
    elif "title" in col_names and "content" in col_names:
        for row in ds:
            records.append({
                "id": make_id("hf"),
                "title": row["title"].strip(),
                "content": row["content"].strip(),
                "source": dataset_name,
                "date": row.get("date", row.get("publish_date", "")),
            })
    else:
        # 通用兜底：取前两个文本字段
        text_cols = [c for c in col_names if isinstance(ds[0].get(c), str)]
        if len(text_cols) < 1:
            print("[ERROR] 无法自动映射数据集字段，请修改脚本手动指定。", file=sys.stderr)
            sys.exit(1)
        title_col = text_cols[0]
        content_col = text_cols[1] if len(text_cols) > 1 else text_cols[0]
        for row in ds:
            records.append({
                "id": make_id("hf"),
                "title": row[title_col].strip(),
                "content": row[content_col].strip(),
                "source": dataset_name,
                "date": "",
            })

    return records


def filter_records(records: list[dict], min_content_len: int = 100,
                   max_content_len: int = 8000) -> tuple[list[dict], int]:
    """过滤过短/过长的正文。"""
    filtered = []
    dropped = 0
    for r in records:
        clen = len(r.get("content", ""))
        if clen < min_content_len or clen > max_content_len:
            dropped += 1
            continue
        filtered.append(r)
    return filtered, dropped


def save_records(records: list[dict], output_path: Path, preview: bool = False,
                 preview_count: int = 3) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[INFO] 已保存 {len(records)} 条记录到 {output_path}")

    if preview:
        print("\n===== 前几条样本预览 =====")
        for r in records[:preview_count]:
            print(json.dumps(r, ensure_ascii=False, indent=2))
            print("-" * 40)


def save_sample(records: list[dict], sample_count: int = 20) -> None:
    sample = records[:sample_count]
    with open(SAMPLE_FILE, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[INFO] 样本文件已保存: {SAMPLE_FILE} ({len(sample)} 条)")


def main():
    parser = argparse.ArgumentParser(description="新闻数据收集脚本")
    parser.add_argument("--source", choices=["local", "hf"], required=True,
                        help="数据来源：local（本地文件）或 hf（HuggingFace）")

    # 本地文件参数
    parser.add_argument("--input", type=str, help="本地 JSONL/JSON 文件路径（source=local 时必填）")
    parser.add_argument("--title_key", default="title", help="标题字段名（默认: title）")
    parser.add_argument("--content_key", default="content", help="正文字段名（默认: content）")
    parser.add_argument("--date_key", default="date", help="日期字段名（默认: date）")

    # HuggingFace 参数
    parser.add_argument("--dataset", default="cnn_dailymail", help="HuggingFace 数据集名（默认: cnn_dailymail）")
    parser.add_argument("--config", default="3.0.0", help="数据集配置名（CNN/DM 需要 3.0.0）")
    parser.add_argument("--split", default="train", help="数据集分割（默认: train）")

    # 通用参数
    parser.add_argument("--max_samples", type=int, default=5000, help="最大样本数（默认: 5000）")
    parser.add_argument("--min_content_len", type=int, default=100, help="最小正文长度（默认: 100）")
    parser.add_argument("--max_content_len", type=int, default=8000, help="最大正文长度（默认: 8000）")
    parser.add_argument("--output", type=str, help=f"输出文件路径（默认: {OUTPUT_FILE}）")
    parser.add_argument("--preview", action="store_true", help="预览前几条样本")
    parser.add_argument("--no_sample", action="store_true", help="不生成 sample_raw.jsonl")

    args = parser.parse_args()

    output_path = Path(args.output) if args.output else OUTPUT_FILE

    # 收集数据
    if args.source == "local":
        if not args.input:
            parser.error("--source local 时必须指定 --input 参数")
        records = collect_from_local(args.input, args.title_key, args.content_key,
                                     args.date_key, args.max_samples)
    else:
        records = collect_from_hf(args.dataset, args.split, args.config, args.max_samples)

    print(f"[INFO] 收集原始记录: {len(records)} 条")

    # 过滤
    records, dropped = filter_records(records, args.min_content_len, args.max_content_len)
    print(f"[INFO] 过滤后: {len(records)} 条（丢弃 {dropped} 条）")

    if not records:
        print("[ERROR] 过滤后没有任何数据，请检查参数。", file=sys.stderr)
        sys.exit(1)

    # 保存
    save_records(records, output_path, preview=args.preview)
    if not args.no_sample:
        save_sample(records)

    print("[INFO] 数据收集完成！")


if __name__ == "__main__":
    main()
