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
  "lang": "语言标识（zh/en）",
  "date": "发布日期（可选）"
}

数据集方案：XL-Sum 中英混合
  - 中文：csebuetnlp/xlsum, config=chinese_simplified（BBC中文，专业摘要）
  - 英文：csebuetnlp/xlsum, config=english（BBC英文，专业摘要）
  - 各取 max_samples//2 条，混合后统一由 API 生成中文结构化摘要

用法：
  # XL-Sum 中英混合（推荐，默认方案）
  python scripts/01_collect_news.py --source xlsum --max_samples 5000

  # 仅中文
  python scripts/01_collect_news.py --source xlsum --lang zh --max_samples 5000

  # 仅英文
  python scripts/01_collect_news.py --source xlsum --lang en --max_samples 5000

  # 从本地 JSONL 导入
  python scripts/01_collect_news.py --source local --input path/to/news.jsonl

  # 预览样本（不保存）
  python scripts/01_collect_news.py --source xlsum --max_samples 10 --preview
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


# ---- 中国/香港/台湾政治及社会运动过滤 ----
# 包含以下关键词的中文或英文新闻将被排除，避免触发 API 审查且与摘要任务无关
ZH_POLITICAL_KEYWORDS = [
    # === 中国大陆政治领导 / 机构 ===
    "习近平", "李克强", "李强", "王岐山", "赵乐际", "王沪宁", "王毅",
    "政治局常委", "中共中央", "中央政治局", "中央纪委", "国家监委",
    "十八大", "十九大", "二十大", "党代会", "党代表大会",
    "全国人大", "全国政协", "人大常委",
    # === 香港政治人物 / 机构 ===
    "梁振英", "林郑月娥", "李家超", "特首", "行政长官",
    "建制派", "泛民", "民主派", "民建联", "公民党",
    "立法会", "施政报告", "国安法", "基本法第二十三条",
    # === 香港政治抗议 / 社会运动 ===
    "占领中环", "占中", "雨伞运动", "雨伞革命",
    "反修例", "反国教", "港独", "罢课",
    "示威学生", "示威者", "抗议学生",
    # === 台湾政治 ===
    "台独", "台湾独立", "太阳花", "民进党", "国民党",
    "蔡英文", "韩国瑜", "柯文哲", "赖清德",
    "总统大选", "立法委员选举", "台湾选举",
    # === 新疆 / 西藏独立运动 ===
    "藏独", "西藏独立", "疆独", "新疆独立", "东突",
    # === 敏感历史事件 / 人物 ===
    "天安门", "六四", "刘晓波", "艾未未", "法轮功",
    "维权律师", "709大抓捕", "709案",
    # === 政治迫害 / 异见 ===
    "政治犯", "异见人士", "被失踪", "政治迫害", "一党专政",
    # === 宗教政治 ===
    "达赖喇嘛", "班禅喇嘛",
    # === 中国社会运动 ===
    "茉莉花革命", "劳工维权", "上访者", "维稳",
    # === 领土主权争议 ===
    "钓鱼岛", "南海仲裁", "九段线",
]

# 同步过滤英文文章中涉及中国政治核心词
EN_POLITICAL_KEYWORDS = [
    "tiananmen", "june 4", "falun gong", "falun dafa",
    "uyghur detention", "xinjiang camp", "hong kong protest",
    "umbrella revolution", "umbrella movement",
    "taiwan independence", "tibet independence",
    "xi jinping crackdown", "chinese dissident",
    "709 crackdown", "liu xiaobo",
]


def is_political(title: str, content: str, lang: str) -> bool:
    """判断新闻是否为政治敏感内容（中英文均检查）。"""
    text_short = (title + content[:400]).lower() if lang == "en" else title + content[:400]
    if lang == "zh":
        return any(kw in text_short for kw in ZH_POLITICAL_KEYWORDS)
    else:
        return any(kw in text_short for kw in EN_POLITICAL_KEYWORDS)


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


def collect_from_xlsum(lang: str, max_samples: int) -> list[dict]:
    """从 XL-Sum 数据集收集新闻（支持 zh/en）。

    XL-Sum 字段：title, text, summary, url, id
    lang: 'zh' -> chinese_simplified, 'en' -> english
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] 请先安装 datasets: pip install datasets", file=sys.stderr)
        sys.exit(1)

    config_map = {"zh": "chinese_simplified", "en": "english"}
    config = config_map.get(lang)
    if not config:
        print(f"[ERROR] 不支持的语言: {lang}，可选 zh / en", file=sys.stderr)
        sys.exit(1)

    split_str = f"train[:{max_samples}]" if max_samples else "train"
    print(f"[INFO] 加载 XL-Sum ({config}, {split_str})")

    # XL-Sum 使用旧版加载脚本，datasets 4.x 不兼容，改用 parquet 直接加载
    HF_BASE = "hf://datasets/csebuetnlp/xlsum@refs/convert/parquet"
    data_files = {
        "train": f"{HF_BASE}/{config}/train/*.parquet",
        "validation": f"{HF_BASE}/{config}/validation/*.parquet",
        "test": f"{HF_BASE}/{config}/test/*.parquet",
    }
    ds = load_dataset("parquet", data_files=data_files, split=split_str)
    print(f"[INFO] 加载完成，共 {len(ds)} 条，字段: {ds.column_names}")

    records = []
    skipped_political = 0
    for row in ds:
        title = (row.get("title") or "").strip()
        content = (row.get("text") or "").strip()
        if not content:
            continue
        # 过滤政治新闻（中英文均检查）
        if is_political(title, content, lang):
            skipped_political += 1
            continue
        records.append({
            "id": make_id(f"xlsum_{lang}"),
            "title": title,
            "content": content,
            "source": f"xlsum_{lang}",
            "lang": lang,
            "date": "",
        })
    if skipped_political:
        print(f"[INFO] 已过滤中文政治新闻: {skipped_political} 条")
    return records


def collect_from_xlsum_mixed(max_samples: int) -> list[dict]:
    """从 XL-Sum 收集中英混合数据，各取 max_samples//2 条。"""
    half = max_samples // 2
    zh_records = collect_from_xlsum("zh", half)
    en_records = collect_from_xlsum("en", half)

    # 交错混合（中英交替），避免批次偏斜
    mixed = []
    for zh, en in zip(zh_records, en_records):
        mixed.append(zh)
        mixed.append(en)
    # 补齐奇数余量
    for r in zh_records[len(en_records):] + en_records[len(zh_records):]:
        mixed.append(r)

    print(f"[INFO] 混合完成：中文 {len(zh_records)} 条 + 英文 {len(en_records)} 条 = {len(mixed)} 条")
    return mixed


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
    parser = argparse.ArgumentParser(description="新闻数据收集脚本（XL-Sum 中英混合方案）")
    parser.add_argument("--source", choices=["xlsum", "local"], default="xlsum",
                        help="数据来源：xlsum（XL-Sum，默认）或 local（本地文件）")
    parser.add_argument("--lang", choices=["zh", "en", "mixed"], default="mixed",
                        help="xlsum 语言：zh / en / mixed（默认 mixed，中英各半）")

    # 本地文件参数
    parser.add_argument("--input", type=str, help="本地 JSONL/JSON 文件路径（source=local 时必填）")
    parser.add_argument("--title_key", default="title", help="标题字段名（默认: title）")
    parser.add_argument("--content_key", default="content", help="正文字段名（默认: content）")
    parser.add_argument("--date_key", default="date", help="日期字段名（默认: date）")

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
    else:  # xlsum
        if args.lang == "mixed":
            records = collect_from_xlsum_mixed(args.max_samples)
        else:
            records = collect_from_xlsum(args.lang, args.max_samples)

    print(f"[INFO] 收集原始记录: {len(records)} 条")

    # 过滤
    records, dropped = filter_records(records, args.min_content_len, args.max_content_len)
    print(f"[INFO] 过滤后: {len(records)} 条（丢弃 {dropped} 条）")

    if not records:
        print("[ERROR] 过滤后没有任何数据，请检查参数。", file=sys.stderr)
        sys.exit(1)

    # 统计语言分布
    zh_count = sum(1 for r in records if r.get("lang") == "zh")
    en_count = sum(1 for r in records if r.get("lang") == "en")
    if zh_count or en_count:
        print(f"[INFO] 语言分布：中文 {zh_count} 条 / 英文 {en_count} 条")

    # 保存
    save_records(records, output_path, preview=args.preview)
    if not args.no_sample:
        save_sample(records)

    print("[INFO] 数据收集完成！")


if __name__ == "__main__":
    main()
