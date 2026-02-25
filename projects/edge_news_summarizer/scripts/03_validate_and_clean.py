#!/usr/bin/env python3
"""
03_validate_and_clean.py
对打标完成的数据进行格式校验、类别检查、去重、长度过滤，输出清洗后的数据。

校验项：
  1. 必须包含 6 个结构字段
  2. "事件类别"必须在白名单中
  3. "核心要点"至少有 3 条编号列表
  4. 输入（instruction + input）和输出长度在合理范围内
  5. 按 instruction+input 去重

输入：data/labeled/news_labeled_v1.jsonl
输出：data/cleaned/train.json（含所有清洗后数据）
      data/reports/data_quality_report.md
      data/reports/labeling_stats.json

用法：
  python scripts/03_validate_and_clean.py
  python scripts/03_validate_and_clean.py --input data/labeled/news_labeled_v1.jsonl --strict
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_CLEANED_DIR = PROJECT_DIR / "data" / "cleaned"
DATA_REPORTS_DIR = PROJECT_DIR / "data" / "reports"
DATA_CLEANED_DIR.mkdir(parents=True, exist_ok=True)
DATA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_INPUT = PROJECT_DIR / "data" / "labeled" / "news_labeled_v1.jsonl"
DEFAULT_OUTPUT = DATA_CLEANED_DIR / "cleaned_all.jsonl"

REQUIRED_SECTIONS = [
    "【一句话摘要】",
    "【核心要点】",
    "【事件类别】",
    "【主要主体】",
    "【时间信息】",
    "【潜在影响】",
]

VALID_CATEGORIES = {
    "科技", "财经", "政治", "社会", "体育", "文化", "国际", "军事", "环境", "健康",
    # 英文别名（兼容 CNN/DailyMail 生成结果）
    "technology", "finance", "politics", "society", "sports", "culture",
    "international", "military", "environment", "health",
}


def check_sections(output_text: str) -> tuple[bool, list[str]]:
    """检查是否包含所有必需字段。"""
    missing = [s for s in REQUIRED_SECTIONS if s not in output_text]
    return len(missing) == 0, missing


def extract_category(output_text: str) -> str:
    """从输出中提取事件类别。"""
    pattern = r"【事件类别】\s*\n?\s*([^\n【]+)"
    match = re.search(pattern, output_text)
    if match:
        return match.group(1).strip()
    return ""


def check_category(category: str) -> bool:
    """检查类别是否在白名单中（宽松匹配）。"""
    cat_lower = category.lower().strip()
    for valid in VALID_CATEGORIES:
        if valid.lower() in cat_lower or cat_lower in valid.lower():
            return True
    return False


def check_bullet_points(output_text: str, min_points: int = 3) -> tuple[bool, int]:
    """检查核心要点是否有足够的编号条目。"""
    # 提取核心要点部分
    pattern = r"【核心要点】(.*?)(?:【|$)"
    match = re.search(pattern, output_text, re.DOTALL)
    if not match:
        return False, 0

    section = match.group(1)
    # 匹配编号行（1. 或 1、）
    bullets = re.findall(r"^\s*\d+[\.、．]\s*.+", section, re.MULTILINE)
    return len(bullets) >= min_points, len(bullets)


def deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    """按 input 字段去重。"""
    seen = set()
    unique = []
    dup_count = 0
    for r in records:
        key = r.get("input", "") + r.get("instruction", "")
        key_hash = hash(key)
        if key_hash in seen:
            dup_count += 1
            continue
        seen.add(key_hash)
        unique.append(r)
    return unique, dup_count


def validate_record(record: dict, strict: bool = False,
                    min_input_len: int = 50, max_input_len: int = 4000,
                    min_output_len: int = 100, max_output_len: int = 2000) -> tuple[bool, list[str]]:
    """
    完整校验单条记录。
    返回：(是否通过, 错误原因列表)
    """
    errors = []
    output_text = record.get("output", "")
    input_text = record.get("input", "")

    # 1. 长度检查
    if len(input_text) < min_input_len:
        errors.append(f"input_too_short({len(input_text)})")
    if len(input_text) > max_input_len:
        errors.append(f"input_too_long({len(input_text)})")
    if len(output_text) < min_output_len:
        errors.append(f"output_too_short({len(output_text)})")
    if len(output_text) > max_output_len:
        errors.append(f"output_too_long({len(output_text)})")

    # 2. 结构字段检查
    sections_ok, missing = check_sections(output_text)
    if not sections_ok:
        errors.append(f"missing_sections:{','.join(missing)}")

    # 3. 类别检查
    category = extract_category(output_text)
    if not category:
        errors.append("no_category")
    elif strict and not check_category(category):
        errors.append(f"invalid_category:'{category}'")

    # 4. 要点数量检查
    bullets_ok, bullet_count = check_bullet_points(output_text)
    if not bullets_ok:
        errors.append(f"insufficient_bullets({bullet_count})")

    return len(errors) == 0, errors


def generate_report(stats: dict, output_path: Path) -> None:
    """生成 Markdown 格式的数据质量报告。"""
    total = stats["total"]
    passed = stats["passed"]
    report = f"""# 数据质量报告

## 概览

| 指标 | 数量 | 比例 |
|------|------|------|
| 原始记录 | {total} | 100% |
| 通过校验 | {passed} | {passed/total*100:.1f}% if total > 0 else 0% |
| 去重丢弃 | {stats['duplicates']} | {stats['duplicates']/total*100:.1f}% if total > 0 else 0% |

## 错误分布

| 错误类型 | 数量 |
|----------|------|
"""
    for err_type, count in sorted(stats["error_counts"].items(), key=lambda x: -x[1]):
        report += f"| {err_type} | {count} |\n"

    report += f"""
## 类别分布

| 类别 | 数量 |
|------|------|
"""
    for cat, count in sorted(stats["category_dist"].items(), key=lambda x: -x[1]):
        report += f"| {cat} | {count} |\n"

    output_path.write_text(report, encoding="utf-8")
    print(f"[INFO] 质量报告已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="数据校验与清洗脚本")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--strict", action="store_true", help="严格模式（类别必须在白名单中）")
    parser.add_argument("--min_input_len", type=int, default=50)
    parser.add_argument("--max_input_len", type=int, default=4000)
    parser.add_argument("--min_output_len", type=int, default=100)
    parser.add_argument("--max_output_len", type=int, default=2000)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 输入文件不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    # 读取数据
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[INFO] 读取 {len(records)} 条记录")

    # 去重
    records, dup_count = deduplicate(records)
    print(f"[INFO] 去重后: {len(records)} 条（丢弃重复 {dup_count} 条）")

    # 校验
    passed = []
    error_counts = defaultdict(int)
    category_dist = defaultdict(int)
    failed_records = []

    for record in records:
        ok, errors = validate_record(
            record, strict=args.strict,
            min_input_len=args.min_input_len,
            max_input_len=args.max_input_len,
            min_output_len=args.min_output_len,
            max_output_len=args.max_output_len,
        )
        if ok:
            passed.append(record)
            cat = extract_category(record.get("output", ""))
            category_dist[cat] += 1
        else:
            for err in errors:
                err_type = err.split("(")[0].split(":")[0]
                error_counts[err_type] += 1
            failed_records.append({**record, "_errors": errors})

    print(f"[INFO] 通过校验: {len(passed)} 条 / 失败: {len(failed_records)} 条")

    # 保存清洗后数据（Alpaca 格式，LLaMA-Factory 标准）
    output_path = Path(args.output)
    alpaca_records = []
    for r in passed:
        alpaca_records.append({
            "instruction": r.get("instruction", "你是一位专业的新闻编辑助手，请对新闻进行结构化摘要分析。"),
            "input": r.get("input", ""),
            "output": r.get("output", ""),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for r in alpaca_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[INFO] 清洗数据已保存: {output_path}")

    # 统计信息
    stats = {
        "total": len(records) + dup_count,
        "passed": len(passed),
        "duplicates": dup_count,
        "error_counts": dict(error_counts),
        "category_dist": dict(category_dist),
    }

    stats_path = DATA_REPORTS_DIR / "labeling_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] 统计信息已保存: {stats_path}")

    report_path = DATA_REPORTS_DIR / "data_quality_report.md"
    generate_report(stats, report_path)

    # 打印摘要
    total = stats["total"]
    print(f"\n===== 数据清洗摘要 =====")
    print(f"原始记录: {total}")
    print(f"通过率: {len(passed)/max(total,1)*100:.1f}%")
    print(f"错误类型分布: {dict(error_counts)}")


if __name__ == "__main__":
    main()
