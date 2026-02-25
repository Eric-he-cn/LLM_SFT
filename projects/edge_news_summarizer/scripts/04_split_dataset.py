#!/usr/bin/env python3
"""
04_split_dataset.py
将清洗后的数据集划分为 train / val / test 三个子集。

输入：data/cleaned/cleaned_all.jsonl（03 脚本输出）
输出：
  data/cleaned/train.json
  data/cleaned/val.json
  data/cleaned/test.json
  data/cleaned/test_manual_eval.json（随机抽取 100 条用于人工评测）

用法：
  python scripts/04_split_dataset.py
  python scripts/04_split_dataset.py --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
"""

import argparse
import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_CLEANED_DIR = PROJECT_DIR / "data" / "cleaned"
DATA_CLEANED_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_INPUT = DATA_CLEANED_DIR / "cleaned_all.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 已保存 {len(records)} 条到 {path}")


def save_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[INFO] 已保存 {len(records)} 条到 {path}")


def split_dataset(records: list[dict], train_ratio: float, val_ratio: float,
                  test_ratio: float, seed: int) -> tuple[list, list, list]:
    """按比例划分数据集。"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为 1.0"

    random.seed(seed)
    shuffled = records.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]

    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="数据集划分脚本")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例（默认: 0.8）")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例（默认: 0.1）")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例（默认: 0.1）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认: 42）")
    parser.add_argument("--manual_eval_count", type=int, default=100,
                        help="人工评测集大小（默认: 100）")
    args = parser.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        print("[ERROR] train_ratio + val_ratio + test_ratio 必须等于 1.0", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 输入文件不存在: {input_path}", file=sys.stderr)
        print("请先运行 03_validate_and_clean.py", file=sys.stderr)
        sys.exit(1)

    records = load_jsonl(input_path)
    print(f"[INFO] 读取 {len(records)} 条记录")

    if len(records) < 10:
        print(f"[WARN] 数据量过少（{len(records)} 条），建议至少 100 条以上再划分。")

    train, val, test = split_dataset(
        records, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )

    print(f"\n[INFO] 划分结果:")
    print(f"  训练集: {len(train)} 条 ({len(train)/len(records)*100:.1f}%)")
    print(f"  验证集: {len(val)} 条 ({len(val)/len(records)*100:.1f}%)")
    print(f"  测试集: {len(test)} 条 ({len(test)/len(records)*100:.1f}%)")

    # LLaMA-Factory 要求 JSON 格式（list of dict）
    save_json(train, DATA_CLEANED_DIR / "train.json")
    save_json(val, DATA_CLEANED_DIR / "val.json")
    save_json(test, DATA_CLEANED_DIR / "test.json")

    # 人工评测集
    random.seed(args.seed + 1)
    manual_eval = random.sample(test, min(args.manual_eval_count, len(test)))
    save_json(manual_eval, DATA_CLEANED_DIR / "test_manual_eval.json")

    print(f"\n[INFO] 人工评测集: {len(manual_eval)} 条")
    print("[INFO] 数据集划分完成！")
    print(f"\n下一步：运行 05_register_dataset_info.py 注册数据集到 LLaMA-Factory")


if __name__ == "__main__":
    main()
