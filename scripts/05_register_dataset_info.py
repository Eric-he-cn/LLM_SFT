#!/usr/bin/env python3
"""
05_register_dataset_info.py
自动向 LLaMA-Factory 的 data/dataset_info.json 追加新闻数据集配置。

注意：
  - 采用 append 模式，不覆盖已有配置
  - 自动检测数据集文件是否存在
  - 支持相对路径（相对于 LlamaFactory 根目录）

用法：
  # 在 LlamaFactory 根目录下执行
  python scripts/05_register_dataset_info.py

  # 或指定 LlamaFactory 根目录
  python scripts/05_register_dataset_info.py --llamafactory_root /path/to/LlamaFactory
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# 尝试自动定位 LlamaFactory 根目录（脚本可能在不同位置运行）
def find_llamafactory_root(start: Path) -> Path | None:
    """从当前目录向上查找包含 data/dataset_info.json 的目录。"""
    current = start
    for _ in range(8):
        candidate = current / "data" / "dataset_info.json"
        if candidate.exists():
            return current
        current = current.parent
    return None


DATASET_ENTRIES = {
    "news_structured_summary": {
        "file_name": "data/cleaned/train.json",
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    },
    "news_structured_summary_val": {
        "file_name": "data/cleaned/val.json",
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    },
    "news_structured_summary_test": {
        "file_name": "data/cleaned/test.json",
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    },
}


def register_datasets(llamafactory_root: Path, dry_run: bool = False) -> None:
    dataset_info_path = llamafactory_root / "data" / "dataset_info.json"

    if not dataset_info_path.exists():
        print(f"[ERROR] 未找到 {dataset_info_path}", file=sys.stderr)
        print("请确认在 LlamaFactory 根目录运行，或使用 --llamafactory_root 指定路径", file=sys.stderr)
        sys.exit(1)

    # 读取现有配置
    with open(dataset_info_path, encoding="utf-8") as f:
        existing = json.load(f)

    print(f"[INFO] 当前已注册数据集: {len(existing)} 个")

    added = []
    skipped = []

    for name, config in DATASET_ENTRIES.items():
        if name in existing:
            skipped.append(name)
            print(f"[INFO] 跳过（已存在）: {name}")
            continue

        # 检查数据文件是否存在
        data_file = llamafactory_root / config["file_name"]
        if not data_file.exists():
            print(f"[WARN] 数据文件不存在: {data_file}（将仍然注册，确保训练前生成数据）")

        existing[name] = config
        added.append(name)
        print(f"[INFO] 注册: {name} -> {config['file_name']}")

    if not added:
        print("[INFO] 没有需要新增的数据集配置。")
        return

    if dry_run:
        print(f"\n[DRY RUN] 将新增 {len(added)} 个数据集配置: {added}")
        print("[DRY RUN] 未写入文件。")
        return

    # 写回
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] 成功注册 {len(added)} 个数据集: {added}")
    print(f"[INFO] 跳过 {len(skipped)} 个已有配置: {skipped}")
    print(f"[INFO] 已更新: {dataset_info_path}")


def main():
    parser = argparse.ArgumentParser(description="注册数据集到 LLaMA-Factory")
    parser.add_argument("--llamafactory_root", type=str, default=None,
                        help="LlamaFactory 根目录路径（自动检测时可不填）")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅预览，不实际写入")
    args = parser.parse_args()

    if args.llamafactory_root:
        root = Path(args.llamafactory_root)
    else:
        # 自动查找
        root = find_llamafactory_root(Path.cwd())
        if root is None:
            root = find_llamafactory_root(PROJECT_DIR)
        if root is None:
            print("[ERROR] 无法自动定位 LlamaFactory 根目录（需包含 data/dataset_info.json）",
                  file=sys.stderr)
            print("请使用 --llamafactory_root 参数指定路径", file=sys.stderr)
            sys.exit(1)

    print(f"[INFO] LlamaFactory 根目录: {root}")
    register_datasets(root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

