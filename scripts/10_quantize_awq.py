#!/usr/bin/env python3
"""
10_quantize_awq.py
使用 AutoAWQ 对已合并 SFT 模型执行 4bit 后训练量化。

默认参数（与实验计划一致）：
- w_bit=4
- q_group_size=128
- zero_point=True
- version=GEMM（失败时自动回退 GEMV）
- max_calib_seq_len=1024

示例：
  python scripts/10_quantize_awq.py \
    --model_path outputs/merged/qwen3-4b-news-v2 \
    --calib_path outputs/awq/calib_prompts.jsonl \
    --output_dir outputs/quantized/qwen3-4b-news-v2-awq4
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DEFAULT_MODEL = PROJECT_DIR / "outputs" / "merged" / "qwen3-4b-news-v2"
DEFAULT_CALIB = PROJECT_DIR / "outputs" / "awq" / "calib_prompts.jsonl"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "quantized" / "qwen3-4b-news-v2-awq4"


def patch_qwen3_attention_type_for_awq() -> None:
    """为 Qwen3 + AutoAWQ 的 Catcher 兼容问题打运行时补丁。

    现象：AWQ 校准阶段替换 layer0 为 Catcher 后，Qwen3 forward 会读取
    decoder_layer.attention_type，导致 AttributeError。
    处理：仅在该异常触发时，按 config.layer_types 回填缺失字段并重试 forward。
    """
    try:
        from transformers.models.qwen3 import modeling_qwen3
    except Exception:
        return

    model_cls = getattr(modeling_qwen3, "Qwen3Model", None)
    if model_cls is None:
        return
    if getattr(model_cls, "_awq_attention_patch_applied", False):
        return

    original_forward = model_cls.forward

    def patched_forward(self, *args, **kwargs):
        try:
            return original_forward(self, *args, **kwargs)
        except AttributeError as exc:
            if "attention_type" not in str(exc):
                raise
            layer_types = getattr(getattr(self, "config", None), "layer_types", None)
            for idx, layer in enumerate(getattr(self, "layers", [])):
                if hasattr(layer, "attention_type"):
                    continue
                if isinstance(layer_types, (list, tuple)) and idx < len(layer_types):
                    setattr(layer, "attention_type", layer_types[idx])
                else:
                    setattr(layer, "attention_type", "full_attention")
            return original_forward(self, *args, **kwargs)

    model_cls.forward = patched_forward
    model_cls._awq_attention_patch_applied = True


def load_calib_prompts(path: Path, limit: int) -> List[str]:
    """加载校准文本（jsonl）。"""
    prompts = []  # type: List[str]
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = str(obj.get("prompt", "")).strip()
            if not prompt:
                continue
            prompts.append(prompt)
            if limit > 0 and len(prompts) >= limit:
                break
    return prompts


def quantize_once(model, tokenizer, calib_data: List[str], quant_config: Dict, max_calib_seq_len: int) -> None:
    """执行一次量化，兼容不同 AutoAWQ 版本签名。"""
    try:
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=calib_data,
            max_calib_seq_len=max_calib_seq_len,
        )
        return
    except TypeError:
        pass

    try:
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=calib_data,
        )
        return
    except TypeError:
        pass

    # 最小参数兜底
    model.quantize(tokenizer, quant_config=quant_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoAWQ 4bit 量化")
    parser.add_argument("--model_path", type=str, required=True, help="输入模型路径（merged SFT）")
    parser.add_argument("--calib_path", type=str, required=True, help="校准集 JSONL 路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出量化模型目录")
    parser.add_argument("--w_bit", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--zero_point", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    parser.add_argument("--version", type=str, default="GEMM", choices=["GEMM", "GEMV"])
    parser.add_argument("--max_calib_seq_len", type=int, default=1024)
    parser.add_argument("--calib_samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    calib_path = Path(args.calib_path)
    output_dir = Path(args.output_dir)

    if not model_path.exists():
        print(f"[ERROR] 模型目录不存在: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not calib_path.exists():
        print(f"[ERROR] 校准文件不存在: {calib_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        print("[ERROR] 缺少依赖：pip install autoawq transformers", file=sys.stderr)
        sys.exit(1)

    # AutoAWQ 对新版本 Qwen3 的 Catcher 兼容性补丁（仅运行时生效）
    patch_qwen3_attention_type_for_awq()

    calib_data = load_calib_prompts(calib_path, args.calib_samples)
    if not calib_data:
        print("[ERROR] 校准数据为空", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 加载模型: {model_path}")
    print(f"[INFO] 校准样本: {len(calib_data)}")
    t0 = time.perf_counter()
    model = AutoAWQForCausalLM.from_pretrained(str(model_path), safetensors=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    load_s = time.perf_counter() - t0

    quant_config = {
        "w_bit": args.w_bit,
        "q_group_size": args.group_size,
        "zero_point": args.zero_point,
        "version": args.version,
    }
    used_version = args.version
    fallback_to_gemv = False
    quant_error = None

    print(f"[INFO] 开始量化: {quant_config}")
    q0 = time.perf_counter()
    try:
        quantize_once(
            model=model,
            tokenizer=tokenizer,
            calib_data=calib_data,
            quant_config=quant_config,
            max_calib_seq_len=args.max_calib_seq_len,
        )
    except Exception as exc:  # noqa: BLE001
        quant_error = exc
        if args.version == "GEMM":
            print(f"[WARN] GEMM 量化失败，尝试回退 GEMV: {exc}")
            quant_config["version"] = "GEMV"
            used_version = "GEMV"
            fallback_to_gemv = True
            quantize_once(
                model=model,
                tokenizer=tokenizer,
                calib_data=calib_data,
                quant_config=quant_config,
                max_calib_seq_len=args.max_calib_seq_len,
            )
        else:
            raise
    quant_s = time.perf_counter() - q0

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 保存量化模型: {output_dir}")
    try:
        model.save_quantized(str(output_dir), safetensors=True)
    except TypeError:
        model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    report = {
        "model_path": str(model_path),
        "calib_path": str(calib_path),
        "output_dir": str(output_dir),
        "calib_samples": len(calib_data),
        "max_calib_seq_len": args.max_calib_seq_len,
        "seed": args.seed,
        "quant_config": {
            "w_bit": args.w_bit,
            "q_group_size": args.group_size,
            "zero_point": args.zero_point,
            "requested_version": args.version,
            "used_version": used_version,
            "fallback_to_gemv": fallback_to_gemv,
        },
        "timing": {
            "load_time_s": load_s,
            "quant_time_s": quant_s,
        },
        "quant_error_preview": (str(quant_error)[:500] if quant_error else ""),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = output_dir / "awq_quantize_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== AWQ 量化完成 =====")
    print(f"加载耗时:   {load_s:.2f}s")
    print(f"量化耗时:   {quant_s:.2f}s")
    print(f"量化版本:   {used_version}")
    print(f"模型输出:   {output_dir}")
    print(f"报告文件:   {report_path}")


if __name__ == "__main__":
    main()
