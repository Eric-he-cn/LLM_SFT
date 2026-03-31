#!/usr/bin/env python3
"""
15_build_awq_baseline_manifest.py
固化当前 AWQ 质量回收实验基线，不重跑评测。
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DEFAULT_OUT = PROJECT_DIR / "outputs" / "eval" / "awq_recovery" / "baseline_manifest.json"


def pick_existing(paths: List[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]


def load_json_if_exists(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def file_info(path: Path) -> Dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": st.st_size,
        "modified_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 AWQ 改进实验基线清单")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUT), help="baseline 清单输出路径")
    args = parser.parse_args()

    quality_summary = pick_existing(
        [
            PROJECT_DIR / "outputs" / "eval" / "awq_benchmark_vllm_full" / "benchmark_summary_awq_vllm.json",
            PROJECT_DIR / "outputs" / "eval" / "awq_benchmark_vllm_smoke" / "benchmark_summary_awq_vllm.json",
        ]
    )
    perf_bf16 = pick_existing(
        [
            PROJECT_DIR / "outputs" / "eval" / "latency_report_sft_bf16_vllm_promptfix30.json",
            PROJECT_DIR / "outputs" / "eval" / "latency_report_sft_bf16_vllm_full.json",
        ]
    )
    perf_awq = pick_existing(
        [
            PROJECT_DIR / "outputs" / "eval" / "latency_report_sft_awq4_vllm_promptfix30.json",
            PROJECT_DIR / "outputs" / "eval" / "latency_report_sft_awq4_vllm_full.json",
        ]
    )
    quant_report = pick_existing(
        [
            PROJECT_DIR / "outputs" / "quantized" / "qwen3-4b-news-v2-awq4" / "awq_quantize_report.json",
        ]
    )

    quality_obj = load_json_if_exists(quality_summary)
    perf_bf16_obj = load_json_if_exists(perf_bf16)
    perf_awq_obj = load_json_if_exists(perf_awq)
    quant_obj = load_json_if_exists(quant_report)

    manifest = {
        "task": "awq_recovery_baseline_snapshot",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "quality_summary": file_info(quality_summary),
            "perf_bf16": file_info(perf_bf16),
            "perf_awq": file_info(perf_awq),
            "quant_report": file_info(quant_report),
        },
        "baseline_metrics": {
            "quality": {
                "n_samples": quality_obj.get("n_samples"),
                "bf16": (quality_obj.get("variants") or [{}])[0] if quality_obj.get("variants") else {},
                "awq": (quality_obj.get("variants") or [{}, {}])[1] if len(quality_obj.get("variants") or []) > 1 else {},
                "comparison": quality_obj.get("comparison", {}),
            },
            "performance": {
                "bf16": {
                    "model_variant": perf_bf16_obj.get("model_variant"),
                    "load_time_s": perf_bf16_obj.get("load_time_s"),
                    "ttft_p50_s": perf_bf16_obj.get("ttft_p50_s"),
                    "ttft_p95_s": perf_bf16_obj.get("ttft_p95_s"),
                    "latency_p50_s": perf_bf16_obj.get("latency_p50_s"),
                    "latency_p95_s": perf_bf16_obj.get("latency_p95_s"),
                    "tokens_per_s": perf_bf16_obj.get("tokens_per_s"),
                    "peak_gpu_memory_mb": perf_bf16_obj.get("peak_gpu_memory_mb"),
                    "disk_size_gb": perf_bf16_obj.get("disk_size_gb"),
                },
                "awq": {
                    "model_variant": perf_awq_obj.get("model_variant"),
                    "load_time_s": perf_awq_obj.get("load_time_s"),
                    "ttft_p50_s": perf_awq_obj.get("ttft_p50_s"),
                    "ttft_p95_s": perf_awq_obj.get("ttft_p95_s"),
                    "latency_p50_s": perf_awq_obj.get("latency_p50_s"),
                    "latency_p95_s": perf_awq_obj.get("latency_p95_s"),
                    "tokens_per_s": perf_awq_obj.get("tokens_per_s"),
                    "peak_gpu_memory_mb": perf_awq_obj.get("peak_gpu_memory_mb"),
                    "disk_size_gb": perf_awq_obj.get("disk_size_gb"),
                },
            },
            "quantization": quant_obj,
        },
        "commands_reference": {
            "quality_eval_script": "python scripts/14_benchmark_quality_vllm.py ...",
            "latency_eval_script": "python scripts/07_benchmark_latency.py ...",
            "quantize_script": "python scripts/10_quantize_awq.py ...",
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] baseline manifest: {out_path}")
    for key, item in manifest["paths"].items():
        print(f"[INFO] {key}: {item['path']} | exists={item['exists']}")


if __name__ == "__main__":
    main()
