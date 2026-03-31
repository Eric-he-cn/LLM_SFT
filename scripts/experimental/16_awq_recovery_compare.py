#!/usr/bin/env python3
"""
16_awq_recovery_compare.py
对比 AWQ 候选方案（Q1/Q2）与已固化基线，并输出阈值判断。
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_awq_variant(summary: Dict) -> Dict:
    for v in summary.get("variants", []):
        if str(v.get("group", "")).lower().find("awq") >= 0:
            return v
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="AWQ 改进方案与基线对比")
    parser.add_argument("--baseline_manifest", type=str, required=True)
    parser.add_argument("--candidate_quality_summary", type=str, required=True, help="14 脚本输出的 summary json")
    parser.add_argument("--candidate_perf_report", type=str, required=True, help="07 脚本输出的延迟报告 json")
    parser.add_argument("--candidate_name", type=str, default="Q1")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    baseline = load_json(Path(args.baseline_manifest))
    cand_quality = load_json(Path(args.candidate_quality_summary))
    cand_perf = load_json(Path(args.candidate_perf_report))

    base_quality = baseline.get("baseline_metrics", {}).get("quality", {})
    base_awq = base_quality.get("awq", {})
    base_comp = base_quality.get("comparison", {})
    base_perf_awq = baseline.get("baseline_metrics", {}).get("performance", {}).get("awq", {})

    cand_awq = find_awq_variant(cand_quality)
    cand_comp = cand_quality.get("comparison", {})

    report = {
        "task": "awq_recovery_candidate_compare",
        "candidate_name": args.candidate_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_manifest": str(Path(args.baseline_manifest)),
        "candidate_inputs": {
            "quality_summary": str(Path(args.candidate_quality_summary)),
            "perf_report": str(Path(args.candidate_perf_report)),
        },
        "quality_compare": {
            "baseline_awq": {
                "rougeL": base_awq.get("rougeL"),
                "all_sections_pass_rate": base_awq.get("all_sections_pass_rate"),
                "valid_category_rate": base_awq.get("valid_category_rate"),
                "valid_bullets_rate": base_awq.get("valid_bullets_rate"),
                "has_time_info_rate": base_awq.get("has_time_info_rate"),
            },
            "candidate_awq": {
                "rougeL": cand_awq.get("rougeL"),
                "all_sections_pass_rate": cand_awq.get("all_sections_pass_rate"),
                "valid_category_rate": cand_awq.get("valid_category_rate"),
                "valid_bullets_rate": cand_awq.get("valid_bullets_rate"),
                "has_time_info_rate": cand_awq.get("has_time_info_rate"),
            },
            "delta_vs_baseline_awq": {
                "rougeL": (cand_awq.get("rougeL", 0.0) - base_awq.get("rougeL", 0.0)),
                "all_sections_pass_rate": (
                    cand_awq.get("all_sections_pass_rate", 0.0) - base_awq.get("all_sections_pass_rate", 0.0)
                ),
            },
            "candidate_vs_bf16_thresholds": {
                "rougeL_drop_ratio": cand_comp.get("rougeL_drop_ratio"),
                "all_sections_pass_drop": cand_comp.get("all_sections_pass_drop"),
                "rougeL_drop_ratio_le_0_03": bool(cand_comp.get("thresholds", {}).get("rougeL_drop_ratio_le_0_03", False)),
                "all_sections_pass_drop_le_0_02": bool(
                    cand_comp.get("thresholds", {}).get("all_sections_pass_drop_le_0_02", False)
                ),
            },
            "baseline_vs_bf16_thresholds": base_comp,
        },
        "performance_compare": {
            "baseline_awq": {
                "latency_p50_s": base_perf_awq.get("latency_p50_s"),
                "latency_p95_s": base_perf_awq.get("latency_p95_s"),
                "tokens_per_s": base_perf_awq.get("tokens_per_s"),
                "peak_gpu_memory_mb": base_perf_awq.get("peak_gpu_memory_mb"),
            },
            "candidate_awq": {
                "latency_p50_s": cand_perf.get("latency_p50_s"),
                "latency_p95_s": cand_perf.get("latency_p95_s"),
                "tokens_per_s": cand_perf.get("tokens_per_s"),
                "peak_gpu_memory_mb": cand_perf.get("peak_gpu_memory_mb"),
            },
            "delta_vs_baseline_awq": {
                "latency_p50_s": (cand_perf.get("latency_p50_s", 0.0) - base_perf_awq.get("latency_p50_s", 0.0)),
                "latency_p95_s": (cand_perf.get("latency_p95_s", 0.0) - base_perf_awq.get("latency_p95_s", 0.0)),
                "tokens_per_s": (cand_perf.get("tokens_per_s", 0.0) - base_perf_awq.get("tokens_per_s", 0.0)),
                "peak_gpu_memory_mb": (
                    (cand_perf.get("peak_gpu_memory_mb") or 0.0) - (base_perf_awq.get("peak_gpu_memory_mb") or 0.0)
                ),
            },
        },
    }

    pass_rouge = bool(report["quality_compare"]["candidate_vs_bf16_thresholds"]["rougeL_drop_ratio_le_0_03"])
    pass_format = bool(report["quality_compare"]["candidate_vs_bf16_thresholds"]["all_sections_pass_drop_le_0_02"])
    report["decision_gate"] = {
        "quality_threshold_pass": (pass_rouge and pass_format),
        "suggest_run_q2": (not (pass_rouge and pass_format)),
        "reason": (
            "Q1 已达到质量阈值，可不进入 Q2"
            if (pass_rouge and pass_format)
            else "Q1 未达到质量阈值，建议进入 Q2"
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] compare report: {out}")
    print(f"[INFO] decision: {report['decision_gate']['reason']}")


if __name__ == "__main__":
    main()
