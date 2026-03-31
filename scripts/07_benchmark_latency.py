#!/usr/bin/env python3
"""
07_benchmark_latency.py
统一推理性能评测入口（HF / AutoAWQ / vLLM 后端）：
- 加载耗时（load_time_s）
- 首 token 延迟 TTFT（p50/p95）
- 端到端延迟（p50/p95）
- 吞吐（tokens/s）
- 峰值显存与模型磁盘体积

示例：
  python scripts/07_benchmark_latency.py \
    --backend hf \
    --model_path outputs/merged/qwen3-4b-news-v2 \
    --report_tag sft_bf16

  python scripts/07_benchmark_latency.py \
    --backend awq \
    --model_path outputs/quantized/qwen3-4b-news-v2-awq4 \
    --report_tag sft_awq4

  python scripts/07_benchmark_latency.py \
    --backend vllm \
    --model_path outputs/quantized/qwen3-4b-news-v2-awq4 \
    --quantization awq_marlin \
    --report_tag sft_awq4_vllm
"""

import argparse
import json
import math
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
EVAL_DIR = PROJECT_DIR / "outputs" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TEST = PROJECT_DIR / "data" / "cleaned" / "test.json"
# 与 SFT/主评测保持一致的中等约束系统提示词
SYSTEM_PROMPT = (
    "你是专业的新闻编辑助手。请对新闻内容进行结构化摘要，严格按以下6个标签顺序输出，禁止使用 Markdown：\n"
    "【一句话摘要】【核心要点】【事件类别】【主要主体】【时间信息】【潜在影响】\n\n"
    "其中【核心要点】用阿拉伯数字编号列出至少3条；\n"
    "【事件类别】只能从以下选择：政治、经济、科技、文化、社会、军事、体育、健康、环境、国际、历史、旅游、财经"
)


def load_test_samples(path: Path, num_samples: int) -> List[Dict]:
    """加载测试样本，返回指定数量切片。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    if num_samples <= 0:
        return data
    return data[:num_samples]


def build_prompt(record: dict) -> str:
    """从样本中提取输入字段。"""
    return str(record.get("input", "")).strip()


def percentile(values: List[float], p: float) -> float:
    """计算分位数，空列表返回 0。"""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(math.ceil((p / 100.0) * len(sorted_values))) - 1
    idx = min(max(idx, 0), len(sorted_values) - 1)
    return sorted_values[idx]


def calc_disk_size_gb(path: Path) -> float:
    """计算模型目录体积（GB）。"""
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / (1024 ** 3)
    total = 0
    for fp in path.rglob("*"):
        if fp.is_file():
            total += fp.stat().st_size
    return total / (1024 ** 3)


def maybe_prepare_tokenizer_fix(model_path: Path, tokenizer_path: Path) -> Tuple[Path, bool]:
    """兼容 transformers 对 extra_special_tokens=list 的报错，生成修复副本。"""
    cfg_path = tokenizer_path / "tokenizer_config.json"
    if not cfg_path.exists():
        return tokenizer_path, False
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return tokenizer_path, False

    extra = cfg.get("extra_special_tokens")
    if not isinstance(extra, list):
        return tokenizer_path, False

    fixed_dir = model_path.parent / f"{model_path.name}_tokenizerfix"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"]:
        src = tokenizer_path / name
        if src.exists():
            shutil.copy2(src, fixed_dir / name)

    fixed_cfg_path = fixed_dir / "tokenizer_config.json"
    fixed_cfg = json.loads(fixed_cfg_path.read_text(encoding="utf-8"))
    fixed_map = {}
    for i, tok in enumerate(extra):
        key = re.sub(r"[^0-9a-zA-Z_]+", "_", str(tok).strip("<>|")).strip("_").lower()
        if not key:
            key = f"tok_{i}"
        if key in fixed_map:
            key = f"{key}_{i}"
        fixed_map[key] = tok
    fixed_cfg["extra_special_tokens"] = fixed_map
    fixed_cfg["tokenizer_class"] = "Qwen2TokenizerFast"
    fixed_cfg_path.write_text(json.dumps(fixed_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    return fixed_dir, True


def _gpu_memory_used_mb_nvml() -> Optional[float]:
    """读取当前 GPU0 已用显存（MB），失败返回 None。"""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024 / 1024
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        first = out.splitlines()[0].strip()
        return float(first)
    except Exception:
        return None


def _sync_cuda(torch_mod) -> None:
    """在 CUDA 场景同步，确保计时准确。"""
    try:
        if torch_mod.cuda.is_available():
            torch_mod.cuda.synchronize()
    except Exception:
        pass


def _move_input_ids_to_model_device(input_ids, model, torch_mod, prefer_cuda: bool):
    """将输入迁移到模型所在设备。"""
    try:
        model_device = next(model.parameters()).device
        return input_ids.to(model_device)
    except Exception:
        if prefer_cuda and torch_mod.cuda.is_available():
            return input_ids.to("cuda")
    return input_ids


def _safe_generate(model, input_ids, tokenizer, max_new_tokens: int, temperature: float, attention_mask=None):
    """兼容不同后端 generate 参数签名。"""
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": temperature > 0.01,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.1,
    }
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    try:
        return model.generate(input_ids, **kwargs)
    except TypeError:
        kwargs.pop("repetition_penalty", None)
        try:
            return model.generate(input_ids, **kwargs)
        except TypeError:
            # 最小参数兜底
            return model.generate(input_ids, max_new_tokens=max_new_tokens)


def _load_hf_model(model_path: str, adapter_path: Optional[str], device: str):
    """加载 HF 模型后端（可选 merge LoRA）。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    load_kwargs = {"trust_remote_code": True}
    if device == "cuda":
        load_kwargs["torch_dtype"] = torch.bfloat16
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError:
            print("[ERROR] 缺少 peft：pip install peft", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] 加载 LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("[INFO] LoRA 权重已合并")

    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return model, tokenizer


def _load_awq_model(model_path: str, device: str):
    """加载 AutoAWQ 量化模型后端。"""
    import torch
    from transformers import AutoTokenizer

    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        print("[ERROR] 缺少 autoawq：pip install autoawq", file=sys.stderr)
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if not hasattr(AutoAWQForCausalLM, "from_quantized"):
        print("[ERROR] 当前 AutoAWQ 版本缺少 from_quantized 接口，请升级 autoawq", file=sys.stderr)
        sys.exit(1)

    kwargs_candidates = [
        {
            "quant_path": model_path,
            "trust_remote_code": True,
            "fuse_layers": True,
            "safetensors": True,
            "device_map": "auto" if device == "cuda" else "cpu",
        },
        {
            "quant_path": model_path,
            "trust_remote_code": True,
            "fuse_layers": False,
            "safetensors": True,
            "device_map": "auto" if device == "cuda" else "cpu",
        },
        {
            "quant_path": model_path,
            "trust_remote_code": True,
            "device_map": "auto" if device == "cuda" else "cpu",
        },
    ]

    model = None
    last_err = None
    for kwargs in kwargs_candidates:
        try:
            model = AutoAWQForCausalLM.from_quantized(**kwargs)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue

    if model is None:
        print(f"[ERROR] AWQ 模型加载失败: {last_err}", file=sys.stderr)
        sys.exit(1)

    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return model, tokenizer


def _load_vllm_model(model_path: str, quantization: str, tokenizer_path: Optional[str]):
    """加载 vLLM 模型后端。"""
    try:
        from vllm import LLM
    except ImportError:
        print("[ERROR] 缺少 vllm：pip install vllm", file=sys.stderr)
        sys.exit(1)

    model_dir = Path(model_path)
    raw_tokenizer_path = Path(tokenizer_path) if tokenizer_path else model_dir
    if not raw_tokenizer_path.exists():
        print(f"[ERROR] tokenizer_path 不存在: {raw_tokenizer_path}", file=sys.stderr)
        sys.exit(1)
    fixed_tokenizer_path, used_fix = maybe_prepare_tokenizer_fix(
        model_path=model_dir,
        tokenizer_path=raw_tokenizer_path,
    )
    if used_fix:
        print(f"[INFO] tokenizer 已自动修复: {fixed_tokenizer_path}")

    q = None if quantization == "none" else quantization
    llm = LLM(
        model=model_path,
        tokenizer=str(fixed_tokenizer_path),
        tokenizer_mode="auto",
        trust_remote_code=True,
        quantization=q,
        dtype="float16",
        tensor_parallel_size=1,
    )
    tokenizer = llm.get_tokenizer()
    meta = {
        "tokenizer_path_resolved": str(fixed_tokenizer_path),
        "tokenizer_fix_applied": used_fix,
    }
    return llm, tokenizer, meta


def _build_input_ids(tokenizer, prompt: str):
    """构建 chat 输入 token。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )
    except Exception:
        text = f"{SYSTEM_PROMPT}\n\n{prompt}"
        return tokenizer(text, return_tensors="pt").input_ids


def _build_prompt_text(tokenizer, prompt: str) -> str:
    """构建 vLLM 文本 prompt。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return f"{SYSTEM_PROMPT}\n\n{prompt}"


def _normalize_model_inputs(raw_inputs):
    """统一为 generate 可接受的 kwargs 输入。"""
    if isinstance(raw_inputs, dict):
        result = {}
        if "input_ids" in raw_inputs:
            result["input_ids"] = raw_inputs["input_ids"]
        if "attention_mask" in raw_inputs:
            result["attention_mask"] = raw_inputs["attention_mask"]
        return result

    input_ids = getattr(raw_inputs, "input_ids", None)
    attention_mask = getattr(raw_inputs, "attention_mask", None)
    if input_ids is not None:
        out = {"input_ids": input_ids}
        if attention_mask is not None:
            out["attention_mask"] = attention_mask
        return out

    return {"input_ids": raw_inputs}


def run_benchmark(
    backend: str,
    model_path: str,
    adapter_path: Optional[str],
    test_samples: List[Dict],
    max_new_tokens: int,
    temperature: float,
    device: str,
    warmup_steps: int,
    repeat: int,
    top_p: float,
    quantization: str,
    tokenizer_path: Optional[str],
) -> Dict:
    """执行统一性能评测并返回报告。"""
    torch = None
    if backend in {"hf", "awq"}:
        try:
            import torch as _torch

            torch = _torch
        except ImportError:
            print("[ERROR] 缺少 torch：pip install torch", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            import torch as _torch

            torch = _torch
        except Exception:
            torch = None

    if not test_samples:
        print("[ERROR] 测试样本为空", file=sys.stderr)
        sys.exit(1)

    vllm_meta = {}
    peak_gpu_memory_mb_nvml = _gpu_memory_used_mb_nvml()
    load_t0 = time.perf_counter()
    if backend == "hf":
        model, tokenizer = _load_hf_model(model_path, adapter_path, device)
    elif backend == "awq":
        if adapter_path:
            print("[WARN] AWQ 后端将忽略 adapter_path（请传入已量化模型目录）。")
        model, tokenizer = _load_awq_model(model_path, device)
    elif backend == "vllm":
        if device != "cuda":
            print("[ERROR] vllm 后端当前仅支持 device=cuda", file=sys.stderr)
            sys.exit(1)
        if adapter_path:
            print("[WARN] vllm 后端将忽略 adapter_path。")
        model, tokenizer, vllm_meta = _load_vllm_model(
            model_path=model_path,
            quantization=quantization,
            tokenizer_path=tokenizer_path,
        )
    else:
        print(f"[ERROR] 不支持 backend={backend}", file=sys.stderr)
        sys.exit(1)
    load_time_s = time.perf_counter() - load_t0
    mem_now = _gpu_memory_used_mb_nvml()
    if mem_now is not None:
        peak_gpu_memory_mb_nvml = max(peak_gpu_memory_mb_nvml or 0.0, mem_now)

    def run_once(sample: Dict) -> Tuple[float, float, int]:
        """返回 (ttft_s, latency_s, output_tokens)。"""
        prompt = build_prompt(sample)
        if not prompt:
            return 0.0, 0.0, 0
        if backend == "vllm":
            from vllm import SamplingParams

            prompt_text = _build_prompt_text(tokenizer, prompt)
            ttft_sp = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=1,
                repetition_penalty=1.1,
            )
            lat_sp = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                repetition_penalty=1.1,
            )

            ttft_t0 = time.perf_counter()
            _ = model.generate([prompt_text], ttft_sp, use_tqdm=False)
            ttft_s = time.perf_counter() - ttft_t0

            lat_t0 = time.perf_counter()
            outputs = model.generate([prompt_text], lat_sp, use_tqdm=False)
            latency_s = time.perf_counter() - lat_t0

            output_tokens = 0
            if outputs and outputs[0].outputs:
                first = outputs[0].outputs[0]
                token_ids = getattr(first, "token_ids", None)
                if token_ids is not None:
                    output_tokens = len(token_ids)
                else:
                    output_tokens = len(tokenizer.encode(first.text))
            return ttft_s, latency_s, max(output_tokens, 0)

        raw_inputs = _build_input_ids(tokenizer, prompt)
        model_inputs = _normalize_model_inputs(raw_inputs)
        model_inputs["input_ids"] = _move_input_ids_to_model_device(
            model_inputs["input_ids"], model, torch, prefer_cuda=(device == "cuda")
        )
        if "attention_mask" in model_inputs:
            model_inputs["attention_mask"] = model_inputs["attention_mask"].to(model_inputs["input_ids"].device)
        input_len = int(model_inputs["input_ids"].shape[-1])

        _sync_cuda(torch)
        ttft_t0 = time.perf_counter()
        with torch.no_grad():
            _ = _safe_generate(
                model=model,
                input_ids=model_inputs["input_ids"],
                tokenizer=tokenizer,
                max_new_tokens=1,
                temperature=temperature,
                attention_mask=model_inputs.get("attention_mask"),
            )
        _sync_cuda(torch)
        ttft_s = time.perf_counter() - ttft_t0

        _sync_cuda(torch)
        lat_t0 = time.perf_counter()
        with torch.no_grad():
            generated = _safe_generate(
                model=model,
                input_ids=model_inputs["input_ids"],
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                attention_mask=model_inputs.get("attention_mask"),
            )
        _sync_cuda(torch)
        latency_s = time.perf_counter() - lat_t0
        output_tokens = int(generated[0].shape[-1] - input_len)
        return ttft_s, latency_s, max(output_tokens, 0)

    warmup_steps = max(warmup_steps, 0)
    repeat = max(repeat, 1)

    print(f"[INFO] backend={backend} | device={device} | load_time={load_time_s:.2f}s")
    print(f"[INFO] 预热 {warmup_steps} 次，正式计时 {repeat} 次")

    for i in range(warmup_steps):
        sample = test_samples[i % len(test_samples)]
        _ = run_once(sample)
        mem_now = _gpu_memory_used_mb_nvml()
        if mem_now is not None:
            peak_gpu_memory_mb_nvml = max(peak_gpu_memory_mb_nvml or 0.0, mem_now)
        if (i + 1) % 5 == 0 or (i + 1) == warmup_steps:
            print(f"[WARMUP] {i + 1}/{warmup_steps}")

    ttfts = []  # type: List[float]
    latencies = []  # type: List[float]
    output_tokens_list = []  # type: List[int]

    print("-" * 60)
    for i in range(repeat):
        sample = test_samples[i % len(test_samples)]
        ttft_s, latency_s, output_tokens = run_once(sample)
        if latency_s <= 0:
            continue
        ttfts.append(ttft_s)
        latencies.append(latency_s)
        output_tokens_list.append(output_tokens)
        mem_now = _gpu_memory_used_mb_nvml()
        if mem_now is not None:
            peak_gpu_memory_mb_nvml = max(peak_gpu_memory_mb_nvml or 0.0, mem_now)
        print(
            f"[{i + 1:3d}/{repeat}] "
            f"TTFT={ttft_s:.3f}s | Latency={latency_s:.3f}s | out_tokens={output_tokens}"
        )

    if not latencies:
        print("[ERROR] 没有有效推理结果", file=sys.stderr)
        sys.exit(1)

    total_latency = sum(latencies)
    total_output_tokens = sum(output_tokens_list)
    tokens_per_s = (total_output_tokens / total_latency) if total_latency > 0 else 0.0

    report = {
        "model_variant": Path(model_path).name,
        "backend": backend,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "num_samples_pool": len(test_samples),
        "num_runs": len(latencies),
        "warmup_steps": warmup_steps,
        "repeat": repeat,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "device": device,
        "quantization": quantization if backend == "vllm" else "",
        "load_time_s": load_time_s,
        "ttft_p50_s": percentile(ttfts, 50),
        "ttft_p95_s": percentile(ttfts, 95),
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "tokens_per_s": tokens_per_s,
        "avg_output_tokens": (statistics.mean(output_tokens_list) if output_tokens_list else 0.0),
        "model_disk_size_gb": calc_disk_size_gb(Path(model_path)),
        "disk_size_gb": calc_disk_size_gb(Path(model_path)),
        # 兼容旧字段
        "avg_latency_s": statistics.mean(latencies),
        "median_latency_s": statistics.median(latencies),
        "p50_latency_s": percentile(latencies, 50),
        "p95_latency_s": percentile(latencies, 95),
        "min_latency_s": min(latencies),
        "max_latency_s": max(latencies),
        "stddev_latency_s": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    report.update(vllm_meta)

    if device == "cuda":
        torch_peak = None
        if torch is not None:
            try:
                torch_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
            except Exception:
                torch_peak = None
        if peak_gpu_memory_mb_nvml is not None and torch_peak is not None:
            report["peak_gpu_memory_mb"] = max(float(peak_gpu_memory_mb_nvml), float(torch_peak))
        elif peak_gpu_memory_mb_nvml is not None:
            report["peak_gpu_memory_mb"] = float(peak_gpu_memory_mb_nvml)
        elif torch_peak is not None:
            report["peak_gpu_memory_mb"] = float(torch_peak)

    return report


def main() -> None:
    """脚本入口。"""
    parser = argparse.ArgumentParser(description="统一推理性能评测（HF / AWQ / vLLM）")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--adapter_path", type=str, default=None, help="LoRA adapter 路径（仅 HF 后端）")
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "awq", "vllm"], help="推理后端")
    parser.add_argument("--test", type=str, default=str(DEFAULT_TEST), help="测试集路径")
    parser.add_argument("--num_samples", type=int, default=20, help="样本池大小（<=0 表示全量）")
    parser.add_argument("--max_new_tokens", type=int, default=800, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.0, help="生成温度")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p")
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["none", "awq_marlin"],
        help="vLLM 量化模式（仅 backend=vllm 生效）",
    )
    parser.add_argument("--tokenizer_path", type=str, default="", help="可选 tokenizer 路径（仅 backend=vllm）")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="推理设备")
    parser.add_argument("--warmup_steps", type=int, default=5, help="预热次数")
    parser.add_argument("--repeat", type=int, default=30, help="正式计时次数")
    parser.add_argument("--report_tag", type=str, default="", help="报告标签（如 sft_bf16 / sft_awq4）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=str, default="", help="输出报告路径（默认按 report_tag 生成）")
    args = parser.parse_args()

    random.seed(args.seed)

    test_path = Path(args.test)
    if not test_path.exists():
        print(f"[ERROR] 测试集不存在: {test_path}", file=sys.stderr)
        sys.exit(1)
    test_samples = load_test_samples(test_path, args.num_samples)
    print(f"[INFO] 加载样本池: {len(test_samples)} 条")

    report = run_benchmark(
        backend=args.backend,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        test_samples=test_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        quantization=args.quantization,
        tokenizer_path=(args.tokenizer_path or None),
        device=args.device,
        warmup_steps=args.warmup_steps,
        repeat=args.repeat,
    )

    if args.report_tag:
        report["model_variant"] = args.report_tag

    if args.output:
        output_path = Path(args.output)
    else:
        suffix = f"_{args.report_tag}" if args.report_tag else ""
        output_path = EVAL_DIR / f"latency_report{suffix}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 延迟评测报告 =====")
    print(f"模型标识:   {report['model_variant']}")
    print(f"后端:       {report['backend']}")
    print(f"加载耗时:   {report['load_time_s']:.3f}s")
    print(f"TTFT P50:   {report['ttft_p50_s']:.3f}s")
    print(f"TTFT P95:   {report['ttft_p95_s']:.3f}s")
    print(f"延迟 P50:   {report['latency_p50_s']:.3f}s")
    print(f"延迟 P95:   {report['latency_p95_s']:.3f}s")
    print(f"吞吐:       {report['tokens_per_s']:.2f} tokens/s")
    print(f"模型体积:   {report['model_disk_size_gb']:.3f} GB")
    if "peak_gpu_memory_mb" in report:
        print(f"峰值显存:   {report['peak_gpu_memory_mb']:.1f} MB")
    print(f"\n报告已保存: {output_path}")


if __name__ == "__main__":
    main()
