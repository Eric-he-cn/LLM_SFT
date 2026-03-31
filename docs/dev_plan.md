# Qwen3-4B 结构化摘要项目开发计划（已升级 AWQ 主线）

## 项目目标

围绕“端侧信息流结构化摘要”完成一条稳定可复现的工程链路：

- 数据构建与清洗（XL-Sum + DeepSeek 标注）
- QLoRA SFT（Qwen3-4B）
- AWQ 后训练量化（W4A16）
- vLLM 统一推理评测（质量 + 性能）

当前决策：
- 结论模型采用 `g128 + calib256 + chat-template + 分层抽样`。
- vLLM 作为主评测后端，AutoAWQ 推理保留为备选链路。

---

## 技术路线

```text
原始新闻 -> API 标注 -> 校验清洗 -> train/val/test
       -> QLoRA SFT (Qwen3-4B)
       -> AutoAWQ 量化 (W4A16)
       -> vLLM 冒烟 + 全量质量 + 性能压测
```

---

## 里程碑

### M1 数据与 SFT（已完成）

- 数据清洗后可用集：`train/val/test = 3843/480/481`
- SFT 完成并导出 merged 模型：`outputs/merged/qwen3-4b-news-v2`
- 基础评测链路（ROUGE + 格式）可复现

### M2 AWQ 主链（已完成）

- 量化链路：`09 -> 10`
- vLLM 冒烟：`13`
- vLLM 全量质量：`14`
- 统一性能评测：`07 --backend vllm`

### M3 结论固化与文档化（进行中）

- 主流程和备选流程分层（vLLM 主链、AutoAWQ 备选）
- README 与命令清单口径统一
- 常见故障与排查文档完善

---

## 验收口径

### 质量

- ROUGE-1/2/L
- all_sections_pass_rate
- valid_category_rate
- valid_bullets_rate
- has_time_info_rate

### 性能

- load_time_s
- ttft_p50/p95
- latency_p50/p95
- tokens_per_s
- peak_gpu_memory_mb
- model_disk_size_gb

### 阈值

- ROUGE-L 相对下降 <= 3%
- all_sections_pass_rate 下降 <= 2%
- 性能满足其一：latency_p50 提升 >= 15% 或显存下降 >= 25%

---

## 当前风险与对策

- Windows/WSL 混合运行时可能残留 GPU 进程占卡
  - 对策：统一通过 WSL 跑 vLLM，异常时先清理残留进程再续跑。
- tokenizer_config 兼容问题导致加载失败
  - 对策：使用 `*_tokenizerfix` 自动修复目录。
- 量化参数继续探索可能提升 ROUGE 但破坏格式稳定
  - 对策：固定最终方案，探索项仅作为附录记录。
