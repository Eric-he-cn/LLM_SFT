# AWQ / dvAWQ Backlog（附录，暂缓执行）

本清单用于记录已讨论但暂不执行的 AWQ 改进项，后续可按优先级逐步落地。

## P0（优先）

1. 校准集改为 chat-template 分布
- 从仅 `input` 文本改为 `system + user` 拼装后的 prompt。
- system prompt 与 SFT/主评测保持同一版本。

2. 两组重量化对比
- Q1: `group_size=128`, `calib_samples=256`（chat-template）
- Q2: `group_size=64`, `calib_samples=512`（chat-template + 长度分桶）

3. 分层抽样（长度分桶）
- 按 token 长度区间抽样，避免校准样本长度分布偏置。

## P1（后续）

1. 质量评测分解指标扩展
- 六字段逐项覆盖率
- missing tail fields（主要主体/时间信息/潜在影响）
- 重复标题率、替代字段率、failure signature top-k

2. 性能报告显存分段
- `gpu_mem_before_load_mb`
- `gpu_mem_after_load_mb`
- `gpu_mem_after_warmup_mb`
- `peak_gpu_memory_mb`
- `gpu_memory_utilization`

## 备注

- 本清单只记录决策，不改变当前已验证流程。
- 当前阶段先聚焦“统一提示词后的性能复测”。
