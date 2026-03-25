# 常见问题排查指南

## 1. 环境安装问题

### 1.1 `torch.cuda.is_available()` 返回 `False`

**原因**：安装了 CPU 版 PyTorch，或 CUDA 驱动版本不匹配。

**解决方案**：

```bash
# 先查看驱动支持的 CUDA 版本
nvidia-smi  # 右上角显示 CUDA Version

# 卸载旧版本，重新安装 GPU 版
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 验证
python -c "import torch; print(torch.cuda.is_available())"
```

> 如果 `nvidia-smi` 显示 CUDA 12.6，使用 `cu126`；12.4 用 `cu124`；以此类推。

---

### 1.2 `bitsandbytes` 导入失败（Windows 常见）

**症状**：
```
CUDA Setup failed despite GPU being available.
```

**解决方案 A（Windows 预编译 wheel）**：

```bash
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.2.post2-py3-none-win_amd64.whl
```

**解决方案 B（新显卡 / 新 CUDA 兼容问题）**：

改用 **WSL2 Ubuntu** 运行训练（Windows 编辑代码，WSL2 跑训练）：

```bash
wsl --install -d Ubuntu-22.04
# 在 WSL2 中安装 CUDA 版 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install bitsandbytes  # Linux 版本兼容性好很多
```

**解决方案 C（临时绕过，先验证流程）**：

将训练配置中的 QLoRA 改为普通 LoRA（去掉 `quantization_bit: 4`），验证训练流程后再回来解决 bitsandbytes。

---

### 1.3 `datasets` 加载 CNN/DailyMail 失败

**症状**：连接超时或 403 错误。

**解决方案**：

```bash
# 方案 A：设置 HuggingFace 镜像（国内用户）
export HF_ENDPOINT=https://hf-mirror.com

# 方案 B：先登录 HuggingFace
huggingface-cli login

# 方案 C：使用本地 JSONL 数据集
python scripts/01_collect_news.py --source local --input /path/to/your/news.jsonl
```

---

## 2. 训练问题

### 2.1 训练 OOM（显存不足）

**按以下顺序逐步降低显存需求**：

1. 降低序列长度（`cutoff_len: 512`）
2. 减小批量大小（`per_device_train_batch_size: 1`）
3. 增大梯度累积（`gradient_accumulation_steps: 16`）
4. 降低 LoRA rank（`lora_rank: 4`）
5. 换 3B 模型替代 7B
6. 确认使用了 QLoRA（`quantization_bit: 4`）

```yaml
# 极限省显存配置示例
cutoff_len: 512
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
lora_rank: 4
quantization_bit: 4
```

---

### 2.2 训练模板不一致导致输出乱码

**症状**：微调后模型输出奇怪字符或重复内容。

**原因**：训练和推理使用了不同的 `template`。

**解决方案**：训练和推理都必须使用 `template: qwen`，不可混用。

```yaml
# 训练配置（configs/train_qwen25_3b_qlora_news.yaml）
template: qwen

# 推理配置（configs/infer_news.yaml）
template: qwen  # 必须一致！
```

---

### 2.3 训练 loss 不降或 NaN

**可能原因与解决方案**：

| 原因 | 解决方案 |
|------|----------|
| 学习率过大 | 降低 `learning_rate`（如 `1e-4` → `5e-5`） |
| 数据格式错误 | 检查 `train.json` 格式是否符合 Alpaca 规范 |
| 梯度爆炸 | 添加 `max_grad_norm: 1.0` |
| fp16 溢出 | 改用 `bf16: true`（Ampere 及以上 GPU 支持） |

---

### 2.4 数据集注册失败

**症状**：`KeyError: news_structured_summary`

**解决方案**：确保在 LlamaFactory 根目录运行注册脚本：

```bash
cd /path/to/LlamaFactory
python scripts/05_register_dataset_info.py
```

然后验证 `data/dataset_info.json` 中是否有 `news_structured_summary` 条目。

---

## 3. API 标注问题

### 3.1 OpenAI API 调用速率限制（429 错误）

**解决方案**：

```python
# 02_generate_labels_api.py 默认使用指数退避重试
# 如果仍然频繁 429，可以增加基础等待时间：
python scripts/02_generate_labels_api.py --max_retries 10
```

或在 `.env` 中使用其他兼容 API（如 DeepSeek、智谱等）：

```env
OPENAI_API_BASE=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
```

---

### 3.2 标注格式校验失败率高（>20%）

**排查步骤**：

1. 检查 `data/labeled/label_errors.jsonl` 中的失败原因
2. 查看 `raw_label` 字段，了解模型实际输出
3. 可能需要调整 `label_prompt_news_structured.txt` 中的提示词
4. 考虑换用更强的模型（如 `gpt-4o` 替代 `gpt-4o-mini`）

---

## 4. 推理问题

### 4.1 推理速度过慢

**优化建议**：

```bash
# 减少最大生成长度
python scripts/08_demo_cli.py --model_path ... --max_new_tokens 256

# 使用量化（需要 bitsandbytes）
python scripts/08_demo_cli.py --model_path ... --quantize
```

---

### 4.2 LoRA adapter 加载失败

**症状**：`RuntimeError: Error(s) in loading state_dict`

**可能原因**：base model 与 adapter 不匹配。

**解决方案**：确保 `--model_path` 指向训练时使用的相同基座模型：

```bash
# 如果训练时用的是 Qwen2.5-3B-Instruct，推理也必须用同一个
python scripts/08_demo_cli.py \
  --model_path Qwen/Qwen2.5-3B-Instruct \
  --adapter_path outputs/checkpoints/qwen25-3b-qlora-news
```

---

## 5. 其他

### 5.1 如何重置训练并重新开始

```bash
# 删除 checkpoint 目录后重新训练
rm -rf outputs/checkpoints/qwen25-3b-qlora-news
llamafactory-cli train configs/train_qwen25_3b_qlora_news.yaml
```

### 5.2 如何合并 LoRA 权重为完整模型

```bash
llamafactory-cli export \
  --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --adapter_name_or_path outputs/checkpoints/qwen25-3b-qlora-news \
  --template qwen \
  --finetuning_type lora \
  --export_dir outputs/merged/qwen25-3b-news \
  --export_size 4 \
  --export_device cpu
```

