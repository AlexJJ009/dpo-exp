# DPO 实验归档索引

> 最后更新：2026-04-07（Server B 补充）
> 维护：GongxunLi
> Git 仓库：`/data-1/dpo-experiment`

本文件通过 git 进行版本管理。各服务器的实验结果分节记录，便于多人协作时区分来源。

---

## 全局实验总览

| # | 服务器 | 实验名 | 基座模型 | 数据集规模 | 状态 |
|---|--------|--------|----------|-----------|------|
| 1 | Server A（本机）| Qwen3-4B-Base DPO v1 | Qwen3-4B-Base | 6,013 对 | ✅ 完成 |
| 2 | Server A（本机）| Qwen3-4B-Base DPO v2 | Qwen3-4B-Base | 4,485 对（严格过滤）| ✅ 完成 |
| 3 | Server A（本机）| Qwen3-4B-SFT DPO | Qwen3-4B-Base-SFT-stage-1 | 5,860 对 | ✅ 完成 |
| 4 | Server A（本机）| Qwen3-8B-Base DPO | Qwen3-8B-Base | 7,934 对 | ✅ 完成 |
| 5 | Server A（本机）| Gemma3-4B-SFT DPO | Gemma3-4B-Base-SFT-stage-1 | 3,781 对（补充中）| 🔄 进行中 |
| B-1 | Server B | Qwen2.5-3B-Base DPO | Qwen/Qwen2.5-3B | 10,298 对（严格过滤）| ✅ 完成 |
| B-2 | Server B | Gemma3-4B-Base DPO | google/gemma-3-4b-pt | 10,781 对（严格过滤）| 🔄 训练中 |
| B-3 | Server B | Gemma3-4B-SFT DPO | Gemma3-4B-Base-SFT-stage-1 | 5,202 对（已同步）| ⏳ 待启动（watcher 就绪）|

<!-- 其他服务器的实验请在下方「Server B」节中补充 -->

---

## Server A（本机，GongxunLi）

**数据根目录**：`/data-1/`
**Checkpoint 根目录**：`/data-1/checkpoints/`
**数据集根目录**：`/data-1/dataset/dpo/`

---

## 实验 1：Qwen3-4B-Base DPO v1

### 文件位置

| 类型 | 路径 |
|------|------|
| 偏好对数据集 | `/data-1/dataset/dpo-4b-pairs.jsonl` |
| 模型 Checkpoint | `/data-1/checkpoints/qwen3-4b-dpo/` |
| 最终 Checkpoint | `/data-1/checkpoints/qwen3-4b-dpo/checkpoint-376` |
| 训练摘要 | `/data-1/checkpoints/qwen3-4b-dpo/training_logs/training_summary.json` |
| 评估结果 | `/data-1/checkpoints/qwen3-4b-dpo/inference_n3_sysprompt/eval_metrics.json` |
| Pipeline 日志 | `/data-1/dpo-experiment/run_4b_full_pipeline.log` |

### 训练超参数

| 参数 | 值 |
|------|----|
| beta | 0.1 |
| learning_rate | 5e-7 |
| epochs | 1 |
| per_device_batch | 2 |
| grad_accumulation | 8 |
| effective_batch | 16 |
| max_length | 2048 |
| warmup_ratio | 0.1 |
| lr_scheduler | cosine |
| total_steps | 76 |
| runtime | 7,283 s（~2.0 h）|

### 训练结果

| 指标 | 初始值 | 最终值 |
|------|--------|--------|
| Loss | 0.7746 | 0.023 |
| Reward Margin | -0.1505 | 3.9896 |
| Chosen Reward | — | +2.3918 |
| Rejected Reward | — | -1.5978 |

### 评估结果（n=3, temp=1.0, top_p=0.95, max_tokens=4096）

> ⚠️ 注意：除 MATH-500 外，其余数据集抽取失败率极高（GSM8K/MAWPS/SVAMP 均 >98%），
> 原因是 Base 模型未经 SFT 对齐，答案格式不符合提取器预期。GSM8K/MAWPS/SVAMP 数据无效。

| 数据集 | mean@3 | pass@1 | pass@3 | maj@3 | extraction_fail |
|--------|--------|--------|--------|-------|----------------|
| MATH-500 | 35.7% | 35.7% | 65.6% | 55.4% | 36.4% |
| AIME-2025 | 1.1% | 1.1% | 3.3% | 0.0% | 33.3% |
| AMC23 | 22.5% | 22.5% | 45.0% | 37.5% | 40.0% |
| AQUA | 0.4% | 0.4% | 1.2% | 1.2% | **98.7%** ⚠️ |
| GSM8K | 1.1% | 1.1% | 3.2% | 3.0% | **98.4%** ⚠️ |
| MAWPS | 0.9% | 0.9% | 2.8% | 2.8% | **98.8%** ⚠️ |
| SVAMP | 0.6% | 0.6% | 1.7% | 1.3% | **99.4%** ⚠️ |

---

## 实验 2：Qwen3-4B-Base DPO v2

> 相比 v1：启用 `--strict` 严格过滤，数据量减少但质量更高。

### 文件位置

| 类型 | 路径 |
|------|------|
| 偏好对数据集 | `/data-1/dataset/dpo/dpo-4b-v2/dpo-4b-v2-pairs.jsonl` |
| Rollouts | `/data-1/dataset/dpo/dpo-4b-v2/dpo-4b-v2-rollouts.jsonl` |
| Extracted Prompts | `/data-1/dataset/dpo/dpo-4b-v2/dpo-4b-v2-extracted.jsonl` |
| 模型 Checkpoint | `/data-1/checkpoints/qwen3-4b-base-dpo-v2/` |
| 最终 Checkpoint | `/data-1/checkpoints/qwen3-4b-base-dpo-v2/checkpoint-281` |
| 训练摘要 | `/data-1/checkpoints/qwen3-4b-base-dpo-v2/training_logs/training_summary.json` |
| 评估结果 | `/data-1/checkpoints/qwen3-4b-base-dpo-v2/inference_n3/eval_metrics.json` |

### 训练超参数

| 参数 | 值 |
|------|----|
| beta | 0.1 |
| learning_rate | 5e-7 |
| epochs | 1 |
| per_device_batch | 1 |
| grad_accumulation | 16 |
| effective_batch | 16 |
| max_length | 2048 |
| warmup_ratio | 0.1 |
| lr_scheduler | cosine |
| strict_filtering | ✅ 是 |
| total_steps | 57 |
| runtime | 5,676 s（~1.6 h）|

### 训练结果

| 指标 | 初始值 | 最终值 |
|------|--------|--------|
| Loss | 0.7995 | 0.0945 |
| Reward Margin | -0.1897 | 2.5327 |
| Chosen Reward | — | +2.1782 |
| Rejected Reward | — | -0.3545 |

### 评估结果（n=3, temp=1.0, top_p=0.95, max_tokens=4096）

> ⚠️ 注意：Base 模型格式对齐问题依然存在，GSM8K/MAWPS/SVAMP 抽取失败率 54–65%，数据可信度偏低。

| 数据集 | mean@3 | pass@1 | pass@3 | maj@3 | extraction_fail |
|--------|--------|--------|--------|-------|----------------|
| MATH-500 | 33.1% | 33.1% | 62.8% | 50.8% | 36.3% |
| AIME-2025 | 2.2% | 2.2% | 3.3% | 3.3% | 37.8% |
| AMC23 | 17.5% | 17.5% | 40.0% | 30.0% | 35.0% |
| AQUA | 6.6% | 6.6% | 16.9% | 14.2% | 62.1% |
| GSM8K | 31.1% | 31.1% | 65.4% | 57.2% | **53.9%** ⚠️ |
| MAWPS | 27.4% | 27.4% | 61.4% | 57.2% | **64.8%** ⚠️ |
| SVAMP | 24.6% | 24.6% | 56.0% | 51.7% | **63.7%** ⚠️ |

---

## 实验 3：Qwen3-4B-SFT DPO

> 基座为经 SFT 对齐的 Qwen3-4B，是 4B 系列中综合性能最强的实验。

### 文件位置

| 类型 | 路径 |
|------|------|
| 偏好对数据集 | `/data-1/dataset/dpo-4b-sft-pairs.jsonl` |
| 模型 Checkpoint | `/data-1/checkpoints/qwen3-4b-sft-dpo/` |
| 最终 Checkpoint | `/data-1/checkpoints/qwen3-4b-sft-dpo/checkpoint-367` |
| 训练摘要 | `/data-1/checkpoints/qwen3-4b-sft-dpo/training_logs/training_summary.json` |
| 评估结果 | `/data-1/checkpoints/qwen3-4b-sft-dpo/inference_n3/eval_metrics.json` |

### 训练超参数

| 参数 | 值 |
|------|----|
| beta | 0.1 |
| learning_rate | 5e-7 |
| epochs | 1 |
| per_device_batch | 2 |
| grad_accumulation | 8 |
| effective_batch | 16 |
| max_length | 2048 |
| warmup_ratio | 0.1 |
| lr_scheduler | cosine |
| total_steps | 74 |
| runtime | 7,007 s（~1.9 h）|

### 训练结果

| 指标 | 初始值 | 最终值 |
|------|--------|--------|
| Loss | 0.7321 | 0.4601 |
| Reward Margin | -0.0637 | 0.5685 |
| Chosen Reward | — | -0.0137 |
| Rejected Reward | — | -0.5822 |

> 注：Loss 收敛较慢、Margin 偏小，符合 SFT 模型 DPO 的特征（初始分布已较优，两极分化空间小）。

### 评估结果（n=3, temp=1.0, top_p=0.95, max_tokens=4096）

| 数据集 | mean@3 | pass@1 | pass@3 | maj@3 | extraction_fail |
|--------|--------|--------|--------|-------|----------------|
| MATH-500 | **67.7%** | **67.7%** | **80.2%** | **78.4%** | 29.9% |
| AIME-2025 | 7.8% | 7.8% | 13.3% | 13.3% | 87.8% |
| AMC23 | **40.8%** | **40.8%** | **52.5%** | **52.5%** | 57.5% |
| AQUA | **65.0%** | **65.0%** | **80.7%** | **76.4%** | 23.2% |
| GSM8K | **89.8%** | **89.8%** | **94.8%** | **92.5%** | 3.7% |
| MAWPS | **94.4%** | **94.4%** | **96.3%** | **96.3%** | 1.6% |
| SVAMP | **90.7%** | **90.7%** | **95.0%** | **93.3%** | 4.0% |

---

## 实验 4：Qwen3-8B-Base DPO

### 文件位置

| 类型 | 路径 |
|------|------|
| 偏好对数据集 | `/data-1/dataset/dpo-8b-pairs.jsonl` |
| 模型 Checkpoint | `/data-1/checkpoints/qwen3-8b-dpo/` |
| 最终 Checkpoint | `/data-1/checkpoints/qwen3-8b-dpo/checkpoint-496` |
| 训练摘要 | `/data-1/checkpoints/qwen3-8b-dpo/training_logs/training_summary.json` |
| 评估结果 | `/data-1/checkpoints/qwen3-8b-dpo/inference_n3/eval_metrics.json` |

### 训练超参数

| 参数 | 值 |
|------|----|
| beta | 0.1 |
| learning_rate | 5e-7 |
| epochs | 1 |
| per_device_batch | 1 |
| grad_accumulation | 16 |
| effective_batch | 16 |
| max_length | 2048 |
| warmup_ratio | 0.1 |
| lr_scheduler | cosine |
| total_steps | 100 |
| runtime | 11,603 s（~3.2 h）|

### 训练结果

| 指标 | 初始值 | 最终值 |
|------|--------|--------|
| Loss | 0.688 | 0.0155 |
| Reward Margin | +0.0170 | 4.5132 |
| Chosen Reward | — | +3.1052 |
| Rejected Reward | — | -1.4080 |

### 评估结果（n=3, temp=1.0, top_p=0.95, max_tokens=4096）

| 数据集 | mean@3 | pass@1 | pass@3 | maj@3 | extraction_fail |
|--------|--------|--------|--------|-------|----------------|
| MATH-500 | 51.1% | 51.1% | 76.6% | 63.0% | 13.5% |
| AIME-2025 | 8.9% | 8.9% | 13.3% | 10.0% | 17.8% |
| AMC23 | 30.0% | 30.0% | 52.5% | 35.0% | 12.5% |
| AQUA | 5.1% | 5.1% | 14.2% | 6.7% | **31.9%** ⚠️ |
| GSM8K | 57.2% | 57.2% | 88.3% | 75.6% | 20.7% |
| MAWPS | 72.8% | 72.8% | 93.8% | 87.6% | 16.1% |
| SVAMP | 70.2% | 70.2% | 92.7% | 87.3% | 16.2% |

---

## 实验 5：Gemma3-4B-SFT DPO（进行中）

### 文件位置

| 类型 | 路径 |
|------|------|
| 偏好对数据集（现有） | `/data-1/dataset/dpo/dpo-gemma3-4b-sft/dpo-gemma3-4b-sft-pairs.jsonl`（3,781 对）|
| 偏好对数据集（补充中）| `/data-1/dataset/dpo/dpo-gemma3-4b-sft-extra/dpo-gemma3-4b-sft-extra-pairs.jsonl` |
| 模型 Checkpoint | `/data-1/checkpoints/gemma3-4b-sft-dpo/`（训练完成后生成）|
| Pipeline 脚本 | `/data-1/dpo-experiment/run_gemma3_4b_sft_pipeline.sh` |
| 续接脚本 | `/data-1/dpo-experiment/run_gemma3_4b_sft_continue.sh` |
| 原始日志 | `/data-1/dpo-experiment/run_gemma3_4b_sft_pipeline.log` |
| 续接日志 | `/data-1/dpo-experiment/run_gemma3_4b_sft_continue.log` |

### 进度

- [x] Step 1：生成偏好对（3,781 对，原始 1200 prompt，`--strict`）
- [x] Step 2：Clean pairs
- [ ] Step 1'：补充生成（offset=1200，500 新 prompt，目标总量 ≥ 5000）
- [ ] Step 2'：Clean 补充数据
- [ ] Step 3'：合并数据集
- [ ] Step 4：DPO 训练（DeepSpeed ZeRO 2，8 GPUs）
- [ ] Step 5：评估（7 个数学基准）

---

## Server A 数据集来源

| 数据集 | 路径 | 用途 |
|--------|------|------|
| EnsembleLLM 源数据 | `/data-1/dataset/EnsembleLLM-data/am_deepseek_r1_filtered_ad.jsonl`（111,657 条）| 生成 DPO 偏好对的 prompt 来源 |
| MATH-500 | `/data-1/dataset/MATH-500/math500-test_with_system_prompt.parquet` | 评估 |
| AIME-2025 | `/data-1/dataset/AIME-2025/aime-2025_with_system_prompt.parquet` | 评估 |
| AMC23 | `/data-1/dataset/AMC23/amc23-test_with_system_prompt.parquet` | 评估 |
| AQUA | `/data-1/dataset/AQUA/aqua-test_with_system_prompt.parquet` | 评估 |
| GSM8K | `/data-1/dataset/gsm8k/gsm8k-test_with_system_prompt.parquet` | 评估 |
| MAWPS | `/data-1/dataset/MAWPS/mawps-test_with_system_prompt.parquet` | 评估 |
| SVAMP | `/data-1/dataset/SVAMP/svamp-test_with_system_prompt.parquet` | 评估 |

---

## Server B

**数据根目录**：`/data-1/`
**Checkpoint 根目录**：`/data-1/checkpoints/`
**数据集根目录**：`/data-1/dataset/dpo/`
**Git 仓库**：`/data-1/dpo-experiment`

---

## 实验 B-1：Qwen2.5-3B-Base DPO

> Server B 首个 DPO 实验，使用 Qwen2.5-3B（非 Qwen3 系列），严格过滤，DeepSpeed ZeRO 2。

### 文件位置

| 类型 | 路径 |
|------|------|
| 偏好对数据集 | `/data-1/dataset/dpo/dpo-qwen25-3b-base/dpo-qwen25-3b-base-pairs.jsonl` |
| Rollouts | `/data-1/dataset/dpo/dpo-qwen25-3b-base/dpo-qwen25-3b-base-rollouts.jsonl` |
| Extracted Prompts | `/data-1/dataset/dpo/dpo-qwen25-3b-base/dpo-qwen25-3b-base-extracted.jsonl` |
| 模型 Checkpoint | `/data-1/checkpoints/qwen25-3b-base-dpo/` |
| 最终 Checkpoint | `/data-1/checkpoints/qwen25-3b-base-dpo/checkpoint-644` |
| 训练摘要 | `/data-1/checkpoints/qwen25-3b-base-dpo/training_logs/training_summary.json` |
| 评估结果 | `/data-1/checkpoints/qwen25-3b-base-dpo/inference_n3/eval_metrics.json` |
| Pipeline 脚本 | `/data-1/dpo-experiment/run_qwen25_3b_base_pipeline.sh` |
| Pipeline 日志（初次失败）| `/data-1/dpo-experiment/run_qwen25_3b_base_pipeline.log` |
| Resume 日志 | `/data-1/dpo-experiment/run_qwen25_3b_base_resume.log` |

### 训练超参数

| 参数 | 值 |
|------|----|
| beta | 0.1 |
| learning_rate | 5e-7 |
| epochs | 1 |
| per_device_batch | 1 |
| grad_accumulation | 2 |
| effective_batch | 16 |
| max_length | 2048 |
| warmup_ratio | 0.1 |
| lr_scheduler | cosine |
| strict_filtering | ✅ 是 |
| num_gpus | 8 |
| deepspeed | ZeRO Stage 2 |
| total_steps | 644 |
| runtime | 1,364 s（~23 min）|

### 训练结果

| 指标 | 初始值 | 最终值 |
|------|--------|--------|
| Loss | 0.7294 | 0.0 |
| Reward Margin | -0.0502 | 15.025 |
| Chosen Reward | — | +11.075 |
| Rejected Reward | — | -3.959 |

> ⚠️ 注意：Loss 降至 0、Margin 达 15 属极端值，可能发生过拟合。数据集规模（10298 对）相对 effective batch 16 步数过多（644 步），模型可能记忆了训练集。

### 评估结果（n=3, temp=1.0, top_p=0.95, max_tokens=4096）

| 数据集 | mean@3 | pass@1 | maj@3 |
|--------|--------|--------|-------|
| MATH-500 | 18.3% | 18.3% | 33.2% |
| AIME-2025 | 0.0% | 0.0% | 0.0% |
| AMC23 | 11.7% | 11.7% | 15.0% |
| AQUA | 2.6% | 2.6% | 7.1% |
| GSM8K | 18.4% | 18.4% | 40.8% |
| MAWPS | 36.9% | 36.9% | 70.1% |
| SVAMP | 27.0% | 27.0% | 53.7% |

---

## 实验 B-2：Gemma3-4B-Base DPO（训练中）

> Gemma3-4B 预训练基座模型的 DPO 实验，严格过滤，DeepSpeed ZeRO 2，8 GPUs。

### 文件位置

| 类型 | 路径 |
|------|------|
| 偏好对数据集 | `/data-1/dataset/dpo/dpo-gemma3-4b-base/dpo-gemma3-4b-base-pairs.jsonl` |
| Rollouts | `/data-1/dataset/dpo/dpo-gemma3-4b-base/dpo-gemma3-4b-base-rollouts.jsonl` |
| Extracted Prompts | `/data-1/dataset/dpo/dpo-gemma3-4b-base/dpo-gemma3-4b-base-extracted.jsonl` |
| 模型 Checkpoint | `/data-1/checkpoints/gemma3-4b-base-dpo/`（训练中）|
| 训练摘要 | `/data-1/checkpoints/gemma3-4b-base-dpo/training_logs/training_summary.json`（训练后生成）|
| 评估结果 | `/data-1/checkpoints/gemma3-4b-base-dpo/inference_n3/eval_metrics.json`（评估后生成）|
| Pipeline 脚本 | `/data-1/dpo-experiment/run_gemma3_4b_base_pipeline.sh` |
| Pipeline 日志（初次失败）| `/data-1/dpo-experiment/run_gemma3_4b_base_pipeline.log` |
| Resume 日志 | `/data-1/dpo-experiment/run_gemma3_4b_base_resume.log` |

### 训练超参数

| 参数 | 值 |
|------|----|
| beta | 0.1 |
| learning_rate | 5e-7 |
| epochs | 1 |
| per_device_batch | 1 |
| grad_accumulation | 2 |
| effective_batch | 16 |
| max_length | 2048 |
| warmup_ratio | 0.1 |
| lr_scheduler | cosine |
| strict_filtering | ✅ 是 |
| num_gpus | 8 |
| deepspeed | ZeRO Stage 2 |
| attn_implementation | eager（Gemma3 sliding window 兼容）|

### 已知问题与修复

- `google/gemma-3-4b-pt` 为预训练基座，tokenizer 无 chat template。TRL v0.29.0 的 `DPOTrainer` 无条件调用 `apply_chat_template`，导致首次运行报 `ValueError`。
  **修复**：在 `train_dpo_gemma3_4b_base.py` 中于 tokenizer 加载后补充一个纯文本 fallback template（`chat_template is None` 时生效）。

### 进度

- [x] Step 1：生成偏好对（10,817 对原始 → 10,781 对 clean，1200 prompt，`--strict`）
- [x] Step 2：Clean pairs
- [ ] Step 3：DPO 训练（DeepSpeed ZeRO 2，8 GPUs）← 当前运行中（约 32%）
- [ ] Step 4：评估（7 个数学基准）

---

## 实验 B-3：Gemma3-4B-SFT DPO（待启动，watcher 已就绪）

> 基于 SFT 对齐版 Gemma3-4B 的 DPO 实验。数据集已从 Server A 同步到本机，watcher 监控 B-2 完成后自动启动。
>
> ⚠️ 注意：本机应使用 `run_gemma3_4b_sft_resume.sh` 而非 `run_gemma3_4b_sft_continue.sh`。
> 后者会重新生成 extra 数据并覆盖已合并的 5,202 对，导致数据错误。

### 文件位置

| 类型 | 路径 |
|------|------|
| 偏好对数据集（已合并）| `/data-1/dataset/dpo/dpo-gemma3-4b-sft/dpo-gemma3-4b-sft-pairs.jsonl`（5,202 对）|
| Extra 数据集（合并来源）| `/data-1/dataset/dpo/dpo-gemma3-4b-sft-extra/dpo-gemma3-4b-sft-extra-pairs.jsonl`（1,421 对）|
| 模型 | `/data-1/.cache/gemma3-4b-base-sft-stage-1` |
| 模型 Checkpoint | `/data-1/checkpoints/gemma3-4b-sft-dpo/`（训练后生成）|
| Resume 脚本 | `/data-1/dpo-experiment/run_gemma3_4b_sft_resume.sh` |
| Resume 日志 | `/data-1/dpo-experiment/run_gemma3_4b_sft_resume.log`（训练后生成）|

### 进度

- [x] Step 1：生成偏好对（Server A 完成，3,781 对）
- [x] Step 1'：补充生成（Server A 完成，offset=1200，500 新 prompt → 1,421 对 extra）
- [x] Step 2/2'：Clean pairs
- [x] Step 3'：合并数据集（5,202 对，已同步至本机）
- [ ] Step 4：DPO 训练（DeepSpeed ZeRO 2，8 GPUs）← watcher 等待 B-2 完成后自动启动
- [ ] Step 5：评估（7 个数学基准）

---

## 横向对比（MATH-500 mean@3，仅含已评估实验）

| 服务器 | 模型 | mean@3 | pass@1 | pass@3 | extraction_fail |
|--------|------|--------|--------|--------|----------------|
| Server A | Qwen3-4B-SFT DPO | **67.7%** | **67.7%** | **80.2%** | 29.9% |
| Server A | Qwen3-8B-Base DPO | 51.1% | 51.1% | 76.6% | 13.5% |
| Server A | Qwen3-4B-Base DPO v1 | 35.7% | 35.7% | 65.6% | 36.4% |
| Server A | Qwen3-4B-Base DPO v2 | 33.1% | 33.1% | 62.8% | 36.3% |
| Server B | Qwen2.5-3B-Base DPO | 18.3% | 18.3% | — | — |
| Server B | Gemma3-4B-Base DPO | — | — | — | 🔄 训练中 |

> **关键结论**：SFT 对齐对 DPO 效果影响显著。4B-SFT-DPO 在 MATH-500 上超过 8B-Base-DPO（67.7% vs 51.1%）。
> Base 模型 DPO 在 GSM8K/MAWPS/SVAMP 上答案格式对齐问题严重（抽取失败率 >98%），这些数据集的评估结果不可靠。
> Qwen2.5-3B-Base DPO（Server B）训练指标异常（loss→0, margin→15），疑似过拟合，评估结果仅供参考。
