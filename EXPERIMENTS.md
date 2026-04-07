# DPO 实验归档索引

> 最后更新：2026-04-07
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

## Server B（同事服务器）

> 此节由同事维护，请参照上方 Server A 的格式补充实验记录。

### 实验总览

<!-- 请按如下格式添加实验条目：
| # | 实验名 | 基座模型 | 数据集规模 | 状态 |
|---|--------|----------|-----------|------|
| B-1 | ... | ... | ... | ✅ 完成 / 🔄 进行中 |
-->

（待补充）

### 实验详情

<!-- 请按 Server A 各实验的格式，添加文件位置、训练结果、评估结果等信息 -->

（待补充）

---

## 横向对比（MATH-500 mean@3，仅含已评估实验）

| 服务器 | 模型 | mean@3 | pass@1 | pass@3 | extraction_fail |
|--------|------|--------|--------|--------|----------------|
| Server A | Qwen3-4B-SFT DPO | **67.7%** | **67.7%** | **80.2%** | 29.9% |
| Server A | Qwen3-8B-Base DPO | 51.1% | 51.1% | 76.6% | 13.5% |
| Server A | Qwen3-4B-Base DPO v1 | 35.7% | 35.7% | 65.6% | 36.4% |
| Server A | Qwen3-4B-Base DPO v2 | 33.1% | 33.1% | 62.8% | 36.3% |

> **关键结论**：SFT 对齐对 DPO 效果影响显著。4B-SFT-DPO 在 MATH-500 上超过 8B-Base-DPO（67.7% vs 51.1%）。
> Base 模型 DPO 在 GSM8K/MAWPS/SVAMP 上答案格式对齐问题严重（抽取失败率 >98%），这些数据集的评估结果不可靠。
