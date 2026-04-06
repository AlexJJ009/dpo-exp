# Pipeline 监控提示词 — Gemma3-4B-PT (base)

## 使用方法

在新 tmux 窗口中启动一个 Sonnet 会话，然后粘贴下面的 loop 命令：

```bash
# 新开 tmux 窗口
tmux new-session -s monitor-gemma3-base

# 启动 Sonnet 会话
claude --model sonnet
```

进入会话后，粘贴以下内容：

---

## 粘贴内容（从这里开始复制）

```
/loop 10m 你是 DPO training pipeline 的监控员。每次触发时执行以下检查流程。

## 你在监控什么

tmux session `gemma3-4b-base` 中正在运行一个 Gemma3-4B-PT DPO 全流程 pipeline（DeepSpeed ZeRO 2, 8 GPUs），包含 4 个顺序步骤：

| Step | 容器 | 内容 | 预估耗时 |
|------|------|------|----------|
| 1/4 | dpo-harness | 偏好对生成 (1200 prompts x 16 rollouts, --strict) | ~30-60min |
| 2/4 | dpo-harness | clean_pairs 二次验证 | ~2-5min |
| 3/4 | dpo-harness | DPO 训练 (DeepSpeed ZeRO 2, 8 GPUs, ~40 steps) | ~20-40min |
| 4/4 | verl-harness (PYTHONPATH=/data-1/verl07/verl) | 7 benchmark 评测 (n=3) | ~20-30min |

## 检查步骤

### 1. 基础存活检查
```bash
# tmux session 是否存在
tmux has-session -t gemma3-4b-base 2>/dev/null && echo "SESSION: alive" || echo "SESSION: DEAD"

# 是否有 docker 容器在运行
docker ps --format '{{.Image}} {{.Status}} {{.Command}}' | grep -E "dpo-harness|verl-harness" || echo "NO CONTAINER RUNNING"
```

### 2. 读取日志末尾
```bash
tail -30 /data-1/dpo-experiment/run_gemma3_4b_base_pipeline.log
```

### 3. 判断当前阶段和健康状态

根据日志末尾内容判断：
- 如果看到 `Adding requests` 或 `Generating rollouts` → Step 1 rollout 中，正常
- 如果看到 `STEP: Build preference pairs` → Step 1 build_pairs 中，正常
- 如果看到 `Step 1 complete` 或 `STEP 2/4` → Step 2 clean 中，正常
- 如果看到 `STEP 3/4` 或 `Starting DPO training` → Step 3 训练中
- 如果看到 `STEP 4/4` 或 `offline_eval` → Step 4 评测中
- 如果看到 `PIPELINE COMPLETE` → 全部完成
- 如果看到 `ERROR` / `OOM` / `CUDA error` / `Traceback` → 出错了

### 4. 训练阶段额外检查

如果当前在 Step 3（DPO 训练），还需检查：
```bash
# 查看训练 log 中的 loss 和 margins
tail -30 /data-1/dpo-experiment/run_gemma3_4b_base_pipeline.log | grep -E "loss|margin|reward|NaN|Inf"
```

正常标准：
- `loss` 应逐步下降（起始约 0.7，最终应 < 0.2）
- `rewards/margins` 应逐步上升（起始约 -0.15，最终应 > 1.0）
- 不应出现 NaN 或 Inf
- 使用 DeepSpeed ZeRO 2，8 GPUs 并行，总 step 数约 40（比单卡快很多）

### 5. 完成后检查

如果 pipeline 已完成，读取最终结果：
```bash
# 偏好对数量
wc -l /data-1/dataset/dpo/dpo-gemma3-4b-base/dpo-gemma3-4b-base-pairs.jsonl

# 训练 summary
python3 -c "
import json
with open('/data-1/checkpoints/gemma3-4b-base-dpo/training_logs/training_summary.json') as f:
    s = json.load(f)
r = s['results']
print(f'Pairs: {s[\"dataset_size\"]}')
print(f'DeepSpeed: {s.get(\"deepspeed\", \"N/A\")}')
print(f'GPUs: {s.get(\"num_gpus\", \"N/A\")}')
print(f'Loss: {r[\"first_step_loss\"]:.4f} -> {r[\"final_step_loss\"]:.4f}')
print(f'Margins: {r[\"first_step_margins\"]:.4f} -> {r[\"final_step_margins\"]:.4f}')
"

# 评测结果
cat /data-1/checkpoints/gemma3-4b-base-dpo/inference_n3/eval_metrics.json | python3 -m json.tool
```

## 异常处理

### tmux session 挂了但 docker 容器还在
```bash
# 不要干预，等容器自然结束。docker ps 查看状态
docker ps | grep -E "dpo-harness|verl-harness"
```

### tmux session 和容器都挂了
确认失败在哪一步，根据已有产出决定从哪里重启：
```bash
# 检查各阶段产出是否存在
ls -lh /data-1/dataset/dpo/dpo-gemma3-4b-base/
ls -lh /data-1/checkpoints/gemma3-4b-base-dpo/ 2>/dev/null
ls -lh /data-1/checkpoints/gemma3-4b-base-dpo/inference_n3/ 2>/dev/null
```

重启脚本位置：`/data-1/dpo-experiment/run_gemma3_4b_base_pipeline.sh`
如果只需要重跑某一步，参考 pipeline 文档：`/data-1/dpo-experiment/dpo_pipeline/PIPELINE.md`

### 训练出现 NaN/Inf
这是严重问题，报告即可，不要自动重启。

## 报告格式

每次检查后简短报告：
- 当前阶段：Step X/4（描述）
- 状态：正常运行 / 已完成 / 异常
- 如有异常：错误类型 + 日志关键行
- 如在训练中：最新 loss 和 margins 值

## 参考文件索引

| 文件 | 用途 |
|------|------|
| `/data-1/dpo-experiment/run_gemma3_4b_base_pipeline.sh` | 主 pipeline 脚本 |
| `/data-1/dpo-experiment/run_gemma3_4b_base_pipeline.log` | 运行日志（主要检查对象） |
| `/data-1/dpo-experiment/dpo_pipeline/PIPELINE.md` | Pipeline 完整文档 |
| `/data-1/dpo-experiment/dpo_pipeline/train_dpo_gemma3_4b_base.py` | 训练脚本（DeepSpeed ZeRO 2） |
| `/data-1/dataset/dpo/dpo-gemma3-4b-base/` | 偏好对数据输出目录 |
| `/data-1/checkpoints/gemma3-4b-base-dpo/` | 模型 checkpoint 输出目录 |
| `/data-1/checkpoints/gemma3-4b-base-dpo/training_logs/training_summary.json` | 训练结果 summary |
| `/data-1/checkpoints/gemma3-4b-base-dpo/inference_n3/eval_metrics.json` | 评测结果 |
```
