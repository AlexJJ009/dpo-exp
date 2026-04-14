# Platform (Meituan MLP / AFO hope) Debugging Notes

Last updated: 2026-04-14. Written as a handoff so work can continue from another machine.

## Context

- Goal: run the DPO pipeline (`experiments/run_4b_code_sft_code.sh`) on Meituan MLP via `ml-easy-job`.
- Queue: `root.shxs_training_cluster.hadoop-fridayagi.friday_h20_train`
- Hadoop account: `hadoop-ai-search`, MIS user: `yangfengkai02`
- Submission client host: `set-zw06-mlp-codelab-pc178`
- Dolphinfs base (primary): `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yangfengkai02/lgx/`
- Dolphinfs base (symlink): `/home/hadoop-ai-search/dolphinfs_ssd_hadoop-ai-search/yangfengkai02/lgx/`
- Key paths inside `lgx/`:
  - `dpo-exp/` — git repo
  - `dpo-wheels/` — pre-built offline wheels (vLLM 0.12.0, TRL 0.29.0, DeepSpeed 0.18.9, ...)
  - `hope_dir/` — submission staging dir (where `hope run run.hope` is invoked)
  - `logs/` — created on first run by `jupyter.sh` (tee output)
  - `beacons/` — per-step beacon files from `jupyter_min.sh`

## Current state (as of commit `40776ce1`)

- `platform/run.hope` points to the **training_codelab image** from shangou-alg team
  (`...shangou-alg_training_codelab_ubuntu22_cuda12.8_python3.12_torch2.8_trl_1.0.0_a0523d07:1.0.0`).
  Proven AFO-compatible: has 2 active tasks on the platform today, visibility=全员.
- Single-GPU config for debugging: `worker.gcoresh20-141g = 1`, memory 240GB, 16 vcore,
  SHM 64GB. Raise back to 8 GPU / 1875GB / 128 vcore for real training.
- `platform/jupyter.sh` is the heavily instrumented launcher (9 sections, see below).
  Section 7 does per-package dependency checks and only installs missing deepspeed from
  the internal PyPI mirror. It does NOT overwrite image-shipped libs.

## Known issues (and how to handle)

### 1. sglang_dev image is NOT AFO-compatible (avoid)
Old image `training_codelab_lmsysorg_sglang_dev_19d35de6:1.0.0` fails early in
`docker-run.sh` with `su: Authentication failure`. Worker never reaches our
`bash jupyter.sh`. Manifested as exit code 1 with empty dolphinfs logs.
**Fix:** use the training_codelab image instead.

### 2. hope_dir files go stale
`hope_dir/run.hope` and `hope_dir/jupyter.sh` do NOT auto-update when you `git pull`
in `dpo-exp/`. Forgetting to `cp` leaves AFO running an old version — we lost
hours to this. Always run:

```bash
cd $LGX/dpo-exp && git pull
cp platform/jupyter.sh platform/run.hope $LGX/hope_dir/
# Sanity check before submitting:
wc -l $LGX/hope_dir/jupyter.sh          # must match repo
md5sum $LGX/hope_dir/jupyter.sh $LGX/dpo-exp/platform/jupyter.sh
grep afo.docker.image.name $LGX/hope_dir/run.hope   # verify image
```

### 3. Platform UI hides the Worker log under "driver" role
Job detail → 实例 → Worker row → 日志: the role dropdown defaults to `driver`, which
does not exist for `ml-easy-job`. Switch to `worker`. Same for `更多 ▾` menu —
often only "容错" is shown; click into attempts to see failure reasons.

### 4. Pod Events not directly accessible
The UI does not expose `kubectl describe pod`-style events. Two workarounds:
- Search the AM log for `PodChangeMonitor update:` lines — they contain
  `reason=<...>` and `message=<...>` which tell you why worker failed.
- Ask platform ops (search 大象/飞书 for "MLP 支持" / "AFO 运维") to run
  `kubectl describe pod <jobId>-worker-0 -n hadoop-fridayagi`.

### 5. ConfigParser in hope is intolerant
- Non-ASCII chars (esp. U+2014 em-dash `—`) in `run.hope` crash `configparser`.
  Everything should be pure ASCII. Use `--` or `-` instead.
- `#` comments inside `[docker]` section triggered parsing errors once. Safest:
  no comments at all in `run.hope`.

### 6. jobId URL must not have trailing quote
A URL like `.../jobId=psx6jd9xrjzmgsfz%22` (`%22` is `"`) gives spurious 500s —
the trailing quote came from a copy-paste. Type the jobId cleanly in the URL bar.

### 7. Image pulls can be slow but finite
Non-cached ~30GB images take 10-30 min on first pull. 4+ days stuck = NOT slow
pull, it's `ErrImagePull` / `ImagePullBackOff` (registry prefix missing, wrong
tag, wrong repo hash). Check the image name in `run.hope` very carefully.

### 8. Library versions in the new image
The training_codelab image ships (as of 2026-02-17 build):
- python 3.12.11, cuda 12.8, torch 2.8.0+cu128
- vllm 0.10.2, transformers 4.57.1, sglang 0.5.4, ray 2.50.0
- trl (via `uv pip install trl[liger,peft,vlm]`, version unpinned at build time)
- **deepspeed NOT included** — jupyter.sh installs it from `pypi.sankuai.com`
- User wants "latest" TRL, which is what the image already has

If the code targets a different vllm/trl version, either (a) patch the code to
the image versions, or (b) rebuild a custom image FROM this one with pinned
wheels. Do NOT `pip install --force-reinstall` vllm or torch into the running
container — torch ABI mismatches are brutal to debug.

## Diagnostic toolkit

### a) `platform/jupyter.sh` — main launcher (188 lines, 9 sections)

- Line 14-28: discovers a writable log root, mirrors stdout/stderr there with `tee`
- Line 30: `trap EXIT` always prints the final rc
- Sections 1-6: identity, AFO env, cpu/mem, gpu, disk, network, dolphinfs paths,
  wheels inventory, repo inventory, python/pip
- Section 7: per-package check (vllm/trl/transformers/torch/accelerate) then
  optional deepspeed install from internal mirror
- Section 8: jupyter in background (token `oNya685`, port 8420)
- Section 9: `bash experiments/run_4b_code_sft_code.sh`, prints rc

Fails loudly on known issues:
- `exit 10` if `REPO_DIR` missing
- `exit 11` if wheels dir is empty when we need them

### b) `platform/probe/` — minimum-image probe

Separate submission to rule out platform-side issues vs image issues.
Uses a different small image (originally tf1.10, but note that image's
`su: Authentication failure` also breaks — useful negative result).

### c) `platform/jupyter_min.sh` — beacon diagnostic

Writes per-step beacon files to `${LGX_DIR}/beacons/beacon_<TS>_<step>`:
- `00_started` — shebang + mkdir worked
- `01_identity`, `02_env`, `03_mounts`, `04_gpu`, `05_python`
- `99_done` — all steps completed

If only `00_started` exists, the script died on the first section (most likely
mounts or env). If `99_done` exists but main `jupyter.sh` still fails, the
problem is specifically in section 7+ of the full launcher.

## Useful commands on the submission client

```bash
# Always stop orphans before resubmitting — they eat queue quota
hope killjob <run_id>
hope killjob 47844274   # (older runs)

# Submit
cd $LGX/hope_dir && hope run run.hope

# Check results after 5-10 min
ls -lat $LGX/logs/
ls -lat $LGX/beacons/
tail -300 $(ls -t $LGX/logs/run_*.log | head -1)
```

## Failure timeline (chronological, newest first)

| Job ID | Image | Outcome | Lesson |
|---|---|---|---|
| `psx5hf4xrk5ssjk8` | sglang_dev_19d35de6:1.0.0, 1 GPU | 3x fail exit=1, `su: Authentication failure` in docker-run.sh | sglang image not AFO-compatible → switch image |
| `psx5h9nxrk5sv7fh` | sglang_dev_19d35de6:1.0.0, 8 GPU | fail exit=1 | same image issue, plus queue 6 H20 short |
| `psx6jd9xrjzmgsfz` | sglang_dev:1.0.0 (no hash suffix, no registry prefix) | 4 days stuck "资源已分配" | ImagePullBackOff — wrong image name (no registry + wrong repo hash) |
| DPO-Probe (tf1.10) | hadoop-hadoop_tf1.10_mt1.0.4:... | exit=2 failover | tf1.10 is a board-viewer image, not a general worker base |

## Next steps (pick up here on the other machine)

1. On the other machine: `git pull` in `dpo-exp/`, then `cp platform/jupyter.sh
   platform/run.hope hope_dir/` (see section "hope_dir files go stale").
2. Submit with training_codelab image. Expect: container starts within 5-15
   min, jupyter.sh runs, Section 1-8 debug lands in `$LGX/logs/run_*.log`.
3. Section 9 (DPO training) will probably OOM or mis-count processes on 1
   GPU — that's expected. Confirm full debug log is available, then scale
   resources back up in `run.hope`:
   ```ini
   worker.memory = 1920000
   worker.vcore = 128
   worker.gcoresh20-141g = 8
   ```
   and bump `YARN_CONTAINER_RUNTIME_DOCKER_SHM_SIZE_BYTES` back to `549755813888`.
4. If library versions in the image don't match what `run_4b_code_sft_code.sh`
   expects (e.g. `from trl import ...` API changes), decide between:
   - Patching the code to the image's TRL / vLLM versions (fast)
   - Requesting a custom image from codelab team (clean but slow)
   - Installing our wheels on top (risky, torch ABI mismatch likely)
