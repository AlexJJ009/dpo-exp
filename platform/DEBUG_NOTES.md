# Platform (Meituan MLP / AFO hope) Debugging Notes

Last updated: 2026-04-17. Written as a handoff so work can continue from another machine.

> **2026-04-17 status update.** We have a custom-built image that fully passes the probe
> smoke suite and is ready for real DPO runs:
> `registry-offlinebiz.sankuai.com/custom_prod/com.sankuai.data.hadoop.gpu/ai-search/training_ubuntu22_cuda12.8_python3.12_torch2.8_dpo_trl_174de81b:1.0.0`.
> Both `platform/run.hope` and `platform/probe/run.hope` point at it. The hard-won image-build
> lessons from the 2026-04-14 → 2026-04-17 cycle are collected in the **Image-build playbook**
> appendix at the bottom — read that first when building the next image (e.g. the verl image
> at `/data-1/verl07/verl`).

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
| image build #1 (`dpo_trl_8071ad6f`) | FROM `phxmlp.mtjupyter.singleuser/...verl_megatron_1.0.2_534ef92c` | build OK, push 3x retry then `MANIFEST_BLOB_UNKNOWN: sha256:4f4fb700...` | cross-namespace blob cross-mount denied; retry / rename didn't help |
| image build #2 (`dpo_trl_174de81b`) | same FROM, same Dockerfile, **advanced params `--build-on-machine --no-cache`** | push succeeds after 2-3 internal kaniko retries | `--no-cache` forces fresh upload → no cross-mount; `--build-on-machine` fixes registry egress. **Playbook verified — use this combo from the start next time.** |
| DPO-Probe-newimage (174de81b) | `ai-search/...dpo_trl_174de81b:1.0.0` on 1×H20 | probe passed: `ALL PROBE CHECKS PASSED`, rc=0 | new image validated end-to-end for DPO + vLLM imports, CUDA kernel, HF_HUB_OFFLINE tokenizer load |

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

   **Update 2026-04-17:** this step is now unnecessary for DPO — the new
   `dpo_trl_174de81b:1.0.0` image bakes in the correct TRL 0.29.0 / DeepSpeed 0.18.9 /
   math-verify stack. Section 7 of `jupyter.sh` (runtime pip install) is now redundant and
   can be simplified to an import-only check.

---

# Appendix — Image-build playbook (2026-04-17, verified)

Context: built and pushed `...ai-search/training_..._dpo_trl_174de81b:1.0.0`, validated by `platform/probe/`. This appendix is the reference playbook for the next image build, especially the upcoming **verl image** for `/data-1/verl07/verl`.

## Principle: keep the human in the loop

During this cycle the AI made several confident-but-wrong guesses — fabricated registry paths, invented tags (`v1.0.5`), mis-ordered recovery steps — that cost time. Rules for future image builds:

- **Never fabricate image tags / registry paths.** If memory has a specific path, use it; otherwise ask the user to paste the value from the MLP image-list UI (`mlp.sankuai.com/ml/#/image/list`). Do not pattern-match against training-data URLs.
- **Ask the user to verify on the UI before choosing a base.** Authoritative signals on the image detail page: 项目组, 可见范围, 在用任务数, 最近使用时间, 版本数. AI should request these, not assume them.
- **Cite official docs** (`mlp.sankuai.com/ml/#/image/help`, platform flight manual) rather than paraphrasing AI-remembered flags. `--build-on-machine` and `--no-cache` are real advanced-params — confirm by looking them up before instructing the user.
- **Steps that cost real compute (rebuild, resubmit) are user-approved only.** AI drafts; user clicks.

## Step 1 — Base image selection checklist

User performs on MLP UI, AI waits for values:

- [ ] **项目组 = `hadoop-ai-search`** (matches our submit usergroup). 其他项目组即便"全员可见"也可能拉不动。
- [ ] **可见范围 = 全员** OR account has explicit pull permission
- [ ] **在用任务数 > 0 in last 7 days** (proxy for "it really pulls on some queue")
- [ ] **Registry path is `custom_prod/com.sankuai.data.hadoop.gpu/ai-search/...` native**, not `phxmlp.mtjupyter.singleuser/...` republished. Same-namespace base ↔ same-namespace push target means **no cross-mount** and sidesteps the `MANIFEST_BLOB_UNKNOWN` trap (Step 4).
- [ ] **Python / CUDA / torch stack** matches project needs (verl needs cuda 12.8, torch ≥2.8, python 3.12 as of 2026-04 — verify against current `requirements-cuda.txt`).

If the only viable base is in `phxmlp.mtjupyter.singleuser/...`, **pre-commit to `--no-cache`** — don't burn a push cycle to "see if it works".

## Step 2 — Dockerfile content pattern

The dpo_trl Dockerfile pasted into MLP "Dockerfile 定义":

```dockerfile
# Build-time proxy (internal pip egress). Cleared at the end.
ENV http_proxy=http://10.229.18.27:8412
ENV https_proxy=http://10.229.18.27:8412

RUN pip install --no-cache-dir \
      "trl==0.29.0" \
      "deepspeed==0.18.9" \
      "accelerate>=1.4.0" \
      "datasets>=3.0.0"

RUN pip install --no-cache-dir \
      math-verify latex2sympy2-extended pylatexenc

# Smoke-check — fails the BUILD if any required lib is broken.
# Catches bad layer cache / missing transitive deps before push, not in prod.
RUN python -c "import torch, vllm, trl, deepspeed, transformers, accelerate, datasets; \
from trl import DPOConfig, DPOTrainer; \
import math_verify, latex2sympy2_extended, pylatexenc; \
print('torch', torch.__version__, 'vllm', vllm.__version__, 'trl', trl.__version__); \
print('DPO import OK')"

# Runtime ENV — applies to running container, not the RUN steps above.
ENV HF_HUB_OFFLINE=1
ENV VLLM_USE_V1=1
ENV VLLM_NO_USAGE_STATS=1
ENV PYTHONUNBUFFERED=1

# Clear proxy so runtime doesn't accidentally route through build-time proxy.
ENV http_proxy=
ENV https_proxy=
```

Rules encoded:
- **Proxy at top, cleared at bottom.** `10.229.18.27:8412` is the internal pip egress — it does NOT work at runtime.
- **Always `--no-cache-dir`.** Base is already ~33GB; each pip cache adds 2-5GB. Repeated on EVERY `pip install` RUN.
- **Pin versions with `==`**, use `>=` only when minor drift is safe. TRL's API changed between 0.23 → 0.29; pinning saved rework.
- **Final smoke `RUN python -c "import ..."`** — listing every lib you care about. Build fails here if anything is broken, which is much faster than debugging at runtime.
- **Bake runtime ENV last.** Setting `HF_HUB_OFFLINE=1` before pip installs would make pip unable to fetch from HF.
- **No CUDA kernel calls in `RUN`.** Build nodes have no GPU; `torch.cuda.is_available()` is False; flash-attn kernel compile would crash. Defer all kernel smoke tests to the probe.
- **No `COPY` of source code or model weights.** Mount from dolphinfs at runtime — keeps image ≤ base + deps.

## Step 3 — Submit build: MLP form quirks

- **Pick "Dockerfile" type**, not pip/yum/conda item-by-item.
- **Delete the auto-expanded empty 软件包 input box.** An empty field crashes the build.
- **Advanced params box:** leave GPU `deviceplugin.*` empty (build runners have no GPU — those flags just eat queue quota).
- **Failure diagnostic panel:** read "运行日志", not "分层构建进度" (the latter updates lazily).

## Step 4 — Push-stage failure recovery (verified 2026-04-17)

Symptom:
```
Pushing image to ...<target>:1.0.0
Retrying ... MANIFEST_BLOB_UNKNOWN: blob unknown to registry; sha256:<hash>
```
3x retry, all same blob hash → confirmed persistent, not transient.

Cause: Kaniko tries cross-repo blob mount (avoids re-uploading the 30GB base) and the target namespace doesn't have that blob. Happens when base lives under a different top-level namespace than the push target (e.g. `phxmlp.mtjupyter.singleuser/*` base → `data.hadoop.gpu/ai-search/*` target).

**Winning fix: advanced params `--build-on-machine --no-cache`**, confirmed working on `dpo_trl_174de81b`:
- `--no-cache` forces every layer to be uploaded fresh to the target namespace → no cross-mount needed.
- `--build-on-machine` runs the builder on a physical machine with proper registry egress.
- Manifest PUT may still fail 1-2 times inside kaniko, but retries succeed on attempt 2-3 (the builder itself doesn't exit after first failure).
- Cost: rebuild is slower (~all layers re-uploaded, 30+ min); target image is larger on registry (no layer dedup with base). Acceptable one-time cost.

Fallback order if `--build-on-machine --no-cache` somehow still fails:
1. Retry the submission once unchanged — occasionally transient.
2. Try an older version of the same base image family (detail page: 版本数: N). Blob hashes differ between versions; `1.0.1` / `1.0.0` may push where `1.0.2` can't.
3. Switch base to one in the same target namespace. Validate the new base with an **empty Dockerfile (ENV-only) build+push first**, before layering real deps on it.
4. Platform-ops escalation — give them target path + failing blob sha256; ask for cross-mount grant or manual blob copy.

## Step 5 — Validate via probe (mandatory)

Before running real training, submit `platform/probe/run.hope`. Rationale: the build-node and the runtime-node are different environments (GPU, driver, runtime libs) — a Dockerfile smoke RUN passing does not mean runtime passes.

The probe covers:
- Baked ENV sanity (HF_HUB_OFFLINE=1 etc.)
- Pinned-version assertions per lib
- CUDA kernel roundtrip (`torch.randn(...).cuda() @ ...`)
- DPOConfig / class-level smoke (catches TRL API drift)
- math_verify functional check
- vLLM import (no engine spin-up — too slow)
- `HF_HUB_OFFLINE=1` offline tokenizer load (the single most common runtime trap when HF_HUB_OFFLINE is baked in)
- Collects all failures; single SUMMARY block; rc=1 if any FAIL.

≤5 min on 1 GPU. Do **not** skip. Expected output on success: `ALL PROBE CHECKS PASSED` + rc=0.

## Step 6 — Stale-copy + dolphinfs hazards

- **`hope_dir/` does NOT auto-sync** with the repo. After every `git pull`:
  ```bash
  cp platform/run.hope platform/probe/run.hope \
     platform/jupyter.sh platform/probe/jupyter.sh \
     $LGX/hope_dir/
  md5sum $LGX/hope_dir/*.sh $LGX/hope_dir/*.hope    # sanity
  ```
- **Login-machine dolphinfs view can be extremely laggy or even diverge from the container view.**
  Observed 2026-04-17: container wrote `beacons/probe_*.txt` and `logs/probe_*.log` successfully (visible in UI `运行日志`), but the login machine `set-zw06-mlp-codelab-pc178` showed neither file 10+ min later; `find /mnt/dolphinfs ...` hung. Lesson: **MLP UI "运行日志" is the authoritative output stream. Absence of a file on the login machine does not prove the container didn't write it.** For real results, rely on UI logs first; treat dolphinfs files as best-effort persistence.

## Step 7 — Pre-flight for the next image (verl, `/data-1/verl07/verl`)

Before the AI drafts a verl Dockerfile, the user and AI need to agree on:

1. **Target registry path.** Likely `custom_prod/com.sankuai.data.hadoop.gpu/ai-search/training_..._verl_...:1.0.x`. User confirms final string.
2. **Base image.** Candidates:
   - `...ai-search_training_..._verl_megatron_1.0.2_534ef92c` (already pulled successfully for dpo_trl; verl 0.8.0.dev target).
   - A newer version if MLP UI shows one. User pastes path from UI — AI does not guess.
3. **Delta packages.** AI reads `/data-1/verl07/verl/requirements.txt`, `requirements-cuda.txt`, `pyproject.toml` and drafts a pinned-version list. User reviews before build.
4. **Smoke RUN content.** Equivalent of the dpo_trl `python -c "import ..."` line but for verl — probably `from verl.trainer import ...` and a `from verl.workers...` import. User sanity-checks symbols actually exist in the current verl checkout.
5. **Runtime ENV.** verl typically needs different flags than TRL (Ray cluster config, Megatron paths, possibly different `VLLM_*` flags). List before baking.
6. **Probe script.** Clone `platform/probe/jupyter.sh` pattern; replace the TRL/DPO block with verl-equivalent checks (verl imports, Megatron availability if bundled, etc.).

User action before AI drafts the Dockerfile:
- [ ] Paste the exact base-image path from MLP UI.
- [ ] Confirm verl's minimum torch / vllm / transformers versions from `requirements*.txt`.
- [ ] Decide if Megatron is required (image gets much larger).
- [ ] Decide target tag scheme (version / date / git hash).
- [ ] Tell AI the target registry path.

Once confirmed, AI drafts Dockerfile + probe smoke-check block, user reviews and pastes into MLP "新建镜像", submits **with `--build-on-machine --no-cache` from the start** (saves one failed push cycle). Post-push, submit the probe first; only after probe rc=0 run real jobs.
