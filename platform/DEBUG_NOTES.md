# Platform (Meituan MLP / AFO hope) Debugging Notes

Last updated: 2026-04-17. Written as a handoff so work can continue from another machine.

> **Current state.** Custom image `...ai-search/training_ubuntu22_cuda12.8_python3.12_torch2.8_dpo_trl_174de81b:1.0.0` is built, pushed, and validated by `platform/probe/` (all checks [OK], rc=0). `platform/run.hope` points at it and is configured for a 2-GPU smoke run of the existing code-DPO experiment via `experiments/run_4b_code_m1_dpo_smoke.sh`. The next image to build is the verl image for `/data-1/verl07/verl` — follow the Image-build playbook appendix at the bottom.

## Context

- Goal: run the DPO pipeline on Meituan MLP via `ml-easy-job`.
- Queue: `root.shxs_training_cluster.hadoop-fridayagi.friday_h20_train`
- Hadoop account: `hadoop-ai-search`, MIS user: `yangfengkai02`
- Submission client host: `set-zw06-mlp-codelab-pc178`
- Dolphinfs base (primary): `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yangfengkai02/lgx/`
- Dolphinfs base (symlink): `/home/hadoop-ai-search/dolphinfs_ssd_hadoop-ai-search/yangfengkai02/lgx/`
- **Path guarantee.** Everything under `lgx/` has consistent sub-folder names across machines; the absolute prefix up to `lgx/` may differ. **All scripts must derive `LGX_DIR` from their own location, never hardcode the absolute path.** See `platform/jupyter.sh` for the pattern.
- Key paths under `lgx/`:
  - `dpo-exp/` — git repo (this)
  - `dpo-wheels/` — pre-built offline wheels (legacy, not needed for the new image)
  - `hope_dir/` — submission staging dir; `cp platform/*.sh platform/*.hope` targets
  - `logs/` — created on first run by `jupyter.sh`
  - `beacons/` — launcher start/done beacons
  - `dataset/` — training & eval data (user transfers via dolphinfs)
  - `checkpoints/` — model weights (user transfers via dolphinfs)

## MLP / AFO quirks that still bite

### hope_dir files go stale
`hope_dir/run.hope` and `hope_dir/jupyter.sh` do NOT auto-update when you `git pull` in `dpo-exp/`. Always re-copy and sanity-check before resubmitting:

```bash
cd $LGX/dpo-exp && git pull
cp platform/run.hope platform/jupyter.sh \
   platform/probe/run.hope platform/probe/jupyter.sh \
   $LGX/hope_dir/
md5sum $LGX/hope_dir/*.sh $LGX/hope_dir/*.hope
grep afo.docker.image.name $LGX/hope_dir/run.hope          # verify image
grep afo.docker.image.name $LGX/hope_dir/probe/run.hope 2>/dev/null || true
```

### Platform UI hides the Worker log under "driver"
Job detail → 实例 → Worker row → 日志: the role dropdown defaults to `driver`, which does not exist for `ml-easy-job`. Switch to `worker`. Same for `更多 ▾` menu — often only "容错" is shown; click into attempts to see failure reasons.

### Pod Events are not directly accessible
No `kubectl describe pod`-style UI. Two workarounds:
- Search the AM log for `PodChangeMonitor update:` lines — they contain `reason=<...>` and `message=<...>` explaining why worker failed.
- Ask platform ops (search 大象/飞书 for "MLP 支持" / "AFO 运维") to run `kubectl describe pod <jobId>-worker-0 -n hadoop-fridayagi`.

### run.hope / ConfigParser is intolerant
- Non-ASCII chars (esp. U+2014 em-dash `—`) crash `configparser`. Stick to pure ASCII. Use `--` or `-` instead.
- `#` comments inside a section once triggered parsing errors. Safest: no comments at all in `run.hope`.

### jobId URL must not have trailing quote
A URL like `.../jobId=psx6jd9xrjzmgsfz%22` (`%22` is `"`) gives spurious 500s — the trailing quote came from a copy-paste. Type the jobId cleanly.

### Image pulls can be slow but finite
Non-cached ~30GB images take 10-30 min on first pull. **4+ days stuck = NOT slow pull**, it's `ErrImagePull` / `ImagePullBackOff` (registry prefix missing, wrong tag, wrong repo hash). Check the image name in `run.hope` character-by-character.

### Login-machine dolphinfs view can lag / diverge
Observed 2026-04-17: container wrote `beacons/probe_*.txt` and `logs/probe_*.log` successfully (visible in UI "运行日志"), but the login machine `set-zw06-mlp-codelab-pc178` showed neither file 10+ min later; `find /mnt/dolphinfs ...` hung. **MLP UI "运行日志" is the authoritative output stream.** Absence of a file on the login machine does not prove the container didn't write it.

## Useful commands on the submission client

```bash
# Always stop orphans before resubmitting — they eat queue quota
hope killjob <run_id>

# Submit
cd $LGX/hope_dir && hope run run.hope             # main
cd $LGX/hope_dir && hope run probe/run.hope       # probe (validate image first)

# Check results after 5-15 min
ls -lat $LGX/logs/
ls -lat $LGX/beacons/
tail -300 $(ls -t $LGX/logs/run_*.log | head -1)
```

## Failure timeline (lessons we paid for)

| Image / attempt | Outcome | Lesson |
|---|---|---|
| sglang_dev_19d35de6:1.0.0 | `su: Authentication failure` in docker-run.sh; worker never reaches our script | Not every MLP-visible image is AFO-compatible; test with a probe first. |
| sglang_dev:1.0.0 (no registry prefix) | 4 days stuck "资源已分配" | `ImagePullBackOff` masked as slow scheduling. Full registry path + exact tag is required. |
| image build #1 `dpo_trl_8071ad6f` | build OK, push fails 3× with `MANIFEST_BLOB_UNKNOWN: sha256:4f4fb700...` | Cross-namespace blob cross-mount denied when base lives in `phxmlp.mtjupyter.singleuser/*` and target is `data.hadoop.gpu/ai-search/*`. Retry / rename tricks don't help. |
| image build #2 `dpo_trl_174de81b` | push succeeds after 2-3 internal kaniko retries | Advanced params `--build-on-machine --no-cache` force fresh upload to target namespace → no cross-mount. **Use these params from the start next time.** |
| DPO-Probe-newimage (174de81b) | `ALL PROBE CHECKS PASSED`, rc=0 | Image validated end-to-end (TRL 0.29.0 / DeepSpeed 0.18.9 / math-verify / CUDA kernel / HF_HUB_OFFLINE tokenizer load). |

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

## Step 6 — Path discipline in launcher scripts

Only the folder layout under `lgx/` is guaranteed consistent — the absolute prefix up to `lgx/` may differ across machines. So:

- `scripts/config.sh` auto-resolves `REPO_DIR = $(cd "${_CONFIG_DIR}/.." && pwd)` → `BASE_DIR = parent of REPO_DIR`. All dataset / checkpoint paths derive from there. No machine-specific overrides needed.
- `platform/jupyter.sh` resolves `LGX_DIR` from `$(dirname ${BASH_SOURCE[0]})/..` — assumes it sits at `lgx/hope_dir/jupyter.sh`. **Do not reintroduce hardcoded `/home/hadoop-ai-search/...` absolute paths.** The `cp platform/jupyter.sh $LGX/hope_dir/` convention preserves the sibling relationship.
- Experiment wrappers (`experiments/run_*.sh`) use `source "$(dirname "$0")/../scripts/config.sh"` — same auto-detection.

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
