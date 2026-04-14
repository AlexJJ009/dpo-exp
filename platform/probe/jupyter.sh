#!/usr/bin/env bash
# Minimal probe script -- prove that container starts, dolphinfs is mounted,
# and worker.script actually runs. No deps, no installs, no heavy work.
set -x

echo "=== PROBE START $(date -Is) ==="
echo "hostname=$(hostname)"
echo "whoami=$(whoami)"
echo "pwd=$(pwd)"
echo "id=$(id)"
echo "uname=$(uname -a)"

# What's in the working dir (should contain run.hope + probe.sh)
ls -la

# Dolphinfs write test -- the two common mount roots
LGX_DIR="/home/hadoop-ai-search/dolphinfs_ssd_hadoop-ai-search/yangfengkai02/lgx"
ALT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yangfengkai02/lgx"

for D in "${LGX_DIR}" "${ALT_DIR}"; do
  echo "--- checking: $D ---"
  ls -la "$D" 2>&1 | head -20 || true
  if [ -d "$D" ]; then
    mkdir -p "$D/logs" 2>&1 || true
    OUT="$D/logs/probe_$(hostname)_$(date +%s).txt"
    echo "hello from $(hostname) at $(date -Is)" > "$OUT" 2>&1
    echo "wrote: $OUT"
    ls -la "$OUT" 2>&1 || true
  fi
done

# GPU visibility (if the base image has nvidia-smi -- may not)
which nvidia-smi && nvidia-smi || echo "no nvidia-smi (ok, this is a test image)"

echo "=== PROBE SLEEP 2h -- inspect state via platform then stop the job ==="
sleep 7200
