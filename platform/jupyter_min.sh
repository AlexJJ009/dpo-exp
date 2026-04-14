#!/usr/bin/env bash
# Minimal launcher that writes a beacon file after EVERY step, so even if
# the script dies we know exactly which step killed it.
# Upload this as jupyter.sh (rename) for a pure diagnosis run.

BEACON_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yangfengkai02/lgx/beacons"
mkdir -p "${BEACON_DIR}" 2>/dev/null

TS=$(date +%Y%m%d_%H%M%S)
B="${BEACON_DIR}/beacon_${TS}_$$"

touch "${B}.00_started" 2>/dev/null

{
  echo "=== $(date -Is) ==="
  echo "hostname=$(hostname)"
  echo "whoami=$(whoami)"
  echo "pwd=$(pwd)"
  echo "SHELL=${SHELL}"
  echo "BASH_VERSION=${BASH_VERSION}"
} > "${B}.01_identity" 2>&1

{
  env | sort
} > "${B}.02_env" 2>&1

{
  echo "--- /mnt/dolphinfs ---"
  ls -la /mnt/dolphinfs/ 2>&1 | head -20
  echo "--- /home ---"
  ls -la /home/ 2>&1 | head -20
  echo "--- primary path ---"
  ls -la /mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yangfengkai02/lgx/ 2>&1 | head -20
  echo "--- alt path ---"
  ls -la /home/hadoop-ai-search/dolphinfs_ssd_hadoop-ai-search/yangfengkai02/lgx/ 2>&1 | head -20
} > "${B}.03_mounts" 2>&1

{
  which nvidia-smi && nvidia-smi 2>&1 || echo "no nvidia-smi"
  ls /dev/nvidia* 2>&1 || echo "no /dev/nvidia*"
} > "${B}.04_gpu" 2>&1

{
  which python3 && python3 --version 2>&1
  which pip && pip --version 2>&1
  pip list 2>&1 | head -50
} > "${B}.05_python" 2>&1

touch "${B}.99_done"

# Keep the container alive so you can inspect via platform / dolphinfs
echo "=== all beacons written, sleeping 2h for inspection ==="
sleep 7200
