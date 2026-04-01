#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace"
WORK_DIR="${VALIDATION_WORK_DIR:-/workspace/sam3_test_outputs/stable-run}"
VIDEO_IN="${1:-${ROOT_DIR}/example/data/Video.mp4}"
SHORT_FRAMES="${SHORT_FRAMES:-2}"
GPU_TIMEOUT_SEC="${GPU_TIMEOUT_SEC:-900}"
CPU_TIMEOUT_SEC="${CPU_TIMEOUT_SEC:-900}"
RUN_VIS="${RUN_VIS:-0}"

VIDEO_SHORT="${WORK_DIR}/Video-short.mp4"
CPU_MASK_DIR="${WORK_DIR}/masks-cpu"
GPU_MASK_DIR="${WORK_DIR}/masks-gpu"
CPU_VIS="${WORK_DIR}/visualization-cpu.mp4"
GPU_VIS="${WORK_DIR}/visualization-gpu.mp4"
CPU_LOG="${WORK_DIR}/cpu.log"
GPU_LOG="${WORK_DIR}/gpu.log"
SUMMARY="${WORK_DIR}/summary.txt"

mkdir -p "${WORK_DIR}"
rm -rf "${CPU_MASK_DIR}" "${GPU_MASK_DIR}" "${CPU_VIS}" "${GPU_VIS}" "${VIDEO_SHORT}" "${CPU_LOG}" "${GPU_LOG}" "${SUMMARY}"

source /opt/conda/etc/profile.d/conda.sh
conda activate psifx-env

echo "=== Environment ==="
python --version
psifx --version
python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

echo "=== CLI Help Checks ==="
psifx --help >/dev/null
psifx video tracking --help >/dev/null
psifx video tracking sam3 --help >/dev/null
psifx video tracking sam3 inference --help >/dev/null

if [[ ! -f "${VIDEO_IN}" ]]; then
  echo "Input video not found: ${VIDEO_IN}" >&2
  exit 2
fi

echo "=== Prepare Very Short Video (${SHORT_FRAMES} frames) ==="
ffmpeg -y -loglevel error -i "${VIDEO_IN}" -frames:v "${SHORT_FRAMES}" "${VIDEO_SHORT}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. SAM3 model download may fail if the repo is gated."
fi

input_frames=$(python - <<PY
import cv2
cap = cv2.VideoCapture("${VIDEO_SHORT}")
print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
cap.release()
PY
)

case_status=0
case_elapsed=0
case_mask_count=0

run_case() {
  local device="$1"
  local timeout_sec="$2"
  local mask_dir="$3"
  local log_file="$4"

  mkdir -p "${mask_dir}"
  rm -rf "${mask_dir:?}/"*

  local start_ts end_ts status elapsed mask_count
  start_ts=$(date +%s)

  set +e
  timeout --foreground "${timeout_sec}" psifx video tracking sam3 inference \
    --video "${VIDEO_SHORT}" \
    --mask_dir "${mask_dir}" \
    --text_prompt "people" \
    --chunk_size 2 \
    --iou_threshold 0.3 \
    --device "${device}" \
    --overwrite >"${log_file}" 2>&1
  status=$?
  set -e

  end_ts=$(date +%s)
  elapsed=$((end_ts - start_ts))
  mask_count=$(find "${mask_dir}" -maxdepth 1 -name '*.mp4' | wc -l | tr -d ' ')

  case_status="${status}"
  case_elapsed="${elapsed}"
  case_mask_count="${mask_count}"
}

gpu_status=125
gpu_elapsed=0
gpu_mask_count=0

echo "=== GPU SAM3 Inference ==="
if python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  run_case "cuda" "${GPU_TIMEOUT_SEC}" "${GPU_MASK_DIR}" "${GPU_LOG}"
  gpu_status="${case_status}"
  gpu_elapsed="${case_elapsed}"
  gpu_mask_count="${case_mask_count}"
else
  echo "CUDA is unavailable in this container; GPU SAM3 test skipped."
fi

cpu_status=0
cpu_elapsed=0
cpu_mask_count=0

echo "=== CPU SAM3 Inference ==="
run_case "cpu" "${CPU_TIMEOUT_SEC}" "${CPU_MASK_DIR}" "${CPU_LOG}"
cpu_status="${case_status}"
cpu_elapsed="${case_elapsed}"
cpu_mask_count="${case_mask_count}"

if [[ "${RUN_VIS}" == "1" ]]; then
  if [[ "${gpu_status}" -eq 0 && "${gpu_mask_count}" -gt 0 ]]; then
    psifx video tracking visualization \
      --video "${VIDEO_SHORT}" \
      --masks "${GPU_MASK_DIR}" \
      --visualization "${GPU_VIS}" \
      --device cuda \
      --overwrite || true
  fi
  if [[ "${cpu_status}" -eq 0 && "${cpu_mask_count}" -gt 0 ]]; then
    psifx video tracking visualization \
      --video "${VIDEO_SHORT}" \
      --masks "${CPU_MASK_DIR}" \
      --visualization "${CPU_VIS}" \
      --overwrite || true
  fi
fi

{
  echo "input_video=${VIDEO_SHORT}"
  echo "input_frames=${input_frames}"
  echo "cuda_exit_code=${gpu_status}"
  echo "cuda_elapsed_sec=${gpu_elapsed}"
  echo "cuda_mask_count=${gpu_mask_count}"
  echo "cpu_exit_code=${cpu_status}"
  echo "cpu_elapsed_sec=${cpu_elapsed}"
  echo "cpu_mask_count=${cpu_mask_count}"
} > "${SUMMARY}"

python - <<'PY' >> "${SUMMARY}"
from pathlib import Path
import cv2

work = Path("/workspace/sam3_test_outputs/stable-run")
for name in ["masks-gpu", "masks-cpu"]:
    d = work / name
    if not d.exists():
        continue
    for p in sorted(d.glob("*.mp4")):
        cap = cv2.VideoCapture(str(p))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"{name}:{p.name}:frames={frames}")
PY

echo "=== Validation Summary ==="
cat "${SUMMARY}"

if [[ "${gpu_status}" -ne 0 && "${gpu_status}" -ne 125 ]]; then
  exit "${gpu_status}"
fi
if [[ "${cpu_status}" -ne 0 ]]; then
  exit "${cpu_status}"
fi

