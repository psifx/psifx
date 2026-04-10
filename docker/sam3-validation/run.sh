#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-psifx-sam3-validation:latest}"
VIDEO_PATH="${VIDEO_PATH:-${ROOT_DIR}/example/data/Video.mp4}"
USE_GPU="${USE_GPU:-1}"
CONTAINER_NAME="${CONTAINER_NAME:-psifx-sam3-validation-run}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${ROOT_DIR}/.tmp/hf-cache}"
VALIDATION_WORK_DIR="${VALIDATION_WORK_DIR:-${ROOT_DIR}/sam3_test_outputs/stable-run}"
SHORT_FRAMES="${SHORT_FRAMES:-2}"
GPU_TIMEOUT_SEC="${GPU_TIMEOUT_SEC:-900}"
CPU_TIMEOUT_SEC="${CPU_TIMEOUT_SEC:-900}"

mkdir -p "${HF_CACHE_DIR}" "${VALIDATION_WORK_DIR}"

echo "Building ${IMAGE_NAME}"
docker build -f "${ROOT_DIR}/docker/sam3-validation/Dockerfile" -t "${IMAGE_NAME}" "${ROOT_DIR}"

# Ensure stale validation container from previous interrupted runs is gone.
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

DOCKER_ARGS=(
  --name "${CONTAINER_NAME}"
  --rm
  --mount "type=bind,source=${ROOT_DIR},target=/workspace"
  --mount "type=bind,source=${HF_CACHE_DIR},target=/root/.cache/huggingface"
  -e "VALIDATION_WORK_DIR=/workspace/sam3_test_outputs/stable-run"
  -e "SHORT_FRAMES=${SHORT_FRAMES}"
  -e "GPU_TIMEOUT_SEC=${GPU_TIMEOUT_SEC}"
  -e "CPU_TIMEOUT_SEC=${CPU_TIMEOUT_SEC}"
)

if [[ -t 1 ]]; then
  DOCKER_ARGS+=( --interactive --tty )
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  DOCKER_ARGS+=( -e "HF_TOKEN=${HF_TOKEN}" )
fi

if [[ "${USE_GPU}" == "1" ]]; then
  DOCKER_ARGS+=( --gpus all )
fi

docker run "${DOCKER_ARGS[@]}" "${IMAGE_NAME}" \
  /bin/bash -lc "/workspace/docker/sam3-validation/validate_sam3.sh '${VIDEO_PATH}'"
