#!/usr/bin/env bash
set -euo pipefail

DOG_LAB_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_PATH="${ISAACLAB_PATH:-$(cd "${DOG_LAB_PATH}/../isaaclab_dog-main" && pwd)}"
CACHE_ROOT="${DOG_LAB_CACHE_ROOT:-${DOG_LAB_PATH}/../.cache_isaac}"

if [[ ! -f "${ISAACLAB_PATH}/isaaclab.sh" ]]; then
  echo "[ERROR] IsaacLab not found at: ${ISAACLAB_PATH}" >&2
  echo "Set ISAACLAB_PATH=/path/to/isaaclab before running this script." >&2
  exit 1
fi

mkdir -p "${CACHE_ROOT}/xdg" "${CACHE_ROOT}/warp" "${CACHE_ROOT}/ov"

cd "${DOG_LAB_PATH}"

exec env -i \
  HOME="${HOME}" \
  USER="${USER:-}" \
  LOGNAME="${LOGNAME:-${USER:-}}" \
  SHELL="/bin/bash" \
  PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  DISPLAY="${DISPLAY:-}" \
  XAUTHORITY="${XAUTHORITY:-}" \
  XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-}" \
  XDG_SESSION_TYPE="x11" \
  LANG="${LANG:-C.UTF-8}" \
  VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}" \
  XDG_CACHE_HOME="${CACHE_ROOT}/xdg" \
  WARP_CACHE_DIR="${CACHE_ROOT}/warp" \
  OV_USER_CACHE_DIR="${CACHE_ROOT}/ov" \
  ISAACLAB_PATH="${ISAACLAB_PATH}" \
  DOG_LAB_PATH="${DOG_LAB_PATH}" \
  PYTHONPATH="${DOG_LAB_PATH}" \
  bash "${ISAACLAB_PATH}/isaaclab.sh" "$@"
