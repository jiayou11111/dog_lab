#!/usr/bin/env bash
set -euo pipefail

DOG_LAB_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_PATH="${ISAACLAB_PATH:-$(cd "${DOG_LAB_PATH}/../isaaclab_dog-main" && pwd)}"
ISAACSIM_DIR="${ISAACLAB_PATH}/_isaac_sim"
CACHE_ROOT="${DOG_LAB_CACHE_ROOT:-${DOG_LAB_PATH}/.cache_isaac}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "[ERROR] No conda environment detected"
    exit 1
fi

mkdir -p \
    "${CACHE_ROOT}/xdg" \
    "${CACHE_ROOT}/warp" \
    "${CACHE_ROOT}/ov" \
    "${CACHE_ROOT}/kit" \
    "${CACHE_ROOT}/pip" \
    "${CACHE_ROOT}/logs" \
    "${CACHE_ROOT}/pyshim/isaacsim"

cat > "${CACHE_ROOT}/pyshim/isaacsim/__init__.py" <<'PY'
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

try:
    from isaacsim.simulation_app import SimulationApp
except Exception:
    pass
PY

set +u
[[ -f "${ISAACSIM_DIR}/setup_conda_env.sh" ]] && source "${ISAACSIM_DIR}/setup_conda_env.sh"
[[ -f "${ISAACSIM_DIR}/setup_python_env.sh" ]] && source "${ISAACSIM_DIR}/setup_python_env.sh"
set -u

ISAAC_EXT_PYTHONPATH=""

for EXT_ROOT in \
    "${ISAACSIM_DIR}/exts" \
    "${ISAACSIM_DIR}/extscache" \
    "${ISAACSIM_DIR}/extsPhysics" \
    "${ISAACSIM_DIR}/kit/exts" \
    "${ISAACSIM_DIR}/kit/extscore"
do
    if [[ -d "${EXT_ROOT}" ]]; then
        while IFS= read -r -d '' EXT_DIR; do
            ISAAC_EXT_PYTHONPATH="${EXT_DIR}:${ISAAC_EXT_PYTHONPATH}"
        done < <(find -L "${EXT_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0)
    fi
done

export PYTHONPATH="${CACHE_ROOT}/pyshim:${ISAACSIM_DIR}/exts/isaacsim.simulation_app:${ISAACSIM_DIR}/kit/kernel/py:${ISAACSIM_DIR}/kit/python/lib/python3.10:${ISAAC_EXT_PYTHONPATH}:${DOG_LAB_PATH}:${ISAACLAB_PATH}/source/isaaclab:${ISAACLAB_PATH}/source/isaaclab_tasks:${ISAACLAB_PATH}/source/isaaclab_assets:${ISAACLAB_PATH}/source/isaaclab_rl"

export ISAACLAB_PATH="${ISAACLAB_PATH}"
export DOG_LAB_PATH="${DOG_LAB_PATH}"
export ISAAC_PATH="${ISAACSIM_DIR}"
export CARB_APP_PATH="${ISAACSIM_DIR}/kit"
export EXP_PATH="${ISAACSIM_DIR}/apps"

ISAAC_LD_LIBRARY_PATH=""

for LIB_DIR in \
    "${ISAACSIM_DIR}/kit" \
    "${ISAACSIM_DIR}/kit/lib" \
    "${ISAACSIM_DIR}/kit/kernel/plugins" \
    "${ISAACSIM_DIR}/kit/plugins" \
    "${ISAACSIM_DIR}/kit/plugins/bindings-python"
do
    if [[ -d "${LIB_DIR}" ]]; then
        ISAAC_LD_LIBRARY_PATH="${LIB_DIR}:${ISAAC_LD_LIBRARY_PATH}"
    fi
done

for LIB_NAME in \
    "libarch.so*" \
    "libomni.usd.so*" \
    "libomniAudioSchema.so*" \
    "libusd_ms.so*" \
    "libusd.so*"
do
    while IFS= read -r -d '' LIB_FILE; do
        LIB_DIR="$(dirname "${LIB_FILE}")"

        case "${LIB_DIR}" in
            *ros1.bridge*|*ros2.bridge*|*noetic*|*humble*)
                ;;
            *)
                ISAAC_LD_LIBRARY_PATH="${LIB_DIR}:${ISAAC_LD_LIBRARY_PATH}"
                ;;
        esac
    done < <(find -L "${ISAACSIM_DIR}" -type f -name "${LIB_NAME}" -print0)
done

export LD_LIBRARY_PATH="${ISAAC_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"

# ===== 强制 headless EGL =====

unset DISPLAY

export HEADLESS=1
export ENABLE_HEADLESS=1

export CARB_WINDOWING_PLUGIN="null"
export OMNI_KIT_DISABLE_X11=1

export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __NV_PRIME_RENDER_OFFLOAD=1

export MESA_LOADER_DRIVER_OVERRIDE=nvidia
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

export QT_QPA_PLATFORM=offscreen

# ===== 禁止 telemetry =====

export OMNI_DISABLE_TELEMETRY=1

# ===== cache =====

export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export WARP_CACHE_DIR="${CACHE_ROOT}/warp"
export OV_USER_CACHE_DIR="${CACHE_ROOT}/ov"
export OMNI_USER_CACHE_DIR="${CACHE_ROOT}/ov"
export KIT_CACHE_DIR="${CACHE_ROOT}/kit"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"

cd "${DOG_LAB_PATH}"

echo "[INFO] DOG_LAB_PATH      = ${DOG_LAB_PATH}"
echo "[INFO] ISAACLAB_PATH     = ${ISAACLAB_PATH}"
echo "[INFO] ISAACSIM_DIR      = ${ISAACSIM_DIR}"
echo "[INFO] Python executable = ${CONDA_PREFIX}/bin/python"

if [[ "${1:-}" == "-p" ]]; then
    shift
    exec "${CONDA_PREFIX}/bin/python" "$@"
else
    exec "$@"
fi