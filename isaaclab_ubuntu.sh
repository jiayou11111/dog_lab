#!/usr/bin/env bash
set -euo pipefail

DOG_LAB_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${ISAACLAB_PATH:-}" ]]; then
    for CANDIDATE in "${DOG_LAB_PATH}/../IsaacLab" "${DOG_LAB_PATH}/../IsaacLab_v2" "${DOG_LAB_PATH}/../isaaclab_dog-main"; do
        if [[ -d "${CANDIDATE}/source/isaaclab" ]]; then
            ISAACLAB_PATH="$(cd "${CANDIDATE}" && pwd)"
            break
        fi
    done
fi

if [[ -z "${ISAACLAB_PATH:-}" || ! -d "${ISAACLAB_PATH}/source/isaaclab" ]]; then
    echo "[ERROR] ISAACLAB_PATH is not a valid Isaac Lab repository." >&2
    echo "        export ISAACLAB_PATH=/path/to/IsaacLab" >&2
    exit 1
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "[ERROR] No conda environment detected. Run 'conda activate <env>' first." >&2
    exit 1
fi

ISAACSIM_DIR="${ISAACLAB_PATH}/_isaac_sim"
CACHE_ROOT="${DOG_LAB_CACHE_ROOT:-${DOG_LAB_PATH}/.cache_isaac}"
DOG_LAB_HEADLESS="${DOG_LAB_HEADLESS:-0}"
DOG_LAB_CLEAN_GUI_ENV="${DOG_LAB_CLEAN_GUI_ENV:-1}"
DOG_LAB_LIVESTREAM="${DOG_LAB_LIVESTREAM:-0}"

if [[ "${DOG_LAB_HEADLESS}" == "stream" ]]; then
    DOG_LAB_HEADLESS=1
    DOG_LAB_LIVESTREAM="${DOG_LAB_LIVESTREAM:-2}"
    [[ "${DOG_LAB_LIVESTREAM}" == "0" ]] && DOG_LAB_LIVESTREAM=2
fi

mkdir -p "${CACHE_ROOT}/xdg" "${CACHE_ROOT}/warp" "${CACHE_ROOT}/ov" "${CACHE_ROOT}/kit" "${CACHE_ROOT}/pip" "${CACHE_ROOT}/logs"

if [[ ! -d "${ISAACSIM_DIR}" ]]; then
    echo "[ERROR] Isaac Sim directory not found: ${ISAACSIM_DIR}" >&2
    exit 1
fi

if [[ "${DOG_LAB_HEADLESS}" == "0" && "${DOG_LAB_CLEAN_GUI_ENV}" == "1" ]]; then
    SAVED_DISPLAY="${DISPLAY:-}"
    SAVED_XAUTHORITY="${XAUTHORITY:-}"
    SAVED_XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-}"

    unset PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
    unset QT_PLUGIN_PATH QML2_IMPORT_PATH QT_QPA_PLATFORM QT_QPA_PLATFORM_PLUGIN_PATH
    unset ROS_VERSION ROS_PYTHON_VERSION ROS_DISTRO ROS_ROOT ROS_PACKAGE_PATH CMAKE_PREFIX_PATH PKG_CONFIG_PATH
    unset GAZEBO_PLUGIN_PATH GAZEBO_MODEL_PATH GAZEBO_RESOURCE_PATH MUJOCO_GL MUJOCO_PATH Torch_DIR CUDA_HOME

    export DISPLAY="${SAVED_DISPLAY}"
    export XAUTHORITY="${SAVED_XAUTHORITY}"
    export XDG_RUNTIME_DIR="${SAVED_XDG_RUNTIME_DIR}"
fi

set +u
source "${ISAACSIM_DIR}/setup_conda_env.sh"
set -u

export ISAACLAB_PATH DOG_LAB_PATH
export PYTHONPATH="${DOG_LAB_PATH}:${ISAACLAB_PATH}/source/isaaclab:${ISAACLAB_PATH}/source/isaaclab_tasks:${ISAACLAB_PATH}/source/isaaclab_assets:${ISAACLAB_PATH}/source/isaaclab_rl:${PYTHONPATH:-}"

if [[ "${DOG_LAB_HEADLESS}" == "0" ]]; then
    unset HEADLESS ENABLE_HEADLESS CARB_WINDOWING_PLUGIN OMNI_KIT_DISABLE_X11 QT_QPA_PLATFORM
    export QT_X11_NO_MITSHM=1
else
    unset DISPLAY
    export HEADLESS=1
    export ENABLE_HEADLESS=1
    export CARB_WINDOWING_PLUGIN="null"
    export OMNI_KIT_DISABLE_X11=1
    export QT_QPA_PLATFORM=offscreen
fi

[[ "${DOG_LAB_LIVESTREAM}" != "0" ]] && export LIVESTREAM="${DOG_LAB_LIVESTREAM}"

export OMNI_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export WARP_CACHE_DIR="${CACHE_ROOT}/warp"
export OV_USER_CACHE_DIR="${CACHE_ROOT}/ov"
export OMNI_USER_CACHE_DIR="${CACHE_ROOT}/ov"
export KIT_CACHE_DIR="${CACHE_ROOT}/kit"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __NV_PRIME_RENDER_OFFLOAD=1
export MESA_LOADER_DRIVER_OVERRIDE=nvidia

for NVIDIA_ICD in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; do
    if [[ -f "${NVIDIA_ICD}" ]]; then
        export VK_ICD_FILENAMES="${NVIDIA_ICD}"
        break
    fi
done

GUI_EXPERIENCE_FILE=""
if [[ "${DOG_LAB_HEADLESS}" == "0" ]]; then
    GUI_EXPERIENCE_FILE="${ISAACLAB_PATH}/apps/isaaclab.python.no_telemetry.kit"
    awk '
        $0 == "\"omni.kit.telemetry\" = {}" { next }
        $0 == "\"omni.physx.bundle\" = {}" {
            print "\"omni.physx\" = {}"
            print "\"omni.physx.tensors\" = {}"
            print "\"omni.physx.fabric\" = {}"
            print "\"omni.physx.stageupdate\" = {}"
            print "\"omni.physx.commands\" = {}"
            print "\"omni.usdphysics\" = {}"
            next
        }
        $0 ~ /^enableAnonymousAppName = / { print "enableAnonymousAppName = false"; next }
        $0 ~ /^enableAnonymousData = / { print "enableAnonymousData = false"; next }
        /exts\."omni.kit.viewport.window"\.windowMenu\.entryCount = / { print "exts.\"omni.kit.viewport.window\".windowMenu.entryCount = 1"; next }
        { print }
    ' "${ISAACLAB_PATH}/apps/isaaclab.python.kit" > "${GUI_EXPERIENCE_FILE}"
fi

cd "${DOG_LAB_PATH}"

echo "[INFO] DOG_LAB_PATH      = ${DOG_LAB_PATH}"
echo "[INFO] ISAACLAB_PATH     = ${ISAACLAB_PATH}"
echo "[INFO] ISAACSIM_DIR      = ${ISAACSIM_DIR}"
echo "[INFO] Headless          = ${DOG_LAB_HEADLESS}"
echo "[INFO] GUI experience    = ${GUI_EXPERIENCE_FILE:-default}"
echo "[INFO] Python executable = ${CONDA_PREFIX}/bin/python"

if [[ "${1:-}" == "-p" ]]; then
    shift
    PY_ARGS=("$@")
    if [[ -n "${GUI_EXPERIENCE_FILE}" && "${#PY_ARGS[@]}" -gt 0 && "${PY_ARGS[0]}" != "-c" && "${PY_ARGS[0]}" != "-m" ]]; then
        PY_ARGS+=("--experience" "${GUI_EXPERIENCE_FILE}")
        PY_ARGS+=("--kit_args" "--disable omni.kit.telemetry --disable omni.physx.telemetry --disable omni.physx.clashdetection.telemetry --disable omni.kit.browser.sample --/settings/telemetry/enableAnonymousAppName=false --/settings/telemetry/enableAnonymousData=false --/exts/omni.warp/enable_menu=false")
    fi
    exec "${CONDA_PREFIX}/bin/python" "${PY_ARGS[@]}"
fi

exec "$@"
# ./isaaclab_ubuntu.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task DogLab-Go2W-Piper-Flat-Play-v0 --num_envs 1 --checkpoint /home/ymy/isaac_s
# torage/projects/dog/dog_lab/output_total/model_15000.pt

# ./isaaclab_ubuntu.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task DogLab-Go2W-Piper-Flat-v0 --num_envs 1