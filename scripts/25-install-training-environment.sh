#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

require_apply "${1:-}"
require_root
load_site_config
require_ubuntu

python_bin="${TRAIN_PYTHON_BIN:-python3}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
lerobot_git_url="${LEROBOT_GIT_URL:-https://github.com/huggingface/lerobot.git}"
command -v "${python_bin}" >/dev/null || die "missing ${python_bin}"
command -v nvidia-smi >/dev/null || die "missing NVIDIA driver"
nvidia-smi >/dev/null || die "nvidia-smi cannot communicate with the GPU"

apt-get update
# evdev and other C extensions need the interpreter's dev headers (Python.h);
# the venv module package is version-specific too. Both are skipped gracefully
# when the interpreter does not come from APT (e.g. conda).
py_version="$("${python_bin}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
# ffmpeg carries the libav* shared libraries torchcodec dlopens to decode the
# dataset videos. Nothing pulls it in as a dependency: torchcodec ships its own
# libtorchcodec_core*.so and only fails at the first training batch when the
# matching libav* is absent, so a node missing it looks healthy until then.
packages=(git build-essential ffmpeg)
for pkg in "python${py_version}-venv" "python${py_version}-dev"; do
  if apt-cache show "${pkg}" >/dev/null 2>&1; then
    packages+=("${pkg}")
  fi
done
apt_install "${packages[@]}"

install -d -o root -g root -m 0755 /opt/robot-platform
"${python_bin}" -m venv "${TRAIN_ENV_ROOT}"
"${TRAIN_ENV_ROOT}/bin/pip" install --upgrade pip
"${TRAIN_ENV_ROOT}/bin/pip" install \
  "lerobot[core_scripts,training,smolvla,pi] @ git+${lerobot_git_url}@${LEROBOT_GIT_REF}"

# Slurm sets HOME from ${TRAIN_USER}'s passwd entry, but that home exists only
# where an installer created it. Training jobs cache torch hub backbone weights
# and Hugging Face artifacts here instead, so every worker has a writable cache.
safe_install_dir "${PLATFORM_STATE_ROOT}/cache" "${TRAIN_USER}" "${DATA_GROUP}" 0750

chown -R "${TRAIN_USER}:${DATA_GROUP}" "${TRAIN_ENV_ROOT}"
runuser -u "${TRAIN_USER}" -- "${TRAIN_ENV_ROOT}/bin/python" -c \
  "import torch; assert torch.cuda.is_available(), 'PyTorch cannot access CUDA'; import lerobot"
# The default video backend loads its FFmpeg bindings lazily, inside a dataloader
# worker on the first batch. Import it here so a broken node fails the install.
runuser -u "${TRAIN_USER}" -- "${TRAIN_ENV_ROOT}/bin/python" -c \
  "from torchcodec.decoders import VideoDecoder"

log "shared training environment installed at ${TRAIN_ENV_ROOT}"
