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
command -v "${python_bin}" >/dev/null || die "missing ${python_bin}"
command -v nvidia-smi >/dev/null || die "missing NVIDIA driver"
nvidia-smi >/dev/null || die "nvidia-smi cannot communicate with the GPU"

apt-get update
apt_install git python3-venv

install -d -o root -g root -m 0755 /opt/robot-platform
"${python_bin}" -m venv "${TRAIN_ENV_ROOT}"
"${TRAIN_ENV_ROOT}/bin/pip" install --upgrade pip
"${TRAIN_ENV_ROOT}/bin/pip" install \
  "lerobot[core_scripts,training] @ git+https://github.com/huggingface/lerobot.git@${LEROBOT_GIT_REF}"

chown -R "${TRAIN_USER}:${DATA_GROUP}" "${TRAIN_ENV_ROOT}"
runuser -u "${TRAIN_USER}" -- "${TRAIN_ENV_ROOT}/bin/python" -c \
  "import torch; assert torch.cuda.is_available(), 'PyTorch cannot access CUDA'; import lerobot"

log "shared training environment installed at ${TRAIN_ENV_ROOT}"
