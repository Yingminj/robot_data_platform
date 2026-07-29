#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

require_apply "${1:-}"
require_root
load_site_config
require_ubuntu

log "installing GPU worker prerequisites"
apt-get update
apt_install \
  ca-certificates chrony curl hdf5-tools jq munge nfs-common \
  openssh-server python3 rsync slurm-wlm

command -v nvidia-smi >/dev/null || die "NVIDIA driver is not installed; install and validate a compatible driver before continuing"
nvidia-smi >/dev/null || die "NVIDIA driver is installed but nvidia-smi cannot communicate with it"

ensure_docker
if ! command -v nvidia-container-cli >/dev/null 2>&1; then
  if apt-cache show nvidia-container-toolkit >/dev/null 2>&1; then
    apt_install nvidia-container-toolkit
  else
    die "nvidia-container-toolkit is unavailable; configure NVIDIA's APT repository and rerun"
  fi
fi
if command -v nvidia-ctk >/dev/null 2>&1; then
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

ensure_group "${DATA_GROUP}" "${DATA_GID}"
ensure_system_user "${TRAIN_USER}" "${DATA_GROUP}" "${TRAIN_UID}"
usermod -aG docker "${TRAIN_USER}"

safe_install_dir "${DATASET_CACHE_ROOT}" "${TRAIN_USER}" "${DATA_GROUP}" 0750
safe_install_dir "${EXPORT_CACHE_ROOT}" "${TRAIN_USER}" "${DATA_GROUP}" 0750
safe_install_dir "${RUN_WORK_ROOT}" "${TRAIN_USER}" "${DATA_GROUP}" 0750
safe_install_dir "${PLATFORM_STATE_ROOT}/slurmd" slurm slurm 0750

"${SCRIPT_DIR}/50-configure-nfs-mount.sh" rw --apply

systemctl enable --now docker chrony ssh
# Do not start authentication or scheduling with an unknown/incorrect key.
# Preserve a previously configured worker during idempotent re-runs.
if [[ ! -r /etc/slurm/slurm.conf && ! -r /etc/slurm-llnl/slurm.conf ]]; then
  systemctl disable --now munge slurmd >/dev/null 2>&1 || true
fi

log "GPU prerequisites installed; NVIDIA driver was not changed"
log "hardware line for config/nodes.conf:"
slurmd -C || true
log "next: securely install /etc/munge/munge.key and the reviewed slurm.conf, then enable munge/slurmd"
