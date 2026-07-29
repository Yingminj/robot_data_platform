#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

require_apply "${1:-}"
require_root
load_site_config
require_ubuntu

log "installing management-node prerequisites"
apt-get update
apt_install \
  ca-certificates chrony curl hdf5-tools jq munge nfs-common openssl \
  openssh-server python3 python3-venv rsync slurm-wlm

command -v nvidia-smi >/dev/null || die "management/GPU node requires an installed NVIDIA driver"
nvidia-smi >/dev/null || die "nvidia-smi cannot communicate with the management-node GPU"

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
ensure_system_user "${INGEST_USER}" "${DATA_GROUP}"
ensure_system_user "${TRAIN_USER}" "${DATA_GROUP}" "${TRAIN_UID}"
usermod -aG docker "${TRAIN_USER}"

if [[ -n "${ADMIN_USER:-}" ]] && getent passwd "${ADMIN_USER}" >/dev/null; then
  usermod -aG docker,"${DATA_GROUP}" "${ADMIN_USER}"
else
  warn "ADMIN_USER=${ADMIN_USER:-unset} does not exist; Docker/data groups were not assigned"
fi

safe_install_dir "${PLATFORM_STATE_ROOT}" root "${DATA_GROUP}" 0750
for subdir in postgres redis app slurm-controller logs; do
  safe_install_dir "${PLATFORM_STATE_ROOT}/${subdir}" root "${DATA_GROUP}" 0750
done
safe_install_dir "${DATASET_CACHE_ROOT}" "${TRAIN_USER}" "${DATA_GROUP}" 0750
safe_install_dir "${EXPORT_CACHE_ROOT}" "${TRAIN_USER}" "${DATA_GROUP}" 0750
safe_install_dir "${RUN_WORK_ROOT}" "${TRAIN_USER}" "${DATA_GROUP}" 0750
safe_install_dir "${PLATFORM_STATE_ROOT}/slurmd" slurm slurm 0750

"${SCRIPT_DIR}/50-configure-nfs-mount.sh" rw --apply

if [[ ! -d "${PLATFORM_ROOT}" ]]; then
  warn "${PLATFORM_ROOT} does not exist. Create the shared QNAP directory before starting services."
fi

install -d -o munge -g munge -m 0700 /etc/munge
if [[ ! -s /etc/munge/munge.key ]]; then
  umask 077
  dd if=/dev/urandom of=/etc/munge/munge.key bs=1024 count=1 status=none
  chown munge:munge /etc/munge/munge.key
  chmod 0400 /etc/munge/munge.key
fi
systemctl enable --now munge

# slurmctld is enabled only after config/nodes.conf is populated and the
# rendered slurm.conf has been reviewed and installed. Preserve an existing
# configured controller during idempotent re-runs.
if [[ ! -r /etc/slurm/slurm.conf && ! -r /etc/slurm-llnl/slurm.conf ]]; then
  systemctl disable --now slurmctld >/dev/null 2>&1 || true
fi
systemctl enable --now chrony
systemctl enable --now ssh

if [[ -n "${ADMIN_USER:-}" ]]; then
  warn "${ADMIN_USER} must log out and back in before new group memberships take effect"
fi
log "management prerequisites installed"
log "next: create the shared QNAP directories, then run deploy/management/bootstrap.sh --apply"
log "next: collect 'slurmd -C' from every GPU node and render Slurm configuration"
