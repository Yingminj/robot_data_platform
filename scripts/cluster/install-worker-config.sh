#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${REPO_ROOT}/scripts/lib/common.sh"

munge_key_file="${1:-}"
slurm_config_file="${2:-}"
apply="${3:-}"
[[ -r "${munge_key_file}" && -r "${slurm_config_file}" ]] \
  || die "usage: $0 <secure-copy-of-munge.key> <slurm.conf.generated> --apply"
require_apply "${apply}"
require_root
load_site_config

command -v slurmd >/dev/null || die "slurmd is missing; run scripts/20-install-gpu-node.sh --apply"
! grep -q 'FILL_ME\|@@' "${slurm_config_file}" || die "Slurm config contains unresolved placeholders"

install -d -o munge -g munge -m 0700 /etc/munge
install -o munge -g munge -m 0400 "${munge_key_file}" /etc/munge/munge.key

slurm_dir=/etc/slurm
[[ -d /etc/slurm-llnl && ! -d /etc/slurm ]] && slurm_dir=/etc/slurm-llnl
install -d -o root -g root -m 0755 "${slurm_dir}"
install -m 0644 "${slurm_config_file}" "${slurm_dir}/slurm.conf"
install -m 0644 "${REPO_ROOT}/config/slurm/cgroup.conf" "${slurm_dir}/cgroup.conf"
install -m 0644 "${REPO_ROOT}/config/slurm/gres.conf" "${slurm_dir}/gres.conf"

safe_install_dir "${PLATFORM_STATE_ROOT}/slurmd" slurm slurm 0750
safe_install_dir /var/log/slurm slurm slurm 0750

systemctl enable munge
systemctl restart munge
munge -n | unmunge >/dev/null || die "local Munge self-test failed"
slurmd -C
systemctl enable slurmd
systemctl restart slurmd
systemctl --no-pager --full status slurmd | sed -n '1,24p'

log "Slurm worker configuration installed"
warn "remove the temporary copy of munge.key from this node after verifying the service"
