#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${REPO_ROOT}/scripts/lib/common.sh"

config_file="${1:-${REPO_ROOT}/config/slurm/slurm.conf.generated}"
apply="${2:-}"
require_apply "${apply}"
require_root
load_site_config

[[ -r "${config_file}" ]] || die "missing rendered Slurm config: ${config_file}"
! grep -q 'FILL_ME\|@@' "${config_file}" || die "Slurm config contains unresolved placeholders"
command -v slurmctld >/dev/null || die "slurmctld is missing; run scripts/10-install-management.sh --apply"
[[ -s /etc/munge/munge.key ]] || die "/etc/munge/munge.key is missing"

slurm_dir=/etc/slurm
[[ -d /etc/slurm-llnl && ! -d /etc/slurm ]] && slurm_dir=/etc/slurm-llnl
install -d -o root -g root -m 0755 "${slurm_dir}"
install -m 0644 "${config_file}" "${slurm_dir}/slurm.conf"
install -m 0644 "${REPO_ROOT}/config/slurm/cgroup.conf" "${slurm_dir}/cgroup.conf"

safe_install_dir "${PLATFORM_STATE_ROOT}/slurm-controller" slurm slurm 0750
safe_install_dir /var/log/slurm slurm slurm 0750

systemctl enable munge
systemctl restart munge
munge -n | unmunge >/dev/null || die "local Munge self-test failed"
systemctl enable slurmctld
systemctl restart slurmctld
systemctl --no-pager --full status slurmctld | sed -n '1,24p'

log "Slurm controller configuration installed"
