#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
load_site_config

role="${1:-all}"
case "${role}" in
  management|gpu|collector|combined|all) ;;
  *) die "usage: $0 [management|gpu|collector|combined|all]" ;;
esac

check_cmd() {
  local command_name="$1"
  if command -v "${command_name}" >/dev/null 2>&1; then
    printf 'OK       command %-20s %s\n' "${command_name}" "$(command -v "${command_name}")"
  else
    printf 'MISSING  command %s\n' "${command_name}"
  fi
}

check_service() {
  local service_name="$1"
  if systemctl is-active --quiet "${service_name}" 2>/dev/null; then
    printf 'OK       service %s active\n' "${service_name}"
  else
    printf 'INACTIVE service %s\n' "${service_name}"
  fi
}

printf '=== identity ===\n'
hostname
date --iso-8601=seconds
uname -a
sed -n '1,12p' /etc/os-release

printf '\n=== resources ===\n'
lscpu | sed -n '1,28p'
free -h
lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS,MODEL
df -hT || true

printf '\n=== network ===\n'
ip -br address 2>/dev/null || true
ip route 2>/dev/null || true
for host in "${NAS_IP}" "${MANAGEMENT_IP}" ${GPU_NODE_IPS:-}; do
  ping -c 1 -W 1 "${host}" >/dev/null 2>&1 \
    && printf 'OK       ping %s\n' "${host}" \
    || printf 'FAILED   ping %s\n' "${host}"
done

printf '\n=== common services ===\n'
for command_name in ssh rsync curl jq mount.nfs; do check_cmd "${command_name}"; done
for service_name in ssh systemd-timesyncd chrony; do check_service "${service_name}"; done

if [[ "${role}" == management || "${role}" == all ]]; then
  printf '\n=== management role ===\n'
  for command_name in docker psql mlflow slurmctld munge; do check_cmd "${command_name}"; done
  for service_name in docker munge slurmctld; do check_service "${service_name}"; done
  findmnt "${NAS_MOUNT}" || true
fi

if [[ "${role}" == gpu || "${role}" == combined || "${role}" == all ]]; then
  printf '\n=== gpu role ===\n'
  for command_name in nvidia-smi docker nvidia-container-cli slurmd munge; do check_cmd "${command_name}"; done
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
  slurmd -C 2>/dev/null || true
  for service_name in docker munge slurmd; do check_service "${service_name}"; done
  findmnt "${NAS_MOUNT}" || true
fi

if [[ "${role}" == collector || "${role}" == combined || "${role}" == all ]]; then
  printf '\n=== collector role ===\n'
  for command_name in python3 h5dump sha256sum curl jq; do check_cmd "${command_name}"; done
  [[ -d "${COLLECTOR_SPOOL_ROOT}" ]] \
    && printf 'OK       spool %s\n' "${COLLECTOR_SPOOL_ROOT}" \
    || printf 'MISSING  spool %s\n' "${COLLECTOR_SPOOL_ROOT}"
fi
