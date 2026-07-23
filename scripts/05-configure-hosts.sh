#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

require_apply "${1:-}"
require_root
load_site_config

read -r -a node_names <<<"${GPU_NODE_NAMES}"
read -r -a node_ips <<<"${GPU_NODE_IPS}"
[[ "${#node_names[@]}" -eq "${#node_ips[@]}" ]] || die "GPU_NODE_NAMES and GPU_NODE_IPS have different lengths"

begin_marker='# BEGIN robot-platform managed hosts'
end_marker='# END robot-platform managed hosts'

block="${begin_marker}"$'\n'
block+="${MANAGEMENT_IP} ${MANAGEMENT_HOST}"$'\n'
for index in "${!node_names[@]}"; do
  block+="${node_ips[$index]} ${node_names[$index]}"$'\n'
done
block+="${end_marker}"

if grep -Fq "${begin_marker}" /etc/hosts; then
  current_block="$(sed -n "/^${begin_marker}$/,/^${end_marker}$/p" /etc/hosts)"
  [[ "${current_block}" == "${block}" ]] || die "existing managed /etc/hosts block differs; review it manually"
  log "/etc/hosts already contains the expected managed block"
  exit 0
fi

for host_name in "${MANAGEMENT_HOST}" "${node_names[@]}"; do
  if awk -v name="${host_name}" '$0 !~ /^#/ {for (i=2; i<=NF; i++) if ($i == name) found=1} END {exit !found}' /etc/hosts; then
    die "/etc/hosts already contains ${host_name}; resolve the existing entry before applying the managed block"
  fi
done

cp -a /etc/hosts "/etc/hosts.robot-platform.$(date +%Y%m%d%H%M%S).bak"
printf '\n%s\n' "${block}" >> /etc/hosts

for host_name in "${MANAGEMENT_HOST}" "${node_names[@]}"; do
  getent hosts "${host_name}" >/dev/null || die "host lookup failed after updating /etc/hosts: ${host_name}"
done
log "installed robot-platform host mappings"

