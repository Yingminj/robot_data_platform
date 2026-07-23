#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${REPO_ROOT}/scripts/lib/common.sh"
load_site_config

nodes_file="${1:-${REPO_ROOT}/config/slurm/nodes.conf}"
output_file="${2:-${REPO_ROOT}/config/slurm/slurm.conf.generated}"
template_file="${REPO_ROOT}/config/slurm/slurm.conf.template"

[[ -r "${nodes_file}" ]] || die "missing ${nodes_file}; copy nodes.conf.example and insert actual slurmd -C values"
grep -q '^NodeName=' "${nodes_file}" || die "${nodes_file} has no NodeName lines"
! grep -q 'FILL_ME' "${nodes_file}" || die "${nodes_file} still contains FILL_ME placeholders"

for node in ${GPU_NODE_NAMES}; do
  grep -Eq "^NodeName=${node}([[:space:]]|$)" "${nodes_file}" || die "missing ${node} in ${nodes_file}"
done

node_count="$(grep -c '^NodeName=' "${nodes_file}")"
expected_count="$(wc -w <<<"${GPU_NODE_NAMES}")"
[[ "${node_count}" -eq "${expected_count}" ]] || die "expected ${expected_count} nodes, found ${node_count}"

first_node="$(awk '{print $1}' <<<"${GPU_NODE_NAMES}")"
last_node="$(awk '{print $NF}' <<<"${GPU_NODE_NAMES}")"
if [[ "${first_node}" == gpu01 && "${last_node}" == gpu04 ]]; then
  node_range='gpu[01-04]'
else
  node_range="$(tr ' ' ',' <<<"${GPU_NODE_NAMES}")"
fi

tmp_file="$(mktemp "${output_file}.XXXXXX")"
awk \
  -v cluster="${SLURM_CLUSTER_NAME}" \
  -v controller_host="${MANAGEMENT_HOST}" \
  -v controller_ip="${MANAGEMENT_IP}" \
  -v nodes_file="${nodes_file}" \
  -v node_range="${node_range}" '
    $0 == "@@NODE_LINES@@" {
      while ((getline line < nodes_file) > 0) {
        if (line ~ /^NodeName=/) print line
      }
      close(nodes_file)
      next
    }
    {
      gsub(/@@CLUSTER_NAME@@/, cluster)
      gsub(/@@MANAGEMENT_HOST@@/, controller_host)
      gsub(/@@MANAGEMENT_IP@@/, controller_ip)
      gsub(/@@GPU_NODE_RANGE@@/, node_range)
      print
    }
  ' "${template_file}" > "${tmp_file}"
mv "${tmp_file}" "${output_file}"
chmod 0644 "${output_file}"

log "rendered ${output_file}"
log "review it before installing on the controller and workers"

