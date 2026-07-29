#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

require_apply "${1:-}"
require_root
load_site_config
require_ubuntu

source_dir="${REPO_ROOT}/apps/lelab"
[[ -r "${source_dir}/pyproject.toml" ]] || die "missing ${source_dir}; clone the Yingminj/leLab fork first"

python_bin="${LELAB_PYTHON_BIN:-python3.12}"
command -v "${python_bin}" >/dev/null || die "leLab requires Python 3.12; install it and set LELAB_PYTHON_BIN"
command -v node >/dev/null || die "Node.js 20.19 or newer is required to build the leLab frontend"
command -v npm >/dev/null || die "npm is required to build the leLab frontend"
node -e '
  const [major, minor] = process.versions.node.split(".").map(Number);
  process.exit(major > 20 || (major === 20 && minor >= 19) ? 0 : 1);
' || die "Node.js 20.19 or newer is required; found $(node --version)"

for command_name in sbatch squeue scontrol sinfo ssh nvidia-smi; do
  command -v "${command_name}" >/dev/null || die "missing required command: ${command_name}"
done

install -d -o root -g root -m 0755 /opt/robot-platform
install -d -o root -g root -m 0755 /opt/robot-platform/lelab
rsync -a --delete --exclude='.git/' "${source_dir}/" /opt/robot-platform/lelab/

"${python_bin}" -m venv /opt/robot-platform/lelab-venv
/opt/robot-platform/lelab-venv/bin/pip install --upgrade pip
/opt/robot-platform/lelab-venv/bin/pip install -e /opt/robot-platform/lelab

(
  cd /opt/robot-platform/lelab/frontend
  npm ci
  npm run build
)

install -d -o root -g "${DATA_GROUP}" -m 0750 /etc/robot-platform
if [[ ! -e /etc/robot-platform/lelab.env ]]; then
  install -o root -g "${DATA_GROUP}" -m 0640 \
    "${REPO_ROOT}/config/lelab.env.example" \
    /etc/robot-platform/lelab.env
fi
if [[ ! -e /etc/robot-platform/model-templates.json ]]; then
  install -o root -g "${DATA_GROUP}" -m 0644 \
    "${source_dir}/config/model-templates.json.example" \
    /etc/robot-platform/model-templates.json
fi
install -o root -g root -m 0644 \
  "${REPO_ROOT}/deploy/systemd/lelab-platform.service.example" \
  /etc/systemd/system/lelab-platform.service

for shared_dir in datasets jobs; do
  [[ -d "${PLATFORM_ROOT}/${shared_dir}" ]] || die "create ${PLATFORM_ROOT}/${shared_dir} on the NAS"
done
runuser -u "${TRAIN_USER}" -- test -r "${PLATFORM_ROOT}/datasets" \
  || die "${TRAIN_USER} cannot read ${PLATFORM_ROOT}/datasets"
runuser -u "${TRAIN_USER}" -- test -w "${PLATFORM_ROOT}/jobs" \
  || die "${TRAIN_USER} cannot write ${PLATFORM_ROOT}/jobs"

chown -R "${TRAIN_USER}:${DATA_GROUP}" /opt/robot-platform/lelab-venv
safe_install_dir "${PLATFORM_STATE_ROOT}/huggingface" "${TRAIN_USER}" "${DATA_GROUP}" 0750

systemctl daemon-reload
systemctl enable --now lelab-platform
systemctl --no-pager --full status lelab-platform | sed -n '1,28p'

log "leLab cluster platform installed at http://${MANAGEMENT_IP}:8000"
warn "configure /etc/robot-platform/lelab_ssh_key for passwordless GPU probes on the four remote nodes"
