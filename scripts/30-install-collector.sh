#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

require_apply "${1:-}"
require_root
load_site_config
require_ubuntu

log "installing collector prerequisites"
apt-get update
apt_install \
  ca-certificates chrony curl hdf5-tools inotify-tools jq \
  openssh-server python3 python3-pip python3-venv rsync

ensure_group "${DATA_GROUP}" "${DATA_GID}"
ensure_system_user "${COLLECTOR_USER}" "${COLLECTOR_UID}" "${DATA_GROUP}"

safe_install_dir "${COLLECTOR_SPOOL_ROOT}" "${COLLECTOR_USER}" "${DATA_GROUP}" 0750
for state in recording ready-to-upload uploading uploaded failed; do
  safe_install_dir "${COLLECTOR_SPOOL_ROOT}/${state}" "${COLLECTOR_USER}" "${DATA_GROUP}" 0750
done
safe_install_dir /etc/robot-platform root "${DATA_GROUP}" 0750

collector_env=/etc/robot-platform/collector.env
if [[ ! -e "${collector_env}" ]]; then
  umask 027
  {
    printf 'COLLECTOR_NODE_ID=%s\n' "$(hostname -s)"
    printf 'PLATFORM_API_URL=%s\n' "${PLATFORM_API_URL}"
    printf 'SPOOL_ROOT=%s\n' "${COLLECTOR_SPOOL_ROOT}"
    printf 'MIN_FREE_GB=%s\n' "${COLLECTOR_MIN_FREE_GB}"
    printf 'UPLOAD_TOKEN_FILE=/etc/robot-platform/upload.token\n'
  } > "${collector_env}"
  chown root:"${DATA_GROUP}" "${collector_env}"
  chmod 0640 "${collector_env}"
fi

install -m 0644 "${REPO_ROOT}/deploy/systemd/robot-upload-agent.service.example" \
  /etc/systemd/system/robot-upload-agent.service
systemctl daemon-reload
systemctl enable --now chrony ssh

log "collector OS prerequisites and spool directories installed"
warn "ROS2, the H5 converter/validator and upload-agent application are project-specific and were not present in this workspace"
warn "the upload service remains stopped until /opt/robot-platform/bin/robot-upload-agent is installed and an upload token is provisioned"
