#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

mode="${1:-}"
apply="${2:-}"
[[ "${mode}" == "rw" || "${mode}" == "ro" ]] || die "usage: $0 <rw|ro> --apply"
require_apply "${apply}"
require_root
load_site_config
require_ubuntu

[[ "${NAS_MOUNT}" == /mnt/* ]] || die "NAS_MOUNT must be below /mnt"
[[ "${NAS_MOUNT}" != "/mnt" ]] || die "NAS_MOUNT cannot be /mnt"

apt-get update
apt_install nfs-common
install -d -o root -g root -m 0755 "${NAS_MOUNT}"

source_spec="${NAS_IP}:${NAS_EXPORT}"
if findmnt -rn "${NAS_MOUNT}" >/dev/null 2>&1; then
  mounted_source="$(findmnt -rn -o SOURCE "${NAS_MOUNT}")"
  [[ "${mounted_source}" == "${source_spec}" ]] || die "${NAS_MOUNT} is already mounted from ${mounted_source}"
  log "${NAS_MOUNT} is already mounted from ${source_spec}"
fi

options="vers=${NFS_VERSION:-4.0},${mode},hard,_netdev,nofail,x-systemd.automount,timeo=600,retrans=2,rsize=1048576,wsize=1048576"
entry="${source_spec} ${NAS_MOUNT} nfs4 ${options} 0 0"
marker="# robot-platform-managed-nfs"

if grep -Fq "${marker}" /etc/fstab; then
  current="$(awk -v marker="${marker}" '$0 == marker {getline; print; exit}' /etc/fstab)"
  [[ "${current}" == "${entry}" ]] || die "managed /etc/fstab entry differs; edit it manually after reviewing the current entry"
else
  cp -a /etc/fstab "/etc/fstab.robot-platform.$(date +%Y%m%d%H%M%S).bak"
  printf '\n%s\n%s\n' "${marker}" "${entry}" >> /etc/fstab
fi

systemctl daemon-reload
mount "${NAS_MOUNT}" 2>/dev/null || true
findmnt "${NAS_MOUNT}" || die "NFS mount did not become available"
test -r "${NAS_MOUNT}" || die "NFS mount is not readable"

if [[ "${mode}" == "rw" ]]; then
  test -w "${NAS_MOUNT}" || warn "mount is rw but current root mapping is not writable; review QNAP host permissions/root squash"
else
  findmnt -rn -o OPTIONS "${NAS_MOUNT}" | tr ',' '\n' | grep -qx ro || die "expected a read-only mount"
fi

log "configured ${source_spec} at ${NAS_MOUNT} (${mode})"
