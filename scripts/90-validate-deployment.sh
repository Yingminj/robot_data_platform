#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
load_site_config

role="${1:-}"
case "${role}" in
  management|gpu|collector|combined) ;;
  *) die "usage: $0 <management|gpu|collector|combined>" ;;
esac

failures=0
check() {
  local description="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    printf 'PASS  %s\n' "${description}"
  else
    printf 'FAIL  %s\n' "${description}"
    failures=$((failures + 1))
  fi
}

check "time is synchronized" bash -c "[[ \"\$(timedatectl show --property=NTPSynchronized --value)\" == yes ]]"
check "management host responds" ping -c 1 -W 1 "${MANAGEMENT_IP}"
check "NAS responds" ping -c 1 -W 1 "${NAS_IP}"

if [[ "${role}" == management ]]; then
  check "NFS mount exists" mountpoint "${NAS_MOUNT}"
  check "NFS mount is writable" test -w "${PLATFORM_ROOT}"
  check "Docker is active" systemctl is-active docker
  check "Munge is active" systemctl is-active munge
  check "Slurm controller is active" systemctl is-active slurmctld
  check "MLflow health endpoint" curl -fsS --max-time 5 "http://${MANAGEMENT_IP}:5000/health"
  check "Slurm reports nodes" sinfo --noheader
fi

if [[ "${role}" == gpu || "${role}" == combined ]]; then
  check "NFS mount exists" mountpoint "${NAS_MOUNT}"
  check "NFS mount is read-only" bash -c "findmnt -rn -o OPTIONS '${NAS_MOUNT}' | tr ',' '\n' | grep -qx ro"
  check "NVIDIA driver works" nvidia-smi
  check "Docker is active" systemctl is-active docker
  check "NVIDIA runtime exists" command -v nvidia-container-cli
  check "Munge is active" systemctl is-active munge
  check "Slurm worker is active" systemctl is-active slurmd
  check "dataset cache exists" test -d "${DATASET_CACHE_ROOT}"
  check "run work directory exists" test -d "${RUN_WORK_ROOT}"
fi

if [[ "${role}" == collector || "${role}" == combined ]]; then
  check "collector spool exists" test -d "${COLLECTOR_SPOOL_ROOT}/ready-to-upload"
  check "collector service account exists" getent passwd "${COLLECTOR_USER}"
  check "HDF5 tooling exists" command -v h5dump
  check "platform API health endpoint" curl -kfsS --max-time 5 "${PLATFORM_API_URL%/}/health"
fi

if (( failures > 0 )); then
  printf '\n%d validation check(s) failed.\n' "${failures}" >&2
  exit 1
fi
printf '\nAll checks for role %s passed.\n' "${role}"
