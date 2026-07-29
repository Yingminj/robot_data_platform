#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_LIB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_LIB_DIR}/../.." && pwd)"
SITE_CONFIG="${SITE_CONFIG:-${REPO_ROOT}/config/site.env}"

log() {
  printf '[robot-platform] %s\n' "$*"
}

warn() {
  printf '[robot-platform] WARNING: %s\n' "$*" >&2
}

die() {
  printf '[robot-platform] ERROR: %s\n' "$*" >&2
  exit 1
}

usage_apply() {
  printf 'This script changes the host. Re-run it with --apply after reviewing config/site.env.\n'
}

require_apply() {
  [[ "${1:-}" == "--apply" ]] || {
    usage_apply
    exit 2
  }
}

require_root() {
  [[ "$(id -u)" -eq 0 ]] || die "run this installer with sudo"
}

load_site_config() {
  [[ -r "${SITE_CONFIG}" ]] || die "missing ${SITE_CONFIG}; copy config/site.env.example and review it"
  # shellcheck disable=SC1090
  source "${SITE_CONFIG}"
  TRAIN_ENV_ROOT="${TRAIN_ENV_ROOT:-/opt/robot-platform/train-venv}"
  LEROBOT_GIT_REF="${LEROBOT_GIT_REF:-v0.6.0}"

  local required=(
    MANAGEMENT_HOST MANAGEMENT_IP NAS_IP NAS_EXPORT NAS_MOUNT PLATFORM_ROOT
    DATA_GROUP DATA_GID TRAIN_USER TRAIN_UID PLATFORM_STATE_ROOT
  )
  local name
  for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || die "${name} is not set in ${SITE_CONFIG}"
  done
}

require_ubuntu() {
  [[ -r /etc/os-release ]] || die "cannot determine operating system"
  # shellcheck disable=SC1091
  source /etc/os-release
  [[ "${ID:-}" == "ubuntu" ]] || die "this installer currently supports Ubuntu only"
  case "${VERSION_ID:-}" in
    22.04|24.04) ;;
    *) warn "Ubuntu ${VERSION_ID:-unknown} has not been validated by this deployment pack" ;;
  esac
}

validate_numeric_id() {
  local value="$1"
  local label="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${label} must be numeric"
  (( value >= 1000 && value <= 60000 )) || die "${label} must be between 1000 and 60000"
}

ensure_group() {
  local group_name="$1"
  local group_id="${2:-}"
  if [[ -n "${group_id}" ]]; then
    validate_numeric_id "${group_id}" "GID for ${group_name}"
  fi
  if getent group "${group_name}" >/dev/null; then
    if [[ -n "${group_id}" ]]; then
      local existing_gid
      existing_gid="$(getent group "${group_name}" | cut -d: -f3)"
      [[ "${existing_gid}" == "${group_id}" ]] \
        || die "group ${group_name} exists with GID ${existing_gid}, expected ${group_id}"
    fi
    return
  elif [[ -n "${group_id}" ]] && getent group "${group_id}" >/dev/null; then
    die "GID ${group_id} is already used by $(getent group "${group_id}" | cut -d: -f1)"
  elif [[ -n "${group_id}" ]]; then
    groupadd --system --gid "${group_id}" "${group_name}"
  else
    groupadd --system "${group_name}"
  fi
}

ensure_system_user() {
  local user_name="$1"
  local primary_group="$2"
  local user_id="${3:-}"
  if [[ -n "${user_id}" ]]; then
    validate_numeric_id "${user_id}" "UID for ${user_name}"
  fi
  if getent passwd "${user_name}" >/dev/null; then
    if [[ -n "${user_id}" ]]; then
      local existing_uid
      existing_uid="$(id -u "${user_name}")"
      [[ "${existing_uid}" == "${user_id}" ]] \
        || die "user ${user_name} exists with UID ${existing_uid}, expected ${user_id}"
    fi
    id -nG "${user_name}" | tr ' ' '\n' | grep -Fxq "${primary_group}" \
      || usermod -aG "${primary_group}" "${user_name}"
  elif [[ -n "${user_id}" ]] && getent passwd "${user_id}" >/dev/null; then
    die "UID ${user_id} is already used by $(getent passwd "${user_id}" | cut -d: -f1)"
  elif [[ -n "${user_id}" ]]; then
    useradd --system --uid "${user_id}" --gid "${primary_group}" --no-create-home \
      --shell /usr/sbin/nologin "${user_name}"
  else
    useradd --system --gid "${primary_group}" --no-create-home --shell /usr/sbin/nologin "${user_name}"
  fi
}

apt_install() {
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
}

ensure_docker() {
  if command -v docker >/dev/null && docker compose version >/dev/null 2>&1; then
    return
  fi

  [[ "${ALLOW_DOCKER_INSTALL:-0}" == "1" ]] || die "Docker with Compose is required but automatic installation is disabled"
  if ! command -v docker >/dev/null; then
    apt_install docker.io
  fi

  if apt-cache show docker-compose-v2 >/dev/null 2>&1; then
    apt_install docker-compose-v2
  elif apt-cache show docker-compose-plugin >/dev/null 2>&1; then
    apt_install docker-compose-plugin
  else
    die "Docker Compose plugin is unavailable from configured APT repositories"
  fi
  systemctl enable --now docker
}

safe_install_dir() {
  local path="$1"
  local owner="$2"
  local group="$3"
  local mode="$4"
  [[ "${path}" == /* ]] || die "directory path must be absolute: ${path}"
  [[ "${path}" != "/" && "${path}" != "/home" && "${path}" != "/var" ]] || die "refusing broad directory target: ${path}"
  install -d -o "${owner}" -g "${group}" -m "${mode}" "${path}"
}
