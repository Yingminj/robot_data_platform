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
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
command -v "${python_bin}" >/dev/null || die "leLab requires Python 3.12; install it and set LELAB_PYTHON_BIN"

for command_name in git sbatch squeue scontrol sinfo ssh nvidia-smi; do
  command -v "${command_name}" >/dev/null || die "missing required command: ${command_name}"
done

retry_command() {
  local max_attempts="$1"
  local retry_delay="$2"
  shift 2

  local attempt=1
  while true; do
    if "$@"; then
      return 0
    fi
    if (( attempt >= max_attempts )); then
      return 1
    fi
    warn "command failed on attempt ${attempt}/${max_attempts}; retrying in ${retry_delay}s"
    sleep "${retry_delay}"
    ((attempt += 1))
  done
}

build_user="${LELAB_BUILD_USER:-${SUDO_USER:-}}"
[[ -n "${build_user}" ]] \
  || die "cannot determine the unprivileged frontend build user; set LELAB_BUILD_USER"
[[ "${build_user}" != "root" && "$(id -u "${build_user}" 2>/dev/null)" -ne 0 ]] \
  || die "LELAB_BUILD_USER must name an existing non-root user"
build_group="$(id -gn "${build_user}")"
build_home="$(getent passwd "${build_user}" | cut -d: -f6)"
[[ -d "${build_home}" ]] || die "home directory for ${build_user} does not exist: ${build_home}"

build_root=""
git_config=""
cleanup_build_root() {
  if [[ "${build_root}" == /tmp/lelab-build.* && -d "${build_root}" ]]; then
    rm -rf -- "${build_root}"
  fi
  if [[ "${git_config}" == /tmp/lelab-gitconfig.* && -f "${git_config}" ]]; then
    rm -f -- "${git_config}"
  fi
}
trap cleanup_build_root EXIT

build_root="$(mktemp -d /tmp/lelab-build.XXXXXX)"
build_source="${build_root}/lelab"
chown "${build_user}:${build_group}" "${build_root}"
chmod 0750 "${build_root}"
install -d -o "${build_user}" -g "${build_group}" -m 0750 "${build_source}"
rsync -a \
  --exclude='.git/' \
  --exclude='frontend/dist/' \
  --exclude='frontend/node_modules/' \
  --chown="${build_user}:${build_group}" \
  "${source_dir}/" "${build_source}/"

log "building the leLab frontend as unprivileged user ${build_user}"
runuser -u "${build_user}" -- env -i \
  HOME="${build_home}" \
  USER="${build_user}" \
  LOGNAME="${build_user}" \
  PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  /bin/bash -c '
    set -Eeuo pipefail
    if [[ -s "${HOME}/.nvm/nvm.sh" ]]; then
      export NVM_DIR="${HOME}/.nvm"
      # nvm is loaded only after runuser has dropped root privileges.
      source "${NVM_DIR}/nvm.sh"
    fi
    command -v node >/dev/null \
      || { printf "Node.js 20.19 or newer is required to build the leLab frontend\n" >&2; exit 1; }
    command -v npm >/dev/null \
      || { printf "npm is required to build the leLab frontend\n" >&2; exit 1; }
    node -e '"'"'
      const [major, minor] = process.versions.node.split(".").map(Number);
      process.exit(major > 20 || (major === 20 && minor >= 19) ? 0 : 1);
    '"'"' || {
      printf "Node.js 20.19 or newer is required; found %s\n" "$(node --version)" >&2
      exit 1
    }
    printf "[robot-platform] frontend toolchain: Node.js %s, npm %s\n" \
      "$(node --version)" "$(npm --version)"
    cd "$1/frontend"
    npm ci
    npm run build
  ' bash "${build_source}" \
  || die "failed to build the leLab frontend as ${build_user}"
[[ -s "${build_source}/frontend/dist/index.html" ]] \
  || die "frontend build completed without producing dist/index.html"

install -d -o root -g root -m 0755 /opt/robot-platform
install -d -o root -g root -m 0755 /opt/robot-platform/lelab
rsync -a \
  --delete \
  --delete-excluded \
  --exclude='.git/' \
  --exclude='frontend/node_modules/' \
  --chown=root:root \
  "${build_source}/" /opt/robot-platform/lelab/

"${python_bin}" -m venv /opt/robot-platform/lelab-venv
/opt/robot-platform/lelab-venv/bin/pip install --upgrade pip

# LeLab pins LeRobot to GitHub in pyproject.toml. Rewrite only that URL for this
# pip invocation so deployments can use the site-configured mirror without
# changing developer installs or writing persistent root Git configuration.
direct_lerobot_url="https://github.com/huggingface/lerobot.git"
lerobot_git_url="${LEROBOT_GIT_URL:-${direct_lerobot_url}}"
git_config="$(mktemp /tmp/lelab-gitconfig.XXXXXX)"
chmod 0600 "${git_config}"
git config --file "${git_config}" http.version HTTP/1.1
if [[ "${lerobot_git_url}" != "${direct_lerobot_url}" ]]; then
  git config --file "${git_config}" \
    "url.${lerobot_git_url}.insteadOf" "${direct_lerobot_url}"
  log "using configured LeRobot Git mirror: ${lerobot_git_url}"
fi
retry_command 3 5 \
  env GIT_CONFIG_GLOBAL="${git_config}" \
  /opt/robot-platform/lelab-venv/bin/pip install -e /opt/robot-platform/lelab \
  || die "failed to install leLab dependencies after 3 attempts"

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
  || die "${TRAIN_USER} cannot read ${PLATFORM_ROOT}/datasets; with all_squash, grant the QNAP guest account access"
runuser -u "${TRAIN_USER}" -- test -w "${PLATFORM_ROOT}/jobs" \
  || die "${TRAIN_USER} cannot write ${PLATFORM_ROOT}/jobs; with all_squash, grant the QNAP guest account write access"

chown -R "${TRAIN_USER}:${DATA_GROUP}" /opt/robot-platform/lelab-venv
safe_install_dir "${PLATFORM_STATE_ROOT}/huggingface" "${TRAIN_USER}" "${DATA_GROUP}" 0750

systemctl daemon-reload
systemctl enable --now lelab-platform
systemctl --no-pager --full status lelab-platform | sed -n '1,28p'

log "leLab cluster platform installed at http://${MANAGEMENT_IP}:8000"
warn "configure /etc/robot-platform/lelab_ssh_key for passwordless GPU probes on every remote node"
