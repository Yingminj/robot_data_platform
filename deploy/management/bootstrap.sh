#!/usr/bin/env bash
set -Eeuo pipefail

DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${DEPLOY_DIR}/../.." && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${REPO_ROOT}/scripts/lib/common.sh"

require_apply "${1:-}"
require_root
load_site_config

command -v docker >/dev/null || die "Docker is missing; run scripts/10-install-management.sh --apply first"
docker compose version >/dev/null 2>&1 || die "Docker Compose plugin is missing"
[[ -d "${PLATFORM_ROOT}/mlflow-artifacts" ]] || die "create ${PLATFORM_ROOT}/mlflow-artifacts on the NAS first"

if ! runuser -u "${INGEST_USER}" -- test -w "${PLATFORM_ROOT}/mlflow-artifacts"; then
  die "${INGEST_USER} cannot write ${PLATFORM_ROOT}/mlflow-artifacts; enable pilot read/write access on the NAS export (with all_squash, grant the QNAP guest account write access to the share and this directory)"
fi

safe_install_dir "${PLATFORM_STATE_ROOT}/postgres" root root 0700
# The official PostgreSQL image currently uses numeric UID/GID 999. Verify this
# again whenever POSTGRES_IMAGE is changed.
chown 999:999 "${PLATFORM_STATE_ROOT}/postgres"

env_file="${DEPLOY_DIR}/.env"
if [[ ! -e "${env_file}" ]]; then
  umask 077
  password="$(openssl rand -hex 32)"
  {
    printf 'MANAGEMENT_IP=%s\n' "${MANAGEMENT_IP}"
    printf 'PLATFORM_STATE_ROOT=%s\n' "${PLATFORM_STATE_ROOT}"
    printf 'MLFLOW_ARTIFACT_ROOT=%s\n' "${PLATFORM_ROOT}/mlflow-artifacts"
    printf 'POSTGRES_USER=robotplatform\n'
    printf 'POSTGRES_PASSWORD=%s\n' "${password}"
    printf 'POSTGRES_DB=platform\n'
    printf 'MLFLOW_DB=mlflow\n'
    printf 'POSTGRES_IMAGE=postgres:16\n'
    printf 'REDIS_IMAGE=redis:7-alpine\n'
    printf 'MLFLOW_VERSION=3.1.1\n'
  } > "${env_file}"
  chmod 0600 "${env_file}"
fi

cd "${DEPLOY_DIR}"
docker compose --env-file "${env_file}" config >/dev/null
docker compose --env-file "${env_file}" build mlflow
docker compose --env-file "${env_file}" up -d
docker compose --env-file "${env_file}" ps

log "management infrastructure started"
log "MLflow: http://${MANAGEMENT_IP}:5000"
warn "the platform API, upload worker, web UI and annotation application are not in this workspace and must be added as separate services"
