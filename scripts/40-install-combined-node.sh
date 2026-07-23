#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

require_apply "${1:-}"
require_root
load_site_config

warn "combined mode shares CPU, RAM, disk I/O and possibly the GPU between collection and training"
warn "the collector still uploads through the management API and must not receive raw NAS write access"

"${SCRIPT_DIR}/20-install-gpu-node.sh" --apply
"${SCRIPT_DIR}/30-install-collector.sh" --apply

log "combined GPU + collector prerequisites installed"
log "configure collection windows or drain the Slurm node whenever collection requires the GPU or high disk bandwidth"

