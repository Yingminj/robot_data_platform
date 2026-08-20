#!/usr/bin/env bash
# Roll the current lerobot_dev code out to every GPU node:
#   local repo + submodule -> latest, then per node:
#   git pull, submodule to origin/${BRANCH}, sudo 25-install-training-environment.sh --sync-lerobot --apply
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_PATH="${REMOTE_PATH:-repo/robot_data_platform}"   # relative to the account's home
BRANCH="${LEROBOT_BRANCH:-dev}"                          # branch of Yingminj/lerobot_dev to track
NODES_DEFAULT=(
  kewei@192.168.100.209
  kewei@192.168.100.206
  yang@192.168.100.216
  snorlax@192.168.100.217
  snorlax@192.168.100.215
)

skip_local=0
dry_run=0
nodes=()
for arg in "$@"; do
  case "${arg}" in
    --skip-local) skip_local=1 ;;
    --dry-run)    dry_run=1 ;;
    -h|--help)    sed -n '2,5p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*)           echo "unknown argument: ${arg}" >&2; exit 2 ;;
    *)            nodes+=("${arg}") ;;
  esac
done
[[ ${#nodes[@]} -gt 0 ]] || nodes=("${NODES_DEFAULT[@]}")

say() { printf '\n== %s\n' "$*"; }

# ---- 1. local repo up to date -------------------------------------------
if [[ "${skip_local}" -eq 1 ]]; then
  say "local update skipped (--skip-local)"
else
  say "updating local ${REPO_ROOT}"
  git -C "${REPO_ROOT}" pull --ff-only
  git -C "${REPO_ROOT}" submodule update --init lerobot
  git -C "${REPO_ROOT}/lerobot" fetch origin "${BRANCH}"
  git -C "${REPO_ROOT}/lerobot" checkout -B "${BRANCH}" "origin/${BRANCH}"
fi
printf 'local lerobot: %s\n' "$(git -C "${REPO_ROOT}/lerobot" log --oneline -1)"

# ---- 2. per node ---------------------------------------------------------
failed=()
for node in "${nodes[@]}"; do
  say "${node}"
  if [[ "${dry_run}" -eq 1 ]]; then
    pw="<prompted>"
  else
    read -rsp "sudo password for ${node}: " pw </dev/tty; echo
  fi

  # The password travels inside the script piped over ssh, so it never lands in
  # the remote argv/env. git reads from /dev/null so it cannot eat the pipe.
  remote=$(cat <<EOF
set -Eeuo pipefail
cd ~/${REMOTE_PATH}
git pull --ff-only </dev/null
git submodule update --init lerobot </dev/null
git -C lerobot fetch origin ${BRANCH} </dev/null
git -C lerobot checkout -B ${BRANCH} origin/${BRANCH} </dev/null
git -C lerobot log --oneline -1
printf '%s\n' '${pw}' | sudo -S -p '' ./scripts/25-install-training-environment.sh --sync-lerobot --apply
EOF
)
  if [[ "${dry_run}" -eq 1 ]]; then
    printf '%s\n' "${remote}"
    continue
  fi
  if ssh -T -o ConnectTimeout=10 "${node}" bash -s <<<"${remote}"; then
    echo "${node}: ok"
  else
    echo "${node}: FAILED" >&2
    failed+=("${node}")
  fi
  unset pw
done

say "done"
if [[ ${#failed[@]} -gt 0 ]]; then
  printf 'failed: %s\n' "${failed[*]}" >&2
  exit 1
fi
