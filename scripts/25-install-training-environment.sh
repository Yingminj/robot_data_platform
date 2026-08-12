#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
  cat <<'EOF'
Usage:
  25-install-training-environment.sh --apply
      Create (or repair) the shared training venv and install LeRobot from
      ${LEROBOT_GIT_URL}@${LEROBOT_GIT_REF} with the ${LEROBOT_EXTRAS} extras.

  25-install-training-environment.sh --sync-lerobot --apply
      Overwrite only the installed LeRobot package with the source from this
      repository's lerobot/ submodule. Use after `git pull && git submodule
      update` to roll a code change out to a node without a reinstall.

      This copies source files; it does NOT resolve dependencies. When a policy
      you rely on has gained a third-party dependency, the sync reports it as
      missing and names the pip command to run. Per-policy dependencies are
      documented in lerobot/src/lerobot/policies/<name>/README.md.
EOF
}

mode="install"
apply=0
for arg in "$@"; do
  case "${arg}" in
    --apply) apply=1 ;;
    --sync-lerobot) mode="sync" ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: ${arg}" ;;
  esac
done
if [[ "${apply}" -ne 1 ]]; then
  usage
  exit 2
fi

require_root
load_site_config

# Fails the sync loudly when a policy in the freshly copied source needs a
# distribution the venv does not have. Resolving the extras out of the
# submodule's pyproject.toml keeps this generic: a new policy extra is checked
# automatically, with nothing to update here.
verify_lerobot_dependencies() {
  local pyproject="$1"
  local extras="$2"
  # Runs as root, unlike the import check below: this reads pyproject.toml from
  # the clone, which lives in an operator's home directory that ${TRAIN_USER}
  # has no path into. Only the venv's own metadata is inspected, nothing is
  # written, so root buys reach without side effects.
  "${TRAIN_ENV_ROOT}/bin/python" - "${pyproject}" "${extras}" <<'PY'
import sys, tomllib
import importlib.metadata as md

pyproject, extras = sys.argv[1], sys.argv[2]
try:
    from packaging.requirements import Requirement
except ImportError:
    print("packaging is unavailable; skipping dependency verification")
    raise SystemExit(0)

optional = tomllib.load(open(pyproject, "rb"))["project"]["optional-dependencies"]

# lerobot's extras reference each other (`lerobot[dataset]`), so walk them.
seen_extras, wanted = set(), []
pending = [e.strip() for e in extras.split(",") if e.strip()]
while pending:
    extra = pending.pop()
    if extra in seen_extras or extra not in optional:
        continue
    seen_extras.add(extra)
    for spec in optional[extra]:
        req = Requirement(spec)
        if req.name == "lerobot":
            pending.extend(req.extras)
        elif req.marker is None or req.marker.evaluate():
            wanted.append(req)

missing = []
for req in wanted:
    try:
        installed = md.version(req.name)
    except md.PackageNotFoundError:
        missing.append((req.name, str(req.specifier) or "any", "not installed"))
        continue
    if req.specifier and not req.specifier.contains(installed, prereleases=True):
        missing.append((req.name, str(req.specifier), f"have {installed}"))

if missing:
    print("MISSING")
    for name, spec, detail in missing:
        print(f"{name}{spec}\t({detail})")
PY
}

sync_lerobot_from_submodule() {
  local src_root="${REPO_ROOT}/lerobot"
  local src="${src_root}/src/lerobot"
  [[ -d "${src}" ]] || die "no LeRobot source at ${src}; run: git submodule update --init --recursive"
  [[ -x "${TRAIN_ENV_ROOT}/bin/python" ]] || die "no venv at ${TRAIN_ENV_ROOT}; run this script without --sync-lerobot first"
  command -v rsync >/dev/null || die "rsync is required; install it with: apt-get install -y rsync"

  local site_packages dest
  site_packages="$("${TRAIN_ENV_ROOT}/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  dest="${site_packages}/lerobot"
  [[ -d "${dest}" ]] || die "no lerobot package under ${site_packages}; run this script without --sync-lerobot first"

  local ref
  ref="$(git -C "${src_root}" describe --always --dirty 2>/dev/null || echo unknown)"
  log "syncing ${src} -> ${dest} (submodule at ${ref})"

  # --delete matters: without it a module deleted upstream keeps shadowing the
  # new code and the node quietly runs a mix of two revisions.
  rsync -a --delete --exclude '__pycache__/' --exclude '*.pyc' "${src}/" "${dest}/"

  # pip's metadata still describes whatever it last installed, so record what
  # the files on disk actually came from.
  printf '%s\n' "${ref}" > "${dest}/SUBMODULE_REVISION"
  chown -R "${TRAIN_USER}:${DATA_GROUP}" "${dest}"

  runuser -u "${TRAIN_USER}" -- "${TRAIN_ENV_ROOT}/bin/python" -c "import lerobot" \
    || die "the synced source does not import; the node is now broken, reinstall with --apply"

  local report
  # The sync itself is already done at this point, so a verifier that cannot run
  # must say exactly that instead of dying on a raw traceback that reads like
  # the sync failed.
  if ! report="$(verify_lerobot_dependencies "${src_root}/pyproject.toml" "${LEROBOT_EXTRAS}" 2>&1)"; then
    warn "source synced from ${ref}, but dependency verification could not run:"
    printf '%s\n' "${report}" | sed 's/^/  /' >&2
    die "verify the venv by hand before training on this node"
  fi
  if [[ "${report}" == MISSING* ]]; then
    warn "the synced source declares dependencies this venv does not satisfy:"
    printf '%s\n' "${report}" | tail -n +2 | sed 's/^/  /' >&2
    warn "install them with:"
    warn "  sudo ${TRAIN_ENV_ROOT}/bin/pip install $(printf '%s\n' "${report}" | tail -n +2 | cut -f1 | sed "s/.*/'&'/" | tr '\n' ' ')"
    warn "then re-run this command to confirm"
    die "LeRobot source synced but its dependencies are incomplete"
  fi
  # Non-empty and not MISSING means the check itself was skipped; say so rather
  # than let a silent pass look like a verified one.
  if [[ -n "${report}" ]]; then
    warn "${report}"
  fi

  log "LeRobot source synced from submodule ${ref} into ${TRAIN_ENV_ROOT}"
}

if [[ "${mode}" == "sync" ]]; then
  sync_lerobot_from_submodule
  exit 0
fi

require_ubuntu

python_bin="${TRAIN_PYTHON_BIN:-python3}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
lerobot_git_url="${LEROBOT_GIT_URL:-https://github.com/huggingface/lerobot.git}"
command -v "${python_bin}" >/dev/null || die "missing ${python_bin}"
command -v nvidia-smi >/dev/null || die "missing NVIDIA driver"
nvidia-smi >/dev/null || die "nvidia-smi cannot communicate with the GPU"

apt-get update
# evdev and other C extensions need the interpreter's dev headers (Python.h);
# the venv module package is version-specific too. Both are skipped gracefully
# when the interpreter does not come from APT (e.g. conda).
py_version="$("${python_bin}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
# ffmpeg carries the libav* shared libraries torchcodec dlopens to decode the
# dataset videos. Nothing pulls it in as a dependency: torchcodec ships its own
# libtorchcodec_core*.so and only fails at the first training batch when the
# matching libav* is absent, so a node missing it looks healthy until then.
# rsync is what --sync-lerobot uses to roll code changes out later.
packages=(git build-essential ffmpeg rsync)
for pkg in "python${py_version}-venv" "python${py_version}-dev"; do
  if apt-cache show "${pkg}" >/dev/null 2>&1; then
    packages+=("${pkg}")
  fi
done
apt_install "${packages[@]}"

install -d -o root -g root -m 0755 /opt/robot-platform
"${python_bin}" -m venv "${TRAIN_ENV_ROOT}"
"${TRAIN_ENV_ROOT}/bin/pip" install --upgrade pip
"${TRAIN_ENV_ROOT}/bin/pip" install \
  "lerobot[${LEROBOT_EXTRAS}] @ git+${lerobot_git_url}@${LEROBOT_GIT_REF}"

# Slurm sets HOME from ${TRAIN_USER}'s passwd entry, but that home exists only
# where an installer created it. Training jobs cache torch hub backbone weights
# and Hugging Face artifacts here instead, so every worker has a writable cache.
safe_install_dir "${PLATFORM_STATE_ROOT}/cache" "${TRAIN_USER}" "${DATA_GROUP}" 0750

chown -R "${TRAIN_USER}:${DATA_GROUP}" "${TRAIN_ENV_ROOT}"
runuser -u "${TRAIN_USER}" -- "${TRAIN_ENV_ROOT}/bin/python" -c \
  "import torch; assert torch.cuda.is_available(), 'PyTorch cannot access CUDA'; import lerobot"
# The default video backend loads its FFmpeg bindings lazily, inside a dataloader
# worker on the first batch. Import it here so a broken node fails the install.
runuser -u "${TRAIN_USER}" -- "${TRAIN_ENV_ROOT}/bin/python" -c \
  "from torchcodec.decoders import VideoDecoder"

log "shared training environment installed at ${TRAIN_ENV_ROOT}"
