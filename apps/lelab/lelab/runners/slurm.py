# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Slurm runner for one-node, one-GPU LeRobot training jobs."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import shlex
import subprocess
import threading
import time
from pathlib import Path
from queue import Empty, Queue

from ..cluster import get_model_template, select_idle_node
from ..jobs import LogLine, TrainingMetrics, extract_wandb_run_url, parse_metrics_into
from ..train import TrainingRequest, build_training_command

logger = logging.getLogger(__name__)

_SLURM_ID_RE = re.compile(r"^(?P<id>\d+)(?:;.*)?$")
_TERMINAL_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "TIMEOUT",
    }
)

# LeRobot caps its torchcodec decoder cache at 100 entries, and every eviction
# leaks host RAM in the DataLoader worker that triggered it. A dataset with more
# video files than the cap therefore grows its workers without bound until the
# job's cgroup limit is hit and the kernel kills a worker mid-step -- the
# traceback then surfaces wherever the main process happened to be, usually
# ``save_checkpoint``. Sizing the cache above the file count removes eviction
# entirely, which measurably stops the growth and speeds decoding up ~20%.
#
# Recording one video file per episode (``video_files_size_in_mb: 1``) is what
# makes this reachable here: it turns a 120-episode dataset from ~37 files into
# 360. ``episodes * cameras`` is the exact upper bound on distinct files, and
# over-estimating is free -- the LRU is a cap, not a preallocation, so nothing
# is held for files the dataset does not have.
_DECODER_CACHE_ENV = "LEROBOT_VIDEO_DECODER_CACHE_SIZE"
_DECODER_MB = 4  # measured ~3.3 MB resident per cached decoder, rounded up
_DECODER_MEMORY_FRACTION = 0.5  # of the job's --mem, left for decoders

# Datasets live on the NAS and are read over NFS for the whole run, so every
# epoch re-fetches the same video files across the network. Setting
# ``LELAB_DATASET_CACHE_ROOT`` to a node-local path stages the dataset there
# once and trains from the copy. Unset -- the default -- keeps reading from the
# NAS, so this is inert until an operator opts in.
_STAGE_ROOT_ENV = "LELAB_DATASET_CACHE_ROOT"
# Deleting the copy when the job ends would re-fetch it for the next sweep run
# and for every resume, and an ``exec``d training process cannot run an EXIT
# trap anyway. The cache is instead swept at job *start*: least-recently-staged
# datasets are evicted until the incoming one fits and the filesystem still has
# this much room left. That reclaims after a SIGKILL (cgroup OOM, scancel past
# KillWait, node failure) without the dying job having to do anything.
_STAGE_MIN_FREE_ENV = "LELAB_DATASET_CACHE_MIN_FREE_GB"
_STAGE_MIN_FREE_GB_DEFAULT = 50
# Written last, so its absence marks a copy that was interrupted; its mtime is
# the LRU key.
_STAGE_STAMP = ".lelab_stage_stamp"
_STAGE_ROOT_PLACEHOLDER = "__LELAB_DATASET_ROOT__"


# Kept as one readable block rather than the concatenated line-by-line style of
# _batch_script: it is 60 lines of bash whose correctness lives in the control
# flow. Every conditional is spelled as an `if` because `set -e` aborts on a
# trailing `cond && action` whose condition is false.
_STAGING_SHELL = """
dataset_root=__SRC_Q__
stage_root="${__STAGE_ROOT_ENV__:-}"
if [ -n "$stage_root" ]; then
  stage_src=__SRC_Q__
  stage_name=__NAME_Q__
  stage_dest="$stage_root/$stage_name"
  stage_lockdir="$stage_root/.locks"
  stage_lock="$stage_lockdir/$stage_name.lock"
  stage_min_free_kb=$(( ${__MIN_FREE_ENV__:-__MIN_FREE_DEFAULT__} * 1024 * 1024 ))

  lelab_space_ok() {
    local avail
    avail="$(df -Pk "$stage_root" 2>/dev/null | awk 'NR==2{print $4}')"
    if [ -z "$avail" ]; then
      return 1
    fi
    [ "$avail" -ge "$(( $1 + stage_min_free_kb ))" ]
  }

  # Oldest stamp first; a directory with no stamp is an interrupted copy, so it
  # sorts first and is reclaimed before anything usable.
  lelab_cache_entries() {
    local dir
    find "$stage_root" -mindepth 1 -maxdepth 1 -type d ! -name '.*' 2>/dev/null |
      while IFS= read -r dir; do
        if [ -f "$dir/__STAMP__" ]; then
          printf '%s %s\\n' "$(stat -c %Y "$dir/__STAMP__" 2>/dev/null || echo 0)" "$dir"
        else
          printf '0 %s\\n' "$dir"
        fi
      done | sort -n -k1,1
  }

  lelab_evict_until_free() {
    local need_kb="$1" victim
    lelab_cache_entries | while IFS=' ' read -r _ victim; do
      if lelab_space_ok "$need_kb"; then
        break
      fi
      if [ "$victim" != "$stage_dest" ]; then
        # The exclusive lock is only grantable once no running job holds its
        # shared lock, so a dataset in use is skipped instead of being deleted
        # out from under the job reading it.
        if flock -n -x "$stage_lockdir/$(basename "$victim").lock" \\
             rm -rf -- "$victim" 2>/dev/null; then
          echo "Reclaimed cached dataset $victim." >&2
        fi
      fi
    done
    lelab_space_ok "$need_kb"
  }

  lelab_stage_dataset() {
    local need_kb=0
    if [ -z "$stage_name" ] || [ "$stage_dest" = "$stage_root" ]; then
      return 1
    fi
    if ! mkdir -p "$stage_lockdir"; then
      return 1
    fi
    # fd 9 survives the exec below, so the shared lock is held for as long as
    # the training process lives and is released by the kernel however it dies.
    exec 9>"$stage_lock" || return 1
    flock -x 9 || return 1
    if [ ! -f "$stage_dest/__STAMP__" ]; then
      need_kb="$(du -sk "$stage_src" 2>/dev/null | cut -f1)" || return 1
      if [ -z "$need_kb" ]; then
        return 1
      fi
    fi
    if ! lelab_space_ok "$need_kb"; then
      if ! lelab_evict_until_free "$need_kb"; then
        echo "Not enough room under $stage_root for $stage_src; training from the NAS." >&2
        return 1
      fi
    fi
    if ! mkdir -p "$stage_dest"; then
      return 1
    fi
    # Dropping the stamp first means an interrupted rsync leaves a directory
    # that is never mistaken for a complete copy.
    rm -f "$stage_dest/__STAMP__"
    if ! rsync -a --delete "$stage_src/" "$stage_dest/"; then
      return 1
    fi
    date +%s > "$stage_dest/__STAMP__" || return 1
    # Downgrade so other jobs can read the same copy concurrently.
    flock -s 9 || return 1
    # Converting the lock is not atomic, so another job's sweep can slip in and
    # reclaim the copy in that window. Training from the NAS is always correct,
    # so a missing stamp here means fall back rather than open a deleted path.
    if [ ! -f "$stage_dest/__STAMP__" ]; then
      return 1
    fi
    dataset_root="$stage_dest"
    echo "Training from the node-local copy at $stage_dest." >&2
  }

  if ! command -v rsync >/dev/null 2>&1 || ! command -v flock >/dev/null 2>&1; then
    echo "rsync and flock are required to stage datasets; training from the NAS." >&2
  elif ! lelab_stage_dataset; then
    echo "Could not stage $stage_src; training from the NAS." >&2
    dataset_root=__SRC_Q__
  fi
fi
"""


def _stage_dir_name(dataset_root: str) -> str:
    """Return a flat, collision-free cache directory name for a dataset path.

    The basename keeps the cache readable when an operator looks at it; the
    path digest keeps two datasets that share a basename apart.
    """
    resolved = str(Path(dataset_root))
    digest = hashlib.sha256(resolved.encode()).hexdigest()[:8]
    base = Path(resolved).name or "dataset"
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return f"{safe}-{digest}"


def _staging_preamble(dataset_root: str) -> str:
    """Render the staging block for one dataset path."""
    replacements = {
        "__SRC_Q__": shlex.quote(str(Path(dataset_root))),
        "__NAME_Q__": shlex.quote(_stage_dir_name(dataset_root)),
        "__STAGE_ROOT_ENV__": _STAGE_ROOT_ENV,
        "__MIN_FREE_ENV__": _STAGE_MIN_FREE_ENV,
        "__MIN_FREE_DEFAULT__": str(_STAGE_MIN_FREE_GB_DEFAULT),
        "__STAMP__": _STAGE_STAMP,
    }
    shell = _STAGING_SHELL
    for placeholder, value in replacements.items():
        shell = shell.replace(placeholder, value)
    return shell


def _video_decoder_cache_size(
    dataset_root: str | None, num_workers: int, memory_gb: int
) -> int | None:
    """Return a decoder-cache cap that avoids eviction, or ``None`` to keep LeRobot's.

    Returns ``None`` when the dataset is not readable locally (e.g. a Hub repo
    id with no root), since the file count cannot be known before the job runs.
    """
    if not dataset_root:
        return None
    info_path = Path(dataset_root) / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text())
        episodes = int(info["total_episodes"])
        cameras = sum(1 for key in info.get("features", {}) if key.startswith("observation.images."))
    except Exception as exc:
        logger.warning("Could not size the video decoder cache from %s: %s", info_path, exc)
        return None
    if episodes <= 0 or cameras <= 0:
        return None

    needed = episodes * cameras
    # Each worker keeps its own cache, so the resident cost is per worker.
    budget_mb = memory_gb * 1024 * _DECODER_MEMORY_FRACTION
    affordable = int(budget_mb / (max(num_workers, 1) * _DECODER_MB))
    if affordable < needed:
        logger.warning(
            "Dataset %s has %d video files but only %d decoders fit in %d GB across "
            "%d workers; workers will still leak. Raise the template memory_gb or "
            "lower num_workers.",
            dataset_root,
            needed,
            affordable,
            memory_gb,
            num_workers,
        )
        return max(affordable, 1)
    return needed


class SlurmJobRunner:
    """Submit through ``sbatch`` and tail the shared Slurm output file.

    ``LELAB_OUTPUT_ROOT`` must point to a path visible at the same absolute
    location on every worker (normally the NFS ``jobs`` directory). This
    makes logs and checkpoints available to the management Web process and
    allows a later run to resume on another node.
    """

    def __init__(
        self,
        metrics: TrainingMetrics,
        log_file_path: Path,
        requested_node: str | None = None,
        reserved_nodes: set[str] | None = None,
        preferred_node: str | None = None,
    ) -> None:
        self._metrics = metrics
        self._log_file_path = log_file_path
        self._requested_node = requested_node
        self._reserved_nodes = reserved_nodes or set()
        self._preferred_node = preferred_node
        self._slurm_job_id: str | None = None
        self._node_name: str | None = None
        self._slurm_output_path: Path | None = None
        self._log_queue: Queue[LogLine] = Queue()
        self._tail_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._wandb_run_url: str | None = None
        self._last_state: str | None = None
        self._last_exit_code: int | None = None
        self._state_fetched_at = 0.0

    def start(self, job_id: str, config: TrainingRequest, output_dir: str) -> None:
        template = get_model_template(config.policy_type)
        node = select_idle_node(
            template.min_gpu_memory_mb,
            self._requested_node,
            self._reserved_nodes,
            policy_type=template.policy_type,
            python_executable=template.python_executable,
            preferred_node=self._preferred_node,
        )
        self._node_name = node.name

        output_path = Path(output_dir)
        if not output_path.is_absolute():
            raise ValueError("Slurm output directory must be an absolute shared path")
        job_dir = output_path.parent
        job_dir.mkdir(parents=True, exist_ok=True)
        self._slurm_output_path = job_dir / "slurm.out"

        command = build_training_command(
            config,
            output_dir,
            python_executable=template.python_executable,
        )
        decoder_cache_size = _video_decoder_cache_size(
            config.dataset_root, config.num_workers, template.memory_gb
        )
        script_path = job_dir / "job.sbatch"
        script_path.write_text(
            self._batch_script(command, decoder_cache_size, config.dataset_root)
        )
        script_path.chmod(0o750)

        sbatch = [
            "sbatch",
            "--parsable",
            f"--job-name=lelab-{job_id[:48]}",
            f"--partition={template.partition}",
            "--nodes=1",
            "--ntasks=1",
            "--gres=gpu:1",
            f"--cpus-per-task={template.cpus_per_task}",
            f"--mem={template.memory_gb}G",
            f"--nodelist={node.name}",
            # Without this the job inherits the service's cwd, which exists only
            # on the management host; slurmd logs a chdir error and silently
            # falls back to /tmp. The job directory is on the NAS, so it is the
            # same path on every worker.
            f"--chdir={job_dir}",
            f"--output={self._slurm_output_path}",
            f"--error={self._slurm_output_path}",
            "--open-mode=append",
            str(script_path),
        ]
        logger.info("Submitting Slurm job %s to %s", job_id, node.name)
        result = subprocess.run(sbatch, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "sbatch failed")
        match = _SLURM_ID_RE.match(result.stdout.strip())
        if not match:
            raise RuntimeError(f"Could not parse sbatch job id: {result.stdout!r}")
        self._slurm_job_id = match.group("id")
        self._start_tailing()

    def reattach(self, slurm_job_id: str, node_name: str | None, output_dir: str) -> None:
        if not _SLURM_ID_RE.match(slurm_job_id):
            raise ValueError(f"Invalid Slurm job id: {slurm_job_id!r}")
        self._slurm_job_id = slurm_job_id
        self._node_name = node_name
        self._slurm_output_path = Path(output_dir).parent / "slurm.out"
        self._start_tailing()

    def stop(self) -> None:
        if self._slurm_job_id is None:
            self._stop_event.set()
            return
        result = subprocess.run(
            ["scancel", self._slurm_job_id],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.info("scancel %s ignored: %s", self._slurm_job_id, result.stderr.strip())

    def is_running(self) -> bool:
        state, _ = self._refresh_state()
        return state is None or state not in _TERMINAL_STATES

    def returncode(self) -> int | None:
        state, exit_code = self._refresh_state(force=True)
        if state is None or state not in _TERMINAL_STATES:
            return None
        if state == "COMPLETED":
            return 0
        return exit_code if exit_code not in (None, 0) else 1

    def stream_log_lines(self) -> list[LogLine]:
        out: list[LogLine] = []
        try:
            while True:
                out.append(self._log_queue.get_nowait())
        except Empty:
            return out

    def wandb_run_url(self) -> str | None:
        return self._wandb_run_url

    def slurm_job_id(self) -> str | None:
        return self._slurm_job_id

    def node_name(self) -> str | None:
        return self._node_name

    @staticmethod
    def _batch_script(
        command: list[str],
        decoder_cache_size: int | None = None,
        dataset_root: str | None = None,
    ) -> str:
        """Build a fixed script and re-check out-of-band CUDA use on-node.

        Slurm sets ``HOME`` from the job user's passwd entry, but that home
        need not exist on a worker: ``robot-train`` is a ``nologin`` service
        account and only the management host creates its home. Anything that
        caches under ``~`` then dies on the worker — torchvision downloading
        pretrained backbone weights is the first thing to try, wandb staging
        an artifact at the first checkpoint is the second. Pin those caches to
        a directory that does exist on every node, and redirect ``HOME``
        itself when it is unusable so libraries that ignore XDG still work.

        ``sbatch`` defaults to ``--export=ALL``, so the job also inherits the
        service's own ``HF_HOME`` — a management-host path that a freshly added
        worker will not have. ``LELAB_JOB_CACHE_ROOT`` therefore wins over the
        inherited value rather than deferring to it.

        ``decoder_cache_size`` sizes LeRobot's video decoder cache above the
        dataset's file count; see :func:`_video_decoder_cache_size`. It is set
        with ``:-`` so an operator-supplied value still wins.

        ``dataset_root`` enables node-local staging: ``--dataset.root`` is
        rewritten to a shell variable that the staging block points at the
        local copy, or leaves on the NAS when staging is off or fails.
        """

        staging = ""
        if dataset_root and "--dataset.root" in command:
            command = list(command)
            command[command.index("--dataset.root") + 1] = _STAGE_ROOT_PLACEHOLDER
            staging = _staging_preamble(dataset_root)

        # The placeholder is bare word characters, so shlex.join leaves it
        # unquoted and the substitution below yields an expanding "$dataset_root".
        command_line = shlex.join(command).replace(
            _STAGE_ROOT_PLACEHOLDER, '"$dataset_root"'
        )
        env = _DECODER_CACHE_ENV
        decoder_export = (
            ""
            if decoder_cache_size is None
            else (
                f'export {env}="${{{env}:-{decoder_cache_size}}}"\n'
                # A node whose hard limit is too low to hold one handle per video
                # would otherwise die on Errno 24 partway through the first epoch.
                # Half the budget leaves room for everything else the process
                # opens; going back to eviction is slower, not fatal.
                'fd_limit="$(ulimit -Sn)"\n'
                f'if [[ "$fd_limit" =~ ^[0-9]+$ && "${{{env}}}" =~ ^[0-9]+$ ]] '
                f'&& [ "${{{env}}}" -gt "$((fd_limit / 2))" ]; then\n'
                f'  echo "Capping {env} at $((fd_limit / 2)) '
                'to stay under the open-file limit of $fd_limit." >&2\n'
                f'  export {env}="$(( fd_limit / 2 > 0 ? fd_limit / 2 : 1 ))"\n'
                "fi\n"
            )
        )
        return (
            "#!/usr/bin/env bash\n"
            "set -Eeuo pipefail\n"
            "# Every cached video decoder pins an open file handle, and torch's\n"
            "# file_descriptor sharing spends more per in-flight batch. sbatch\n"
            "# propagates the submitting service's soft limit -- systemd's default\n"
            "# 1024 -- which a dataset of ~1000 video files exhausts before the\n"
            "# first step. Raise the soft limit to the node's hard limit.\n"
            'if ! ulimit -n "$(ulimit -Hn)" 2>/dev/null; then\n'
            '  echo "Warning: could not raise the open-file limit above $(ulimit -Sn)." >&2\n'
            "fi\n"
            "# Check for compute processes, excluding known system/desktop processes\n"
            "compute_pids=$(nvidia-smi --query-compute-apps=pid,process_name "
            "--format=csv,noheader,nounits 2>/dev/null || true)\n"
            "if [ -n \"$compute_pids\" ]; then\n"
            "  # Filter out known desktop/system processes that may use CUDA\n"
            "  filtered=$(echo \"$compute_pids\" | grep -Ev "
            "'(Xorg|gnome-shell|rustdesk|awesun|code|chrome|firefox)' || true)\n"
            "  if [ -n \"$filtered\" ]; then\n"
            "    echo 'GPU became busy outside this Slurm allocation; refusing to start.'\n"
            "    echo \"Active compute processes:\"\n"
            "    echo \"$filtered\"\n"
            "    exit 75\n"
            "  fi\n"
            "fi\n"
            "export PYTHONUNBUFFERED=1\n"
            f"{decoder_export}"
            'cache_root="${LELAB_JOB_CACHE_ROOT:-}"\n'
            'if [ -n "$cache_root" ]; then\n'
            "  # An explicit job cache root overrides what sbatch inherited from the\n"
            "  # service environment. HF_HOME there names a management-host-local\n"
            "  # directory that a worker need not have -- and cannot create, since its\n"
            "  # parent is root-owned -- so honouring it would send the job right back\n"
            "  # to the path LELAB_JOB_CACHE_ROOT exists to avoid.\n"
            '  export HF_HOME="$cache_root/huggingface"\n'
            '  export TORCH_HOME="$cache_root/torch"\n'
            '  export XDG_CACHE_HOME="$cache_root/xdg/cache"\n'
            '  export XDG_DATA_HOME="$cache_root/xdg/data"\n'
            '  export XDG_CONFIG_HOME="$cache_root/xdg/config"\n'
            "else\n"
            '  cache_root="${HF_HOME:-}"\n'
            '  if [ -n "$cache_root" ]; then\n'
            '    export TORCH_HOME="${TORCH_HOME:-$cache_root/torch}"\n'
            '    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$cache_root/xdg/cache}"\n'
            '    export XDG_DATA_HOME="${XDG_DATA_HOME:-$cache_root/xdg/data}"\n'
            '    export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$cache_root/xdg/config}"\n'
            "  fi\n"
            "fi\n"
            'if [ -n "$cache_root" ]; then\n'
            '  if [ ! -w "${HOME:-}" ]; then export HOME="$cache_root/home"; fi\n'
            '  if ! mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" '
            '"$XDG_DATA_HOME" "$XDG_CONFIG_HOME" "$HOME"; then\n'
            "    echo \"Cannot create the job cache under '$cache_root' on $(hostname);\" >&2\n"
            '    echo "set LELAB_JOB_CACHE_ROOT to a path writable on every worker." >&2\n'
            "    exit 1\n"
            "  fi\n"
            "fi\n"
            f"{staging}"
            f"exec {command_line}\n"
        )

    def _start_tailing(self) -> None:
        if self._tail_thread is not None:
            return
        self._tail_thread = threading.Thread(
            target=self._tail_loop,
            name=f"slurm-job-{self._slurm_job_id}-logs",
            daemon=True,
        )
        self._tail_thread.start()

    def _tail_loop(self) -> None:
        assert self._slurm_output_path is not None
        offset = 0
        while not self._stop_event.is_set():
            path = self._slurm_output_path
            if path.exists():
                with path.open(errors="replace") as source:
                    source.seek(offset)
                    for raw in source:
                        self._consume_line(raw.rstrip())
                    offset = source.tell()
            if not self.is_running():
                # One final pass catches buffered output written at exit.
                if path.exists():
                    with path.open(errors="replace") as source:
                        source.seek(offset)
                        for raw in source:
                            self._consume_line(raw.rstrip())
                return
            self._stop_event.wait(0.5)

    def _consume_line(self, message: str) -> None:
        if not message:
            return
        parse_metrics_into(message, self._metrics)
        if self._wandb_run_url is None:
            self._wandb_run_url = extract_wandb_run_url(message)
        line = LogLine(timestamp=time.time(), message=message)
        self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_file_path.open("a") as sink:
            sink.write(line.model_dump_json() + "\n")
        if self._log_queue.qsize() >= 1000:
            with contextlib.suppress(Empty):
                self._log_queue.get_nowait()
        self._log_queue.put(line)

    def _refresh_state(self, force: bool = False) -> tuple[str | None, int | None]:
        now = time.time()
        if not force and now - self._state_fetched_at < 1.0:
            return self._last_state, self._last_exit_code
        self._state_fetched_at = now
        if self._slurm_job_id is None:
            return None, None

        active = subprocess.run(
            ["squeue", "-h", "-j", self._slurm_job_id, "-o", "%T"],
            check=False,
            capture_output=True,
            text=True,
        )
        if active.returncode == 0 and active.stdout.strip():
            self._last_state = active.stdout.splitlines()[0].strip().upper()
            self._last_exit_code = None
            return self._last_state, None

        detail = subprocess.run(
            ["scontrol", "show", "job", "-o", self._slurm_job_id],
            check=False,
            capture_output=True,
            text=True,
        )
        if detail.returncode != 0:
            # A controller restart or MinJobAge expiry can remove the record.
            self._last_state = "FAILED"
            self._last_exit_code = 1
            return self._last_state, self._last_exit_code

        values: dict[str, str] = {}
        for token in shlex.split(detail.stdout):
            key, separator, value = token.partition("=")
            if separator:
                values[key] = value
        self._last_state = values.get("JobState", "FAILED").split("+", 1)[0].upper()
        raw_exit = values.get("ExitCode", "1:0").split(":", 1)[0]
        with contextlib.suppress(ValueError):
            self._last_exit_code = int(raw_exit)
        return self._last_state, self._last_exit_code


__all__ = ["SlurmJobRunner"]
