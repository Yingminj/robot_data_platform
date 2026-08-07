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
    ) -> None:
        self._metrics = metrics
        self._log_file_path = log_file_path
        self._requested_node = requested_node
        self._reserved_nodes = reserved_nodes or set()
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
        script_path = job_dir / "job.sbatch"
        script_path.write_text(self._batch_script(command))
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
    def _batch_script(command: list[str]) -> str:
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
        """

        command_line = shlex.join(command)
        return (
            "#!/usr/bin/env bash\n"
            "set -Eeuo pipefail\n"
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
