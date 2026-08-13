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

"""Small-cluster discovery and registered training-template helpers.

The management host is the only machine that calls this module. Slurm remains
the authority for allocations; the SSH ``nvidia-smi`` probe adds visibility
for CUDA processes started manually outside Slurm.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_DEFAULT_NODES = "mgmt01,gpu01"
_IDLE_SLURM_STATES = frozenset({"idle"})


class ModelTemplate(BaseModel):
    id: str
    label: str
    policy_type: str
    python_executable: str = "/opt/robot-platform/train-venv/bin/python"
    partition: str = "train"
    min_gpu_memory_mb: int = 0
    cpus_per_task: int = 8
    # Nodes have ~60 GB of RealMemory and one GPU each, so at most one training
    # job runs per node. 48 leaves headroom for the OS and, on mgmt01, for the
    # MLflow/Postgres/Redis/LeLab services that sit outside Slurm's accounting.
    memory_gb: int = 48
    description: str = ""


class ClusterNode(BaseModel):
    name: str
    address: str
    slurm_state: str = "unknown"
    reachable: bool = False
    gpu_name: str | None = None
    memory_total_mb: int | None = None
    memory_free_mb: int | None = None
    compute_processes: int = 0
    eligible: bool = False
    reason: str | None = None


class ClusterStatus(BaseModel):
    enabled: bool
    nodes: list[ClusterNode]


def cluster_enabled() -> bool:
    return os.environ.get("LELAB_CLUSTER_ENABLED", "0").lower() in {"1", "true", "yes", "on"}


def _node_specs() -> list[tuple[str, str]]:
    """Return ``(Slurm name, SSH address)`` pairs from LELAB_CLUSTER_NODES.

    Entries accept either ``name`` or ``name=address``. Keeping the mapping in
    one environment variable makes it easy to use static IPs without requiring
    DNS, while Slurm still sees stable NodeName values.
    """

    raw = os.environ.get("LELAB_CLUSTER_NODES", _DEFAULT_NODES)
    out: list[tuple[str, str]] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, separator, address = entry.partition("=")
        out.append((name.strip(), address.strip() if separator else name.strip()))
    return out


def _run(command: list[str], timeout: float = 5.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _slurm_states() -> dict[str, str]:
    try:
        result = _run(["sinfo", "-N", "-h", "-o", "%N|%T"])
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("Could not query Slurm nodes: %s", exc)
        return {}
    if result.returncode != 0:
        logger.warning("sinfo failed: %s", result.stderr.strip())
        return {}
    states: dict[str, str] = {}
    for line in result.stdout.splitlines():
        name, separator, state = line.strip().partition("|")
        if separator:
            states[name] = state.rstrip("*").lower()
    return states


def _is_local_node(name: str, address: str) -> bool:
    local_names = {
        "localhost",
        socket.gethostname(),
        socket.gethostname().split(".", 1)[0],
        socket.getfqdn(),
    }
    return name in local_names or address in local_names


def _probe_command(name: str, address: str, *remote_args: str) -> list[str]:
    if _is_local_node(name, address):
        return list(remote_args)
    timeout = os.environ.get("LELAB_SSH_CONNECT_TIMEOUT", "3")
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={timeout}",
    ]
    identity = os.environ.get("LELAB_SSH_IDENTITY_FILE")
    if identity:
        command.extend(["-i", str(Path(identity).expanduser())])
    command.extend([address, *remote_args])
    return command


def _parse_int(value: str) -> int | None:
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return None


def _probe_node(name: str, address: str, slurm_state: str) -> ClusterNode:
    node = ClusterNode(name=name, address=address, slurm_state=slurm_state)
    gpu_query = _probe_command(
        name,
        address,
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    )
    try:
        gpu = _run(gpu_query)
    except (OSError, subprocess.TimeoutExpired) as exc:
        node.reason = f"GPU probe failed: {exc}"
        return node
    if gpu.returncode != 0 or not gpu.stdout.strip():
        node.reason = gpu.stderr.strip() or "nvidia-smi returned no GPU"
        return node

    first_gpu = gpu.stdout.splitlines()[0]
    fields = [field.strip() for field in first_gpu.split(",")]
    if len(fields) < 3:
        node.reason = "unexpected nvidia-smi output"
        return node
    node.reachable = True
    node.gpu_name = fields[0]
    node.memory_total_mb = _parse_int(fields[1])
    node.memory_free_mb = _parse_int(fields[2])

    # Query GPU processes with type information (C=Compute, G=Graphics)
    # We only care about Compute processes (training jobs), not Graphics (desktop UI)
    process_query = _probe_command(
        name,
        address,
        "nvidia-smi",
        "--query-compute-apps=pid,process_name",
        "--format=csv,noheader,nounits",
    )
    try:
        processes = _run(process_query)
        if processes.returncode == 0:
            # Filter out known system/graphics processes to avoid false positives
            # Common patterns: Xorg, gnome-shell, code, rustdesk, browsers, etc.
            graphics_patterns = {
                "xorg", "x11", "gnome", "kde", "plasma", "compiz",
                "rustdesk", "teamviewer", "anydesk", "parsec", "awesun",
                "code", "chrome", "firefox", "edge", "safari",
                "clash", "v2ray", "sunlogin",
            }
            compute_count = 0
            debug_lines = []
            for line in processes.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Line format: "PID, process_name"
                parts = line.split(",", 1)
                if len(parts) < 2:
                    compute_count += 1  # No name available, count it to be safe
                    debug_lines.append(f"No name, counted")
                    continue
                process_name = parts[1].strip().lower()
                # If process name contains any graphics pattern, skip it
                is_filtered = any(pattern in process_name for pattern in graphics_patterns)
                debug_lines.append(f"{process_name[:40]} -> filtered={is_filtered}")
                if is_filtered:
                    continue
                compute_count += 1
            if debug_lines:
                logger.warning(f"[GPU-CHECK {name}] {len(debug_lines)} processes: {'; '.join(debug_lines)}")
            node.compute_processes = compute_count
        else:
            node.compute_processes = 1
            node.reason = processes.stderr.strip() or "could not inspect GPU processes"
    except (OSError, subprocess.TimeoutExpired) as exc:
        node.compute_processes = 1
        node.reason = f"GPU process probe failed: {exc}"

    if slurm_state not in _IDLE_SLURM_STATES:
        node.reason = f"Slurm state is {slurm_state}"
    elif node.compute_processes:
        node.reason = "GPU has a compute process outside or inside Slurm"
    else:
        node.eligible = True
        node.reason = None
    return node


def list_cluster_nodes() -> ClusterStatus:
    if not cluster_enabled():
        return ClusterStatus(enabled=False, nodes=[])

    states = _slurm_states()
    specs = _node_specs()
    workers = min(max(len(specs), 1), 8)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gpu-probe") as pool:
        futures = [
            pool.submit(_probe_node, name, address, states.get(name, "unknown"))
            for name, address in specs
        ]
        nodes = [future.result() for future in futures]
    return ClusterStatus(enabled=True, nodes=nodes)


def get_cluster_node(name: str) -> ClusterNode:
    status = list_cluster_nodes()
    for node in status.nodes:
        if node.name == name:
            return node
    raise ValueError(f"Unknown cluster node: {name}")


# ``/opt/robot-platform`` sits on each node's root filesystem, so every worker
# owns its ``train-venv`` and they drift apart: a node whose LeRobot predates a
# policy (or never got its extra installed) rejects ``--policy.type`` in argparse
# and the job dies seconds after being allocated. Asking the node's own
# interpreter which policies it can build turns that into a submit-time error.
_POLICY_PROBE_TTL_S = 300.0
_POLICY_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_policy_probe_cache: dict[tuple[str, str, str], tuple[float, bool | None]] = {}
_policy_probe_lock = threading.Lock()


def _policy_probe_snippet(policy_type: str) -> str:
    """Print ``ok``/``missing``, or ``unknown`` when this LeRobot cannot be asked.

    ``get_policy_class`` is the same lookup ``lerobot_train`` performs, so a
    node that answers ``ok`` can genuinely build the policy. An import failure
    means a LeRobot too different to interrogate, which is reported as unknown
    rather than missing so an unrecognised build is never silently excluded.
    """

    return (
        "try:\n"
        "    from lerobot.policies.factory import get_policy_class\n"
        "except Exception:\n"
        "    print('unknown'); raise SystemExit(0)\n"
        "try:\n"
        f"    get_policy_class({policy_type!r})\n"
        "    print('ok')\n"
        "except Exception:\n"
        "    print('missing')\n"
    )


def node_supports_policy(
    node: ClusterNode, python_executable: str, policy_type: str
) -> bool | None:
    """Return True/False, or ``None`` when support could not be determined.

    Results are cached briefly: a venv does not gain policies mid-session, and
    the probe costs an SSH round trip per node.
    """

    if not policy_type or not _POLICY_NAME_RE.match(policy_type):
        return None
    key = (node.name, python_executable, policy_type)
    now = time.monotonic()
    with _policy_probe_lock:
        cached = _policy_probe_cache.get(key)
        if cached is not None and now - cached[0] < _POLICY_PROBE_TTL_S:
            return cached[1]

    snippet = _policy_probe_snippet(policy_type)
    if _is_local_node(node.name, node.address):
        command = [python_executable, "-c", snippet]
    else:
        # ssh joins its trailing arguments into one remote shell command, so the
        # snippet has to survive a round of shell parsing.
        command = _probe_command(
            node.name, node.address, python_executable, "-c", shlex.quote(snippet)
        )
    try:
        result = _run(command, timeout=30.0)
        answer = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("Policy probe failed on %s: %s", node.name, exc)
        answer = ""

    supported: bool | None
    if answer == "ok":
        supported = True
    elif answer == "missing":
        supported = False
    else:
        supported = None
        logger.warning(
            "Could not determine whether %s supports policy %r; allowing it",
            node.name,
            policy_type,
        )

    with _policy_probe_lock:
        _policy_probe_cache[key] = (now, supported)
    return supported


def select_idle_node(
    min_gpu_memory_mb: int = 0,
    requested_node: str | None = None,
    excluded_nodes: set[str] | None = None,
    policy_type: str | None = None,
    python_executable: str | None = None,
    preferred_node: str | None = None,
) -> ClusterNode:
    status = list_cluster_nodes()
    if not status.enabled:
        raise ValueError("Cluster runner is disabled; set LELAB_CLUSTER_ENABLED=1")

    candidates = status.nodes
    if requested_node and requested_node != "auto":
        candidates = [node for node in candidates if node.name == requested_node]
        if not candidates:
            raise ValueError(f"Unknown cluster node: {requested_node}")

    excluded_nodes = excluded_nodes or set()
    candidates = [
        node
        for node in candidates
        if node.name not in excluded_nodes
        and node.eligible
        and (node.memory_free_mb or 0) >= min_gpu_memory_mb
    ]
    if not candidates:
        suffix = (
            f" with at least {min_gpu_memory_mb} MiB free"
            if min_gpu_memory_mb
            else ""
        )
        raise ValueError(f"No idle GPU node is available{suffix}")

    if policy_type and python_executable:
        with ThreadPoolExecutor(
            max_workers=min(max(len(candidates), 1), 8), thread_name_prefix="policy-probe"
        ) as pool:
            supported = list(
                pool.map(
                    lambda node: node_supports_policy(node, python_executable, policy_type),
                    candidates,
                )
            )
        rejected = [
            node.name for node, ok in zip(candidates, supported, strict=True) if ok is False
        ]
        candidates = [
            node for node, ok in zip(candidates, supported, strict=True) if ok is not False
        ]
        if not candidates:
            raise ValueError(
                f"No idle GPU node can run policy {policy_type!r}; "
                f"{', '.join(sorted(rejected))} "
                f"{'has' if len(rejected) == 1 else 'have'} no such policy in "
                f"{python_executable}"
            )

    # Prefer the node a resumed run started on, so a resume does not wander onto
    # a worker with a different environment; fall back to the emptiest GPU.
    return max(
        candidates,
        key=lambda node: (node.name == preferred_node, node.memory_free_mb or 0),
    )


def _default_templates() -> list[ModelTemplate]:
    return [
        ModelTemplate(
            id="act",
            label="ACT (Action Chunking Transformer)",
            policy_type="act",
            min_gpu_memory_mb=8000,
            description="LeRobot ACT training with the team-managed environment.",
        ),
        ModelTemplate(
            id="diffusion",
            label="Diffusion Policy",
            policy_type="diffusion",
            min_gpu_memory_mb=12000,
            description="LeRobot diffusion-policy training.",
        ),
    ]


def list_model_templates() -> list[ModelTemplate]:
    configured = os.environ.get("LELAB_MODEL_TEMPLATES")
    if not configured:
        return _default_templates()
    path = Path(configured).expanduser()
    try:
        payload = json.loads(path.read_text())
        templates = [ModelTemplate.model_validate(item) for item in payload]
    except Exception as exc:
        raise ValueError(f"Could not load model templates from {path}: {exc}") from exc
    if not templates:
        raise ValueError(f"Model template file is empty: {path}")
    ids = [template.id for template in templates]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Model template IDs must be unique: {path}")
    mismatched = [template.id for template in templates if template.id != template.policy_type]
    if mismatched:
        raise ValueError(
            "For the first-stage UI, every model template ID must equal its "
            f"LeRobot policy_type; mismatched IDs: {', '.join(mismatched)}"
        )
    return templates


def get_model_template(template_id: str) -> ModelTemplate:
    for template in list_model_templates():
        if template.id == template_id:
            return template
    raise ValueError(f"Unknown model template: {template_id}")


__all__ = [
    "ClusterNode",
    "ClusterStatus",
    "ModelTemplate",
    "cluster_enabled",
    "get_cluster_node",
    "get_model_template",
    "list_cluster_nodes",
    "list_model_templates",
    "select_idle_node",
]
