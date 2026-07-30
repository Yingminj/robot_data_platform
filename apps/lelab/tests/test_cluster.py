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

from __future__ import annotations

import json
import subprocess

import pytest


def test_model_templates_load_registered_file(tmp_path, monkeypatch) -> None:
    from lelab.cluster import list_model_templates

    path = tmp_path / "templates.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "act",
                    "label": "Team ACT",
                    "policy_type": "act",
                    "python_executable": "/opt/train/bin/python",
                }
            ]
        )
    )
    monkeypatch.setenv("LELAB_MODEL_TEMPLATES", str(path))

    templates = list_model_templates()

    assert templates[0].id == "act"
    assert templates[0].python_executable == "/opt/train/bin/python"


def test_cluster_disabled_by_default(monkeypatch) -> None:
    from lelab.cluster import list_cluster_nodes

    monkeypatch.delenv("LELAB_CLUSTER_ENABLED", raising=False)
    assert list_cluster_nodes().model_dump() == {"enabled": False, "nodes": []}


def test_select_idle_node_prefers_more_free_memory(monkeypatch) -> None:
    from lelab import cluster

    monkeypatch.setenv("LELAB_CLUSTER_ENABLED", "1")
    monkeypatch.setenv("LELAB_CLUSTER_NODES", "gpu01,gpu02")

    def fake_run(command, timeout=5.0):
        if command[0] == "sinfo":
            return subprocess.CompletedProcess(command, 0, "gpu01|idle\ngpu02|idle\n", "")
        if "--query-gpu=name,memory.total,memory.free" in command:
            free = "22000" if "gpu02" in command else "12000"
            return subprocess.CompletedProcess(command, 0, f"RTX, 24000, {free}\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(cluster, "_run", fake_run)

    assert cluster.select_idle_node(8000).name == "gpu02"


def test_select_idle_node_excludes_web_reserved_gpu(monkeypatch) -> None:
    from lelab import cluster

    monkeypatch.setenv("LELAB_CLUSTER_ENABLED", "1")
    monkeypatch.setenv("LELAB_CLUSTER_NODES", "gpu01,gpu02")

    def fake_run(command, timeout=5.0):
        if command[0] == "sinfo":
            return subprocess.CompletedProcess(command, 0, "gpu01|idle\ngpu02|idle\n", "")
        if "--query-gpu=name,memory.total,memory.free" in command:
            return subprocess.CompletedProcess(command, 0, "RTX, 24000, 22000\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(cluster, "_run", fake_run)

    assert cluster.select_idle_node(excluded_nodes={"gpu01"}).name == "gpu02"


def test_select_idle_node_rejects_manual_cuda_process(monkeypatch) -> None:
    from lelab import cluster

    monkeypatch.setenv("LELAB_CLUSTER_ENABLED", "1")
    monkeypatch.setenv("LELAB_CLUSTER_NODES", "gpu01")

    def fake_run(command, timeout=5.0):
        if command[0] == "sinfo":
            return subprocess.CompletedProcess(command, 0, "gpu01|idle\n", "")
        if "--query-gpu=name,memory.total,memory.free" in command:
            return subprocess.CompletedProcess(command, 0, "RTX, 24000, 22000\n", "")
        return subprocess.CompletedProcess(command, 0, "4321\n", "")

    monkeypatch.setattr(cluster, "_run", fake_run)

    with pytest.raises(ValueError, match="No idle GPU node"):
        cluster.select_idle_node()
