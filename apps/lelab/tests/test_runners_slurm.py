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

import subprocess


def test_slurm_runner_submits_one_gpu_job(tmp_path, monkeypatch) -> None:
    from lelab.cluster import ClusterNode, ModelTemplate
    from lelab.jobs import TrainingMetrics
    from lelab.runners import slurm
    from lelab.train import TrainingRequest

    monkeypatch.setattr(
        slurm,
        "get_model_template",
        lambda template_id: ModelTemplate(
            id="act",
            label="ACT",
            policy_type="act",
            python_executable="/opt/train/bin/python",
        ),
    )
    monkeypatch.setattr(
        slurm,
        "select_idle_node",
        lambda minimum, requested, excluded: ClusterNode(
            name="gpu02",
            address="gpu02",
            slurm_state="idle",
            reachable=True,
            memory_free_mb=20000,
            eligible=True,
        ),
    )
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "12345\n", "")

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)
    monkeypatch.setattr(slurm.SlurmJobRunner, "_start_tailing", lambda self: None)

    output_dir = tmp_path / "jobs" / "job-1" / "run"
    runner = slurm.SlurmJobRunner(TrainingMetrics(), tmp_path / "log.jsonl")
    runner.start(
        "job-1",
        TrainingRequest(dataset_repo_id="team/dataset", policy_type="act"),
        str(output_dir),
    )

    assert runner.slurm_job_id() == "12345"
    assert runner.node_name() == "gpu02"
    sbatch = calls[0]
    assert "--gres=gpu:1" in sbatch
    assert "--nodelist=gpu02" in sbatch
    script = (output_dir.parent / "job.sbatch").read_text()
    assert "/opt/train/bin/python -m lerobot.scripts.lerobot_train" in script
    assert "--dataset.repo_id team/dataset" in script


def test_slurm_batch_script_rechecks_gpu_before_training() -> None:
    from lelab.runners.slurm import SlurmJobRunner

    script = SlurmJobRunner._batch_script(["python", "-m", "trainer"])

    assert "--query-compute-apps=pid" in script
    assert "exit 75" in script
    assert "exec python -m trainer" in script
