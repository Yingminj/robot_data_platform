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


def test_slurm_batch_script_redirects_caches_away_from_home(tmp_path) -> None:
    """The job user's home need not exist on the worker; caches must not use it."""

    import os
    import subprocess as sp

    from lelab.runners.slurm import SlurmJobRunner

    cache_root = tmp_path / "cache"
    script_path = tmp_path / "job.sbatch"
    script_path.write_text(SlurmJobRunner._batch_script(["env"]))

    # Stub nvidia-smi so the GPU precheck stays out of this test.
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    (stub_dir / "nvidia-smi").write_text("#!/bin/sh\nexit 0\n")
    (stub_dir / "nvidia-smi").chmod(0o755)

    env = {
        **os.environ,
        "HOME": "/nonexistent/robot-train",
        "LELAB_JOB_CACHE_ROOT": str(cache_root),
        "PATH": f"{stub_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
    }
    for name in ("HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME"):
        env.pop(name, None)
    result = sp.run(["bash", str(script_path)], env=env, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert f"TORCH_HOME={cache_root}/torch" in result.stdout
    assert f"HF_HOME={cache_root}/huggingface" in result.stdout
    assert f"XDG_CACHE_HOME={cache_root}/xdg/cache" in result.stdout
    # wandb resolves its staging dir through platformdirs, which reads XDG_DATA_HOME.
    assert f"XDG_DATA_HOME={cache_root}/xdg/data" in result.stdout
    assert f"XDG_CONFIG_HOME={cache_root}/xdg/config" in result.stdout
    # Libraries that hardcode ~ need a home that exists, not just XDG overrides.
    assert f"HOME={cache_root}/home" in result.stdout
    assert (cache_root / "torch").is_dir()
    assert (cache_root / "home").is_dir()


def test_slurm_batch_script_overrides_an_inherited_hf_home(tmp_path) -> None:
    """sbatch exports the service's HF_HOME; a worker need not have that path."""

    import os
    import subprocess as sp

    from lelab.runners.slurm import SlurmJobRunner

    cache_root = tmp_path / "cache"
    script_path = tmp_path / "job.sbatch"
    script_path.write_text(SlurmJobRunner._batch_script(["env"]))

    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    (stub_dir / "nvidia-smi").write_text("#!/bin/sh\nexit 0\n")
    (stub_dir / "nvidia-smi").chmod(0o755)

    env = {
        **os.environ,
        "HOME": "/nonexistent/robot-train",
        # The management host's cache, under a root-owned parent the worker
        # cannot write to.
        "HF_HOME": "/nonexistent/robot-platform/huggingface",
        "LELAB_JOB_CACHE_ROOT": str(cache_root),
        "PATH": f"{stub_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
    }
    result = sp.run(["bash", str(script_path)], env=env, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert f"HF_HOME={cache_root}/huggingface" in result.stdout
    assert (cache_root / "huggingface").is_dir()


def test_slurm_batch_script_keeps_a_writable_home(tmp_path) -> None:
    """A worker that does have the home directory should keep using it."""

    import os
    import subprocess as sp

    from lelab.runners.slurm import SlurmJobRunner

    home = tmp_path / "home"
    home.mkdir()
    script_path = tmp_path / "job.sbatch"
    script_path.write_text(SlurmJobRunner._batch_script(["env"]))

    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    (stub_dir / "nvidia-smi").write_text("#!/bin/sh\nexit 0\n")
    (stub_dir / "nvidia-smi").chmod(0o755)

    env = {
        **os.environ,
        "HOME": str(home),
        "LELAB_JOB_CACHE_ROOT": str(tmp_path / "cache"),
        "PATH": f"{stub_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
    }
    result = sp.run(["bash", str(script_path)], env=env, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert f"HOME={home}\n" in result.stdout
