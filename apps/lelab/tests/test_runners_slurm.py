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
        lambda minimum, requested, excluded, **kwargs: ClusterNode(
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


def _write_dataset(root, episodes: int, cameras: int) -> str:
    """Write the minimal ``meta/info.json`` the decoder-cache sizing reads."""

    import json

    meta = root / "meta"
    meta.mkdir(parents=True)
    features = {f"observation.images.cam{i}": {"shape": [480, 640, 3]} for i in range(cameras)}
    features["observation.state"] = {"shape": [16]}
    (meta / "info.json").write_text(
        json.dumps({"total_episodes": episodes, "features": features})
    )
    return str(root)


def test_video_decoder_cache_covers_every_video_file(tmp_path) -> None:
    """One decoder per video file means the LRU never evicts, which is what leaks."""

    from lelab.runners.slurm import _video_decoder_cache_size

    root = _write_dataset(tmp_path / "ds", episodes=120, cameras=3)

    assert _video_decoder_cache_size(root, num_workers=4, memory_gb=48) == 360


def test_video_decoder_cache_is_clamped_to_the_memory_budget(tmp_path) -> None:
    """A cache that cannot fit is capped rather than pushing the job into an OOM."""

    from lelab.runners.slurm import _video_decoder_cache_size

    root = _write_dataset(tmp_path / "ds", episodes=2000, cameras=3)
    size = _video_decoder_cache_size(root, num_workers=4, memory_gb=48)

    assert size is not None
    assert size < 2000 * 3


def test_video_decoder_cache_is_skipped_when_the_dataset_is_not_local(tmp_path) -> None:
    """A Hub repo id has no readable file count, so LeRobot's own default stands."""

    from lelab.runners.slurm import _video_decoder_cache_size

    assert _video_decoder_cache_size(None, num_workers=4, memory_gb=48) is None
    assert _video_decoder_cache_size(str(tmp_path / "missing"), num_workers=4, memory_gb=48) is None


def test_slurm_batch_script_exports_the_decoder_cache_size(tmp_path) -> None:
    """The env var reaches the training process, and an operator override wins."""

    import os
    import subprocess as sp

    from lelab.runners.slurm import SlurmJobRunner

    script_path = tmp_path / "job.sbatch"
    script_path.write_text(SlurmJobRunner._batch_script(["env"], 360))

    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    (stub_dir / "nvidia-smi").write_text("#!/bin/sh\nexit 0\n")
    (stub_dir / "nvidia-smi").chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{stub_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
    }
    env.pop("LEROBOT_VIDEO_DECODER_CACHE_SIZE", None)

    result = sp.run(["bash", str(script_path)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "LEROBOT_VIDEO_DECODER_CACHE_SIZE=360\n" in result.stdout

    override = sp.run(
        ["bash", str(script_path)],
        env={**env, "LEROBOT_VIDEO_DECODER_CACHE_SIZE": "1024"},
        capture_output=True,
        text=True,
    )
    assert override.returncode == 0, override.stderr
    assert "LEROBOT_VIDEO_DECODER_CACHE_SIZE=1024\n" in override.stdout


def _run_batch_script(tmp_path, decoder_cache_size, preamble: str):
    """Run a generated script under ``preamble`` (used here to set ``ulimit``)."""

    import os
    import subprocess as sp

    from lelab.runners.slurm import SlurmJobRunner

    script_path = tmp_path / "job.sbatch"
    script_path.write_text(
        SlurmJobRunner._batch_script(
            ["bash", "-c", 'echo "SOFT=$(ulimit -Sn) CACHE=${LEROBOT_VIDEO_DECODER_CACHE_SIZE:-unset}"'],
            decoder_cache_size,
        )
    )
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir(exist_ok=True)
    (stub_dir / "nvidia-smi").write_text("#!/bin/sh\nexit 0\n")
    (stub_dir / "nvidia-smi").chmod(0o755)
    env = {**os.environ, "PATH": f"{stub_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}"}
    env.pop("LEROBOT_VIDEO_DECODER_CACHE_SIZE", None)

    return sp.run(
        ["bash", "-c", f"{preamble}\nexec bash {script_path}"],
        env=env,
        capture_output=True,
        text=True,
    )


def test_slurm_batch_script_raises_the_open_file_limit(tmp_path) -> None:
    """One open handle per cached decoder needs more than the inherited soft 1024.

    ``sbatch`` propagates the LeLab service's limits, and systemd defaults the
    soft one to 1024 -- below the file count of any dataset past ~340 episodes
    with three cameras. Workers then die on ``OSError: [Errno 24]``.
    """

    result = _run_batch_script(tmp_path, 360, "ulimit -Sn 256")
    assert result.returncode == 0, result.stderr
    soft = int(result.stdout.split("SOFT=")[1].split()[0])
    assert soft > 256


def test_slurm_batch_script_caps_the_decoder_cache_at_the_fd_budget(tmp_path) -> None:
    """A node that cannot raise the limit far enough falls back to eviction.

    Eviction costs RAM growth and decode time; exhausting the descriptors kills
    the job outright, so the cap is the better failure.
    """

    result = _run_batch_script(tmp_path, 360, "ulimit -n 512")
    assert result.returncode == 0, result.stderr
    assert "CACHE=256" in result.stdout
    assert "Capping LEROBOT_VIDEO_DECODER_CACHE_SIZE" in result.stderr


def _node(name: str, free_mb: int):
    from lelab.cluster import ClusterNode

    return ClusterNode(
        name=name,
        address=f"{name}.local",
        slurm_state="idle",
        reachable=True,
        memory_free_mb=free_mb,
        eligible=True,
    )


def test_select_idle_node_skips_nodes_without_the_policy(monkeypatch) -> None:
    """A worker whose venv lacks the policy is rejected before sbatch, not after."""

    import pytest

    from lelab import cluster

    monkeypatch.setattr(
        cluster,
        "list_cluster_nodes",
        lambda: cluster.ClusterStatus(
            enabled=True, nodes=[_node("mgmt01", 8000), _node("gpu03", 24000)]
        ),
    )
    # gpu03 has the emptiest GPU but no vita, so it must lose anyway.
    monkeypatch.setattr(
        cluster,
        "node_supports_policy",
        lambda node, python_executable, policy_type: node.name != "gpu03",
    )

    chosen = cluster.select_idle_node(
        policy_type="vita", python_executable="/opt/train/bin/python"
    )
    assert chosen.name == "mgmt01"

    monkeypatch.setattr(
        cluster, "node_supports_policy", lambda node, python_executable, policy_type: False
    )
    with pytest.raises(ValueError, match="vita"):
        cluster.select_idle_node(
            policy_type="vita", python_executable="/opt/train/bin/python"
        )


def test_select_idle_node_allows_a_node_it_cannot_probe(monkeypatch) -> None:
    """An unrecognised LeRobot build must not be excluded on a failed probe."""

    from lelab import cluster

    monkeypatch.setattr(
        cluster,
        "list_cluster_nodes",
        lambda: cluster.ClusterStatus(enabled=True, nodes=[_node("gpu03", 24000)]),
    )
    monkeypatch.setattr(
        cluster, "node_supports_policy", lambda node, python_executable, policy_type: None
    )

    assert (
        cluster.select_idle_node(
            policy_type="vita", python_executable="/opt/train/bin/python"
        ).name
        == "gpu03"
    )


def test_select_idle_node_prefers_the_node_a_resume_started_on(monkeypatch) -> None:
    """Resume stays put even when another node has more free GPU memory."""

    from lelab import cluster

    monkeypatch.setattr(
        cluster,
        "list_cluster_nodes",
        lambda: cluster.ClusterStatus(
            enabled=True, nodes=[_node("mgmt01", 8000), _node("gpu03", 24000)]
        ),
    )

    assert cluster.select_idle_node().name == "gpu03"
    assert cluster.select_idle_node(preferred_node="mgmt01").name == "mgmt01"
    # A preference for a node that is not a candidate falls back to auto.
    assert cluster.select_idle_node(preferred_node="gpu99").name == "gpu03"


def test_policy_probe_reads_the_nodes_own_interpreter() -> None:
    """The snippet answers ok/missing and survives the shell ssh runs it through."""

    import shlex
    import subprocess as sp
    import sys

    from lelab.cluster import _policy_probe_snippet

    ok = sp.run(
        [sys.executable, "-c", _policy_probe_snippet("shlex")], capture_output=True, text=True
    )
    assert ok.stdout.strip() == "unknown"  # no lerobot in the test interpreter

    quoted = sp.run(
        ["bash", "-c", f"{shlex.quote(sys.executable)} -c {shlex.quote(_policy_probe_snippet('act'))}"],
        capture_output=True,
        text=True,
    )
    assert quoted.stdout.strip() == "unknown"
    assert quoted.returncode == 0


# --- node-local dataset staging -------------------------------------------------

_DF_STUB = """#!/bin/bash
# Report a filesystem whose free space grows as cached datasets are reclaimed:
# 3 GiB total, 1 GiB consumed per cached dataset directory.
root="${!#}"
count=$(find "$root" -mindepth 1 -maxdepth 1 -type d ! -name '.*' 2>/dev/null | wc -l)
avail=$(( 3145728 - 1048576 * count ))
if [ "$avail" -lt 0 ]; then avail=0; fi
echo "Filesystem 1024-blocks Used Available Capacity Mounted"
echo "stub 3145728 0 $avail 1% /"
"""


def _make_dataset(root, name, files=("meta/info.json", "data/chunk.parquet")):
    dataset = root / name
    for rel in files:
        path = dataset / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}:{rel}")
    return dataset


def _run_staging(tmp_path, src, env_extra=None, stub_df=False):
    """Run a generated batch script and return (completed_process, resolved_root)."""
    import os
    import subprocess as sp

    from lelab.runners.slurm import SlurmJobRunner

    script_path = tmp_path / "job.sbatch"
    script_path.write_text(
        SlurmJobRunner._batch_script(
            ["echo", "TRAIN_ROOT", "--dataset.root", str(src)], None, str(src)
        )
    )

    stub_dir = tmp_path / "bin"
    stub_dir.mkdir(exist_ok=True)
    (stub_dir / "nvidia-smi").write_text("#!/bin/sh\nexit 0\n")
    (stub_dir / "nvidia-smi").chmod(0o755)
    if stub_df:
        (stub_dir / "df").write_text(_DF_STUB)
        (stub_dir / "df").chmod(0o755)

    env = {**os.environ, "PATH": f"{stub_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}"}
    for name in ("HF_HOME", "LELAB_JOB_CACHE_ROOT", "LELAB_DATASET_CACHE_ROOT"):
        env.pop(name, None)
    env.update(env_extra or {})

    result = sp.run(["bash", str(script_path)], env=env, capture_output=True, text=True)
    resolved = None
    for line in result.stdout.splitlines():
        if line.startswith("TRAIN_ROOT "):
            resolved = line.split()[-1]
    return result, resolved


def test_staging_is_inert_until_a_cache_root_is_configured(tmp_path) -> None:
    """The default must keep reading from the NAS, unchanged."""

    src = _make_dataset(tmp_path, "express")
    result, resolved = _run_staging(tmp_path, src)

    assert result.returncode == 0, result.stderr
    assert resolved == str(src)


def test_staging_copies_the_dataset_and_trains_from_the_copy(tmp_path) -> None:
    from lelab.runners.slurm import _STAGE_STAMP, _stage_dir_name

    src = _make_dataset(tmp_path, "express")
    cache = tmp_path / "cache"
    cache.mkdir()

    result, resolved = _run_staging(
        tmp_path,
        src,
        {"LELAB_DATASET_CACHE_ROOT": str(cache), "LELAB_DATASET_CACHE_MIN_FREE_GB": "0"},
    )

    dest = cache / _stage_dir_name(str(src))
    assert result.returncode == 0, result.stderr
    assert resolved == str(dest)
    assert (dest / "meta" / "info.json").read_text() == "express:meta/info.json"
    assert (dest / _STAGE_STAMP).is_file()


def test_staging_refreshes_a_copy_whose_source_changed(tmp_path) -> None:
    """A dataset that gained episodes must not train from the stale copy."""

    from lelab.runners.slurm import _stage_dir_name

    src = _make_dataset(tmp_path, "express")
    cache = tmp_path / "cache"
    cache.mkdir()
    env = {"LELAB_DATASET_CACHE_ROOT": str(cache), "LELAB_DATASET_CACHE_MIN_FREE_GB": "0"}

    _run_staging(tmp_path, src, env)
    (src / "data" / "chunk.parquet").write_text("episode-2")
    (src / "data" / "extra.parquet").write_text("episode-3")
    result, resolved = _run_staging(tmp_path, src, env)

    dest = cache / _stage_dir_name(str(src))
    assert result.returncode == 0, result.stderr
    assert resolved == str(dest)
    assert (dest / "data" / "chunk.parquet").read_text() == "episode-2"
    assert (dest / "data" / "extra.parquet").is_file()


def test_staging_redoes_an_interrupted_copy(tmp_path) -> None:
    """A directory with no stamp is a partial copy and must never be trained on."""

    from lelab.runners.slurm import _STAGE_STAMP, _stage_dir_name

    src = _make_dataset(tmp_path, "express")
    cache = tmp_path / "cache"
    dest = cache / _stage_dir_name(str(src))
    (dest / "data").mkdir(parents=True)
    (dest / "data" / "truncated.parquet").write_text("half a file")

    result, resolved = _run_staging(
        tmp_path,
        src,
        {"LELAB_DATASET_CACHE_ROOT": str(cache), "LELAB_DATASET_CACHE_MIN_FREE_GB": "0"},
    )

    assert result.returncode == 0, result.stderr
    assert resolved == str(dest)
    assert not (dest / "data" / "truncated.parquet").exists()
    assert (dest / _STAGE_STAMP).is_file()


def test_staging_evicts_least_recently_used_datasets_to_make_room(tmp_path) -> None:
    import os

    from lelab.runners.slurm import _STAGE_STAMP

    src = _make_dataset(tmp_path, "express")
    cache = tmp_path / "cache"
    cache.mkdir()
    cached = {}
    for name, stamp_time in (("ds_a", 1000), ("ds_b", 2000), ("ds_c", 3000)):
        entry = _make_dataset(cache, name)
        (entry / _STAGE_STAMP).write_text(str(stamp_time))
        os.utime(entry / _STAGE_STAMP, (stamp_time, stamp_time))
        cached[name] = entry

    result, _ = _run_staging(
        tmp_path,
        src,
        {"LELAB_DATASET_CACHE_ROOT": str(cache), "LELAB_DATASET_CACHE_MIN_FREE_GB": "1"},
        stub_df=True,
    )

    assert result.returncode == 0, result.stderr
    assert not cached["ds_a"].exists(), "the oldest copy should be reclaimed first"
    assert not cached["ds_b"].exists()
    assert cached["ds_c"].exists(), "eviction should stop once the dataset fits"


def test_staging_never_evicts_a_dataset_another_job_is_reading(tmp_path) -> None:
    import fcntl
    import os

    from lelab.runners.slurm import _STAGE_STAMP

    src = _make_dataset(tmp_path, "express")
    cache = tmp_path / "cache"
    (cache / ".locks").mkdir(parents=True)
    cached = {}
    for name, stamp_time in (("ds_a", 1000), ("ds_b", 2000), ("ds_c", 3000)):
        entry = _make_dataset(cache, name)
        (entry / _STAGE_STAMP).write_text(str(stamp_time))
        os.utime(entry / _STAGE_STAMP, (stamp_time, stamp_time))
        cached[name] = entry

    # Stand in for a running job holding the shared lock on the oldest copy.
    lock_path = cache / ".locks" / "ds_a.lock"
    lock_path.touch()
    held = os.open(lock_path, os.O_RDWR)
    fcntl.flock(held, fcntl.LOCK_SH)
    try:
        result, _ = _run_staging(
            tmp_path,
            src,
            {"LELAB_DATASET_CACHE_ROOT": str(cache), "LELAB_DATASET_CACHE_MIN_FREE_GB": "1"},
            stub_df=True,
        )
    finally:
        fcntl.flock(held, fcntl.LOCK_UN)
        os.close(held)

    assert result.returncode == 0, result.stderr
    assert cached["ds_a"].exists(), "a dataset in use must survive the sweep"
    assert not cached["ds_b"].exists()


def test_staging_falls_back_to_the_nas_when_the_cache_cannot_fit_it(tmp_path) -> None:
    """A cache too small to hold the dataset must degrade, not fail the job."""

    src = _make_dataset(tmp_path, "express")
    cache = tmp_path / "cache"
    cache.mkdir()

    result, resolved = _run_staging(
        tmp_path,
        src,
        {
            "LELAB_DATASET_CACHE_ROOT": str(cache),
            "LELAB_DATASET_CACHE_MIN_FREE_GB": "1000000",
        },
    )

    assert result.returncode == 0, result.stderr
    assert resolved == str(src)
    assert "training from the NAS" in result.stderr


def test_stage_dir_name_separates_datasets_sharing_a_basename() -> None:
    from lelab.runners.slurm import _stage_dir_name

    first = _stage_dir_name("/mnt/robot_platform/datasets/team_a/express")
    second = _stage_dir_name("/mnt/robot_platform/datasets/team_b/express")

    assert first != second
    assert first.startswith("express-")
    assert "/" not in first
