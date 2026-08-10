# Slurm cluster finalization and acceptance

**English** | [简体中文](06-cluster-finalization.zh-CN.md)

This document is the authoritative configuration order for the current cluster (`mgmt01` + `gpu01` + `gpu02` + `gpu03`). Every command marked "on the management node" is run from the repository root on `mgmt01`; every command marked "on the worker" is run from the repository root on that worker.

**Do not follow this document to add a node to an already running cluster** — see [Adding a GPU node to an existing cluster](09-add-gpu-node.md).

## 0. Entry conditions

Before starting this stage:

- `10` and `25` have been run on `mgmt01`;
- `20` and `25` have been run on every GPU worker;
- the same Slurm 26.05.2 is installed on every machine;
- every machine uses cgroup v2;
- hostnames, time and `/etc/hosts` are correct on every machine;
- the QNAP is mounted at the same absolute path `/mnt/robot_platform` everywhere, and the allow list contains every node IP;
- the numeric UID/GID of `robot-train` and `robotdata` are the same on every machine.

Quick check on **every** machine:

```bash
hostname -s
slurmd -V
stat -fc %T /sys/fs/cgroup
id robot-train
getent group robotdata
getent hosts mgmt01 gpu01 gpu02 gpu03
findmnt /mnt/robot_platform
```

The output of `hostname -s` must exactly match the NodeName that will be written into `nodes.conf`.

At this point `sbatch --version` on a worker reports `DNS SRV lookup failed`, **which is normal**: the machine has no `/etc/slurm/slurm.conf` yet, so newer Slurm falls back to the configless discovery this cluster does not use. Checking the version with `/usr/sbin/slurmd -V` is unaffected, and the error disappears once the configuration is installed.

## 1. Collect node resources

Run on **every** host separately:

```bash
sudo slurmd -C
```

Copy the template on `mgmt01`:

```bash
cp config/slurm/nodes.conf.example config/slurm/nodes.conf
editor config/slurm/nodes.conf
```

Every line should:

- use a fixed `NodeName=` equal to that machine's `hostname -s`;
- use the matching `NodeAddr`;
- keep the CPU topology from **that machine's own** `slurmd -C`;
- set `RealMemory` no higher than the value `slurmd -C` measured;
- append `Gres=gpu:1 State=UNKNOWN`.

The current structure, as an example:

```ini
NodeName=mgmt01 NodeAddr=192.168.100.202 CPUs=... RealMemory=61912 Gres=gpu:1 State=UNKNOWN
NodeName=gpu01 NodeAddr=192.168.100.215 CPUs=... RealMemory=61920 Gres=gpu:1 State=UNKNOWN
NodeName=gpu02 NodeAddr=192.168.100.216 CPUs=... RealMemory=61919 Gres=gpu:1 State=UNKNOWN
NodeName=gpu03 NodeAddr=192.168.100.217 CPUs=... RealMemory=61914 Gres=gpu:1 State=UNKNOWN
```

Do not leave any `FILL_ME` behind. **`RealMemory` usually differs by a few MB between machines; do not copy one machine's value to another for the sake of tidiness** — a declared value above the measured one puts the node into `INVAL`.

The number of lines must equal the number of nodes in `GPU_NODE_NAMES` in `config/site.env`, or the render script reports:

```text
expected 4 nodes, found 2
```

## 2. Render and review the configuration

On `mgmt01`:

```bash
./scripts/cluster/render-slurm-config.sh
sed -n '1,240p' config/slurm/slurm.conf.generated
```

Confirm:

```bash
! grep -E 'FILL_ME|@@' config/slurm/slurm.conf.generated
grep -c '^NodeName=' config/slurm/slurm.conf.generated
grep '^PartitionName=' config/slurm/slurm.conf.generated
```

The node count should match `GPU_NODE_NAMES`, and all three partitions `debug`, `train` and `eval` should list every node:

```ini
PartitionName=debug Nodes=mgmt01,gpu01,gpu02,gpu03 Default=YES MaxTime=01:00:00 State=UP
```

## 3. Install the controller and the mgmt01 worker

On `mgmt01`:

```bash
sudo ./scripts/cluster/install-controller-config.sh \
  config/slurm/slurm.conf.generated \
  --apply
```

> **The argument order is the reverse of the numbered scripts.** Here the configuration file path is the first argument and `--apply` is the second, whereas the numbered scripts such as `10`/`20`/`25` put `--apply` first. Writing only `sudo ./scripts/cluster/install-controller-config.sh --apply` makes `--apply` be read as the configuration filename, and the script prints the same message it prints with no arguments at all:
>
> ```text
> This script changes the host. Re-run it with --apply after reviewing config/site.env.
> ```
>
> Seeing that line does not indicate any problem other than the misplaced `--apply`; just add the configuration file path.

The script restarts `slurmctld` **and the local `slurmd` on mgmt01**, so check `squeue` before running it: jobs running on this machine will be interrupted.

It installs the following files into `/etc/slurm`:

```text
slurm.conf
cgroup.conf
gres.conf
```

and restarts `munge`, `slurmctld` and the local `slurmd`. Check:

```bash
systemctl is-active munge slurmctld slurmd
munge -n | unmunge | sed -n '1,12p'
sudo slurmd -G
scontrol ping
```

## 4. Install every worker

**Every** worker must receive exactly the same:

- `/etc/munge/munge.key`;
- generated `slurm.conf`;
- `cgroup.conf` and `gres.conf` from the repository.

> **When the topology changes, `slurm.conf` must be redistributed to the existing nodes as well.** Slurm requires the configuration to be byte-for-byte identical cluster-wide. If only the new node is updated after adding it, the existing nodes still hold the old configuration without that node, and they fail once the controller reloads. The full add-a-node flow is in [Adding a GPU node to an existing cluster](09-add-gpu-node.md).

The Munge key may only travel over a temporary channel the administrator controls; it must never go into Git, a public NAS directory or a chat log. The concrete `scp` staging commands are in [GPU worker installation: receive and install the cluster configuration](03-gpu-node.md#7-receive-and-install-the-cluster-configuration).

The invocation on each worker is:

```bash
sudo ./scripts/cluster/install-worker-config.sh \
  <local path to munge.key on that worker> \
  <local path to slurm.conf.generated on that worker> \
  --apply
```

The first two arguments are positional and cannot be swapped; both files must already exist on **the machine running the command**. If a non-existent `/secure/temp/...` is passed, the script only prints usage.

During installation `slurmd -G` prints a GRES type notice, **which is normal**:

```text
gres/gpu: _normalize_sys_gres_types: Could not find an unused configuration record
with a GRES type that is a substring of system device `nvidia_geforce_rtx_4090`.
Setting system GRES type to NULL
```

`gres.conf` declares `Name=gpu` with no model, NVML reports the device model as `nvidia_geforce_rtx_4090`, so Slurm sets the type to NULL — consistent with the equally model-free `Gres=gpu:1` in `nodes.conf`. The line right after it is the actual conclusion:

```text
Gres Name=gpu Type=(null) Count=1 Index=0 File=/dev/nvidia0 Flags=HAS_FILE,ENV_NVML
```

`gres.conf` only needs changing if you want to request GPUs by model (`--gres=gpu:rtx4090:1`).

## 5. Compare all machines

Run on each machine:

```bash
slurmd -V
stat -fc %T /sys/fs/cgroup
sha256sum \
  /etc/slurm/slurm.conf \
  /etc/slurm/cgroup.conf \
  /etc/slurm/gres.conf
sudo sha256sum /etc/munge/munge.key
```

The checksums of the three Slurm configuration files and of the Munge key must each be identical across **all** machines. **Compare checksums only; never send the contents of the Munge key.**

On `mgmt01`:

```bash
scontrol ping
sinfo -N -l
sinfo -o '%P|%N|%T|%c|%m|%G'
scontrol show nodes
squeue
```

Every node is expected to:

- be `idle` in all three partitions `debug`, `train` and `eval`;
- have `gpu:1` in its GRES;
- have an address matching `NodeAddr` in `nodes.conf`;
- have CPU and memory matching its own configuration.

## 6. cgroup v2 and GPU isolation checks

Check on each machine:

```bash
stat -fc %T /sys/fs/cgroup
sudo slurmd -G
journalctl -u slurmd -b --no-pager | \
  grep -Ei 'cgroup|gres|gpu|error|fatal'
```

`config/slurm/cgroup.conf` currently enables:

```ini
CgroupPlugin=autodetect
ConstrainCores=yes
ConstrainRAMSpace=yes
ConstrainDevices=yes
ConstrainSwapSpace=yes
```

`gres.conf` uses NVML autodetection of `/dev/nvidia0`. If `slurmd -G` reports a mismatched GPU count or device, fix the NVIDIA driver and `gres.conf` first; do not simply force the node to RESUME.

## 7. Per-node GPU smoke test

Run on `mgmt01` only:

```bash
for node in mgmt01 gpu01 gpu02 gpu03; do
  echo "[$node]"
  srun \
    --partition=debug \
    --nodes=1 \
    --nodelist="$node" \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=1G \
    --gres=gpu:1 \
    --time=00:02:00 \
    /opt/robot-platform/train-venv/bin/python -c \
    'import os, socket, torch, lerobot; print("host=", socket.gethostname()); print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES")); print("cuda=", torch.cuda.is_available()); print("gpu=", torch.cuda.get_device_name(0)); print("lerobot=ok")'
done
```

Every task must:

- run on the specified node;
- see only the GPU it was allocated;
- report `cuda=True`;
- import LeRobot successfully.

Then confirm the nodes really are **different physical machines**:

```bash
for node in mgmt01 gpu01 gpu02 gpu03; do
  echo -n "$node: "
  srun --partition=debug --nodelist="$node" --gres=gpu:1 --time=00:02:00 nvidia-smi -L
done
```

The returned GPU UUIDs must all differ. **A repeated UUID means `NodeAddr` is wrong** and two NodeNames point at the same physical machine — in that case `sinfo` looks perfectly fine and only the UUID reveals the problem.

Then run a concurrent-occupancy test:

```bash
for node in mgmt01 gpu01 gpu02 gpu03; do
  srun --partition=debug --nodelist="$node" --gres=gpu:1 --time=00:02:00 \
    bash -c 'nvidia-smi -L; sleep 20' &
done
wait
```

This warning during the run can be ignored:

```text
error: couldn't chdir to `/home/kewei/YING/robot_data_platform': No such file or directory: going to /tmp instead
```

`srun` passes the submitting side's current directory to the remote end, and that repository path only exists on `mgmt01`. Jobs submitted by leLab use absolute paths and are unaffected; add `--chdir=/tmp` to silence the warning.

## 8. Common abnormal states

| Symptom | Check first |
|---|---|
| `DOWN` / `NOT_RESPONDING` | the `slurmd` service, port 6818, the firewall, hostname, time |
| `INVAL` | `slurmd -C` against the NodeName line, RealMemory, CPU topology |
| `Invalid generic resource` | `sudo slurmd -G`, `gres.conf`, NVML, `/dev/nvidia0` |
| Munge authentication failure | key checksum, `0400 munge:munge`, time synchronization |
| cgroup plugin fails to load | Slurm version, `cgroup2fs`, `cgroup_v2.so` |
| job stuck in `PENDING` | the `Reason` in `scontrol show job <id>` |
| remote job says it cannot enter the submission directory | a normal warning, see section 7 |
| existing nodes go `DOWN` after adding a node | the existing nodes' `slurm.conf` was not updated, see section 4 |

The following three kinds of output are **not** failures; do not reinstall because of them:

| Output | When it appears | Explanation |
|---|---|---|
| `DNS SRV lookup failed` | `sbatch --version` on a worker before the config is installed | the machine has no `slurm.conf` yet and falls back to the unused configless discovery |
| `_normalize_sys_gres_types ... Setting system GRES type to NULL` | on every `slurmd -G` | `gres.conf` uses a model-free `Name=gpu`, consistent with `Gres=gpu:1` |
| `couldn't chdir to ...: going to /tmp instead` | `srun` submitted from the repository directory on `mgmt01` | the submission directory does not exist on the worker; the job itself is unaffected |

Logs:

```bash
# mgmt01
journalctl -u slurmctld -u slurmd -u munge -n 150 --no-pager

# every worker
journalctl -u slurmd -u munge -n 150 --no-pager
```

## 9. Next step after Slurm is done

Only once every check on this page passes, install leLab on `mgmt01`:

```bash
sudo ./scripts/15-install-lelab-platform.sh --apply
```

Then complete [leLab SSH probing and API acceptance](07-lelab-cluster-web.md), and finally submit the first short job using a small LeRobot dataset on the NAS.
