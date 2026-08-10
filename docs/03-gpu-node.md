# GPU worker installation

**English** | [简体中文](03-gpu-node.zh-CN.md)

A GPU worker runs the NVIDIA driver, Docker with the NVIDIA runtime, Munge, `slurmd` and the shared LeRobot training environment. This document uses `gpu01` as the example; **the steps are identical on every GPU worker** — only the hostname, IP and SSH account change. Unless stated otherwise, commands are run from the repository root on that worker.

The current workers:

| Slurm NodeName | IP | Administrator SSH target |
|---|---|---|
| `gpu01` | `192.168.100.215` | `snorlax@192.168.100.215` |
| `gpu02` | `192.168.100.216` | `yang@192.168.100.216` |
| `gpu03` | `192.168.100.217` | `snorlax@192.168.100.217` |

Slurm jobs always run as `robot-train`.

**The SSH account is only used for logging in and for leLab's read-only GPU probing; the Slurm node name has nothing to do with it.** In the table above, `gpu02` uses the account `yang`, unlike the other two — that is allowed. The account does not have to match the NodeName, and it does not have to be the same across nodes. `hostname -s` must equal the Slurm NodeName.

When adding a node to an already running cluster, there is a set of cluster-level changes beyond this document — see [Adding a GPU node to an existing cluster](09-add-gpu-node.md).

## 1. Pre-installation checks

```bash
hostname -s
ip -br address
cp config/site.env.example config/site.env
editor config/site.env
```

If the hostname is not `gpu01` yet:

```bash
sudo hostnamectl set-hostname gpu01
```

After logging in again, run:

```bash
sudo ./scripts/05-configure-hosts.sh --apply
./scripts/00-audit-host.sh gpu
nvidia-smi
timedatectl show --property=NTPSynchronized --value
getent hosts mgmt01 gpu01
```

## 2. Prepare Python 3.12

```bash
python3.12 --version
```

How it is installed in the current Ubuntu 22.04 environment:

```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-dev python3.12-venv
```

Workers do not need Node.js or npm.

## 3. Install the worker base components

```bash
sudo ./scripts/20-install-gpu-node.sh --apply
```

The script installs or prepares:

- NFS, Chrony, Munge, SSH;
- Docker and the NVIDIA Container Toolkit;
- `robotdata` and `robot-train`;
- `/cache/datasets`, `/cache/exports`, `/work/runs`;
- `/var/lib/robot-platform/slurmd`;
- the persistent read/write mount of the QNAP.

On the first run it will not start `slurmd` with an unknown Munge key. That is expected — the cluster configuration is installed later.

Check the local directories and NFS:

```bash
sudo -u robot-train test -d /cache/datasets
sudo -u robot-train test -d /cache/exports
sudo -u robot-train test -d /work/runs
sudo -u robot-train test -w /var/lib/robot-platform/cache
sudo -u robot-train test -r /mnt/robot_platform/datasets
sudo -u robot-train test -w /mnt/robot_platform/jobs
findmnt /mnt/robot_platform
```

If any of these local directories are missing because an earlier installation was interrupted, create them:

```bash
sudo install -d -o robot-train -g robotdata -m 0750 \
  /cache/datasets \
  /cache/exports \
  /work/runs \
  /var/lib/robot-platform/huggingface \
  /var/lib/robot-platform/cache
```

`/var/lib/robot-platform/cache` is leLab's `LELAB_JOB_CACHE_ROOT` and **must exist on every worker and be writable by `robot-train`**. Slurm points `HOME` at the `robot-train` home directory, which does not exist on the workers, so a job that caches into `~` fails on that node. The same local path on each node is enough; shared storage is not needed.

When the NFS mount fails, first confirm the QNAP allow list contains the IP of **this** new node — that is the item most often missed when adding a node.

## 4. Install the shared training environment

```bash
sudo ./scripts/25-install-training-environment.sh --apply
sudo -u robot-train /opt/robot-platform/train-venv/bin/python -c \
  'import torch, lerobot; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

CUDA is expected to be `True`. Do not install Python packages ad hoc at job start time; all workers should use the same path and the same versions.

## 5. Install Slurm 26.05.2

The whole cluster must have exactly the same Slurm major version and package build. Follow [the Slurm 26.05.2 DEB installation notes](Slurm-INSTALL.md), then check:

```bash
slurmd -V
stat -fc %T /sys/fs/cgroup
test -e /usr/lib/x86_64-linux-gnu/slurm-wlm/cgroup_v2.so \
  || test -e /usr/lib/x86_64-linux-gnu/slurm/cgroup_v2.so
```

Expected:

```text
slurm 26.05.2
cgroup2fs
```

At this point `sbatch --version` reports an error. **That is normal and is not evidence of a failed installation**:

```text
sbatch: error: resolve_ctls_from_dns_srv: res_nsearch error: Unknown host
sbatch: error: fetch_config: DNS SRV lookup failed
sbatch: fatal: Could not establish a configuration source
```

Newer Slurm tries to load a configuration before printing the version. This machine has no `/etc/slurm/slurm.conf` yet, so it falls back to configless DNS SRV discovery, which this cluster does not use. The error disappears once the configuration is installed in step 7. Use `/usr/sbin/slurmd -V` to check the version — it is unaffected.

## 6. Collect the real hardware parameters

On `gpu01`:

```bash
sudo slurmd -C
```

Copy `CPUs`, `Boards`, `SocketsPerBoard`, `CoresPerSocket`, `ThreadsPerCore` and `RealMemory` from the output into `config/slurm/nodes.conf` on the management node. Do not copy the example values.

`RealMemory` should be slightly below the physical memory, leaving headroom for the OS and background services.

## 7. Receive and install the cluster configuration

Render the configuration on `mgmt01` first. Below is one concrete secure-transfer example.

### 7.1 Stage and copy on mgmt01

> **Run the whole block at once; do not paste only the second half.** The `:?` in each `${stage_dir:?}` must not be dropped: if the first line `stage_dir="$(mktemp -d)"` is missed, the variable is empty, `"$stage_dir/munge.key"` becomes `/munge.key`, and the `sudo` line **silently succeeds**, writing the cluster's unique authentication key into the filesystem root. With `:?`, bash immediately reports `stage_dir: parameter null or not set` and aborts.

```bash
stage_dir="$(mktemp -d)"
sudo install -o "$USER" -g "$(id -gn)" -m 0600 \
  /etc/munge/munge.key \
  "${stage_dir:?}/munge.key"
install -m 0644 \
  config/slurm/slurm.conf.generated \
  "${stage_dir:?}/slurm.conf.generated"

ssh snorlax@192.168.100.215 \
  'install -d -m 0700 ~/robot-platform-secure'
scp \
  "${stage_dir:?}/munge.key" \
  "${stage_dir:?}/slurm.conf.generated" \
  snorlax@192.168.100.215:~/robot-platform-secure/

shred -u "${stage_dir:?}/munge.key"
rm -f "${stage_dir:?}/slurm.conf.generated"
rmdir "${stage_dir:?}"
```

If it was already written to the root directory by mistake, check and destroy it:

```bash
ls -l /munge.key && sudo shred -u /munge.key
```

### 7.2 Install on gpu01

From the repository root on `gpu01`:

```bash
sudo ./scripts/cluster/install-worker-config.sh \
  /home/snorlax/robot-platform-secure/munge.key \
  /home/snorlax/robot-platform-secure/slurm.conf.generated \
  --apply
```

Both file paths must really exist on **the machine running the command**. `/secure/temp/...` is only a placeholder from older documentation, not a preset directory. If the script only prints usage, then one of the two files is unreadable, or the third argument is not `--apply`.

After verifying, destroy the temporary key on the worker:

```bash
shred -u /home/snorlax/robot-platform-secure/munge.key
rm -f /home/snorlax/robot-platform-secure/slurm.conf.generated
rmdir /home/snorlax/robot-platform-secure
```

Check:

```bash
systemctl is-active munge slurmd
sudo slurmd -G
sha256sum /etc/slurm/slurm.conf /etc/slurm/cgroup.conf /etc/slurm/gres.conf
journalctl -u slurmd -n 50 --no-pager
```

The three checksums must match those on `mgmt01` exactly. That is the only reliable way to tell whether the configuration was really installed correctly.

`slurmd -G` prints a GRES type notice, **which is normal**:

```text
gres/gpu: _normalize_sys_gres_types: Could not find an unused configuration record
with a GRES type that is a substring of system device `nvidia_geforce_rtx_4090`.
Setting system GRES type to NULL
```

`gres.conf` declares `Name=gpu` with no model, NVML reports the device model as `nvidia_geforce_rtx_4090`, so Slurm sets the type to NULL — consistent with the equally model-free `Gres=gpu:1` in `nodes.conf`. The line right after it is the actual conclusion:

```text
Gres Name=gpu Type=(null) Count=1 Index=0 File=/dev/nvidia0 Flags=HAS_FILE,ENV_NVML
```

## 8. Run a scheduling test from the management node

Do not treat running Python over a direct SSH session as Slurm acceptance. Go back to `mgmt01` and run:

```bash
srun \
  --partition=debug \
  --nodes=1 \
  --nodelist=gpu01 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --mem=1G \
  --gres=gpu:1 \
  --time=00:02:00 \
  /opt/robot-platform/train-venv/bin/python -c \
  'import os, socket, torch; print(socket.gethostname()); print(os.environ.get("CUDA_VISIBLE_DEVICES")); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

## 9. Worker operating conventions

- production training goes through Slurm, not through a personal SSH session;
- `/mnt/robot_platform/jobs` holds logs and checkpoints and is shared persistent data;
- `/cache` and `/work` are local, rebuildable space and must not hold the only copy of anything;
- a manual CUDA process makes leLab mark that node as unschedulable (`eligible: false`);
- ordinary CPU processes do not affect leLab's GPU-idle judgement;
- the Docker TCP API is not exposed;
- workers do not run `slurmctld`, PostgreSQL, Redis, MLflow or leLab.

Remote desktop tools (RustDesk, TeamViewer, Sunlogin and similar) occupy the GPU and are counted as compute processes, which keeps the node at `eligible: false` indefinitely. leLab ships an allow list of graphics process names (see `graphics_patterns` in `apps/lelab/lelab/cluster.py`); if the tool in use on this machine is not in it, add the process name rather than working around it by closing the tool.
