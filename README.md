# Robot Data Platform Deployment Package

**English** | [简体中文](README.zh-CN.md)

This repository deploys a small robot training platform: a QNAP NAS provides shared data, `mgmt01` acts as both the management plane and one GPU worker, the remaining nodes contribute more GPUs, Slurm handles scheduling, and leLab provides the training web UI.

Once deployed, leLab presents all four GPUs as a single compute target. A training job can either let Slurm pick a free GPU automatically or be pinned to a specific node:

<img src="apps/assets/1.png" alt="leLab compute target selection: mgmt01, gpu01, gpu02, gpu03, four RTX 4090s" width="600">

The authoritative site topology is `config/site.env.example`:

| Host | Role | Address | SSH login | Install steps |
|---|---|---|---|---|
| `mgmt01` | Management node, Slurm controller, GPU worker, leLab | `192.168.100.202` | local, no SSH | `10`, `25`, controller config, `15` |
| `gpu01` | GPU worker | `192.168.100.215` | `snorlax` | `20`, `25`, worker config |
| `gpu02` | GPU worker | `192.168.100.216` | `yang` | `20`, `25`, worker config |
| `gpu03` | GPU worker | `192.168.100.217` | `snorlax` | `20`, `25`, worker config |
| QNAP | NFS storage | `192.168.100.184:/robot_platform` | — | share and permissions only |

**A Slurm NodeName and an SSH login account are two different things and cannot substitute for each other.** The Slurm names are `gpu01`/`gpu02`/`gpu03`; the SSH targets are `snorlax@192.168.100.215`, `yang@192.168.100.216`, and `snorlax@192.168.100.217`. The SSH account does not have to match the NodeName, and it does not have to be the same across nodes (`gpu02` above differs from the other two).

To add a node to an already running cluster, do not repeat the first-time deployment flow — see [Adding a GPU node to an existing cluster](docs/09-add-gpu-node.md).

## Start here

For a first deployment, follow the order below exactly. Do not infer the execution order from the script numbers: `15-install-lelab-platform.sh` has a low number but must run after Slurm and the training environment are complete on every machine.

```text
0. On every host, verify site.env, hostname, driver, network and time
1. On QNAP, create the shared directories and authorize every node IP
2. On every host, install the managed /etc/hosts block
3. On mgmt01, install the management components, MLflow and the training environment
4. On every GPU worker, install the worker components and the training environment
5. On every host, install the same cgroup v2-capable Slurm 26.05.2
6. On mgmt01, render the Slurm config; install the controller/worker configs separately
7. Run the GPU smoke test on each node
8. On mgmt01, install leLab and configure SSH probing to every worker
9. Check the API, add a small dataset, submit the first short training job
```

The following orderings cannot be swapped — if they are, the failure symptom will not point at the real cause:

- Step 5 must come after steps 3 and 4. The role scripts first install the old Slurm shipped with Ubuntu; 26.05.2 is the version that must win in the end.
- Step 6 must come after step 5, and must wait until **every** node has Slurm installed and has run `slurmd -C`, because rendering needs each machine's real hardware parameters.
- Step 8 must come after step 7. The leLab installer checks that `sbatch`/`sinfo` work.

Corresponding documents:

| Stage | Document |
|---|---|
| QNAP | [QNAP NAS setup](docs/01-qnap-nas.md) |
| Management node | [Management node installation](docs/02-management-node.md) |
| GPU worker | [GPU node installation](docs/03-gpu-node.md) |
| Slurm 26.05.2 DEB | [Slurm 26.05.2 installation](docs/Slurm-INSTALL.md) |
| Slurm config and acceptance | [Slurm cluster finalization](docs/06-cluster-finalization.md) |
| leLab | [leLab cluster web](docs/07-lelab-cluster-web.md) |
| **Scale out: add a GPU node** | [Adding a GPU node to an existing cluster](docs/09-add-gpu-node.md) |
| Common errors | [Installation and runtime troubleshooting](docs/08-troubleshooting.md) |
| Optional collector nodes | [Collector node](docs/04-collector-node.md), [Combined collector and GPU node](docs/05-combined-node.md) |

## Prerequisites

Every Linux host must satisfy the following:

- the currently validated environment is Ubuntu 22.04, with a static IP and automatic suspend disabled;
- the repository has been copied to the machine, and commands are run from the repository root;
- the NVIDIA driver is installed and `nvidia-smi` works;
- the host can reach QNAP TCP 2049 and the management node on TCP 6817, and the management node can reach workers on TCP 6818;
- system time is synchronized;
- the UID of `robot-train` and the GID of `robotdata` are the same number on **all** nodes;
- Slurm 26.05.2 is used, and `stat -fc %T /sys/fs/cgroup` prints `cgroup2fs`.

Verify the UID/GID on each machine with `id robot-train` and `getent group robotdata`. When the numbers differ, jobs still schedule successfully but fail with permission errors when writing to the NAS — a symptom identical to an NFS misconfiguration and very hard to trace.

The scripts never install or upgrade the NVIDIA driver, and never format a disk.

The role scripts allow Ubuntu 24.04, but the Slurm DEBs in this repository are built for Ubuntu 22.04 Jammy and cannot be assumed to work on 24.04. For 24.04, build packages of the same version against the system ABI separately.

## 0. Prepare the shared configuration

Run on each host:

```bash
cp config/site.env.example config/site.env
editor config/site.env
```

The following fields must be byte-for-byte identical on all hosts:

```text
MANAGEMENT_HOST
MANAGEMENT_IP
NAS_IP
NAS_EXPORT
NAS_MOUNT
GPU_NODE_NAMES
GPU_NODE_IPS
DATA_GROUP
DATA_GID
TRAIN_USER
TRAIN_UID
TRAIN_ENV_ROOT
LEROBOT_GIT_REF
```

For the current four-node site, the values are:

```bash
GPU_NODE_NAMES="mgmt01 gpu01 gpu02 gpu03"
GPU_NODE_IPS="192.168.100.202 192.168.100.215 192.168.100.216 192.168.100.217"
```

The two lists correspond positionally and must have the same length; otherwise `05-configure-hosts.sh` reports `GPU_NODE_NAMES and GPU_NODE_IPS have different lengths`.

Do not commit `config/site.env`. It is the local active configuration and is Git-ignored. **That also means it will not come back on its own after a machine swap or a reinstall** — when the topology changes, update `config/site.env.example` as well.

Set the matching hostname on each host, then log in again. The hostname must exactly match the Slurm NodeName that will be written into `nodes.conf`:

```bash
sudo hostnamectl set-hostname mgmt01   # mgmt01 only
sudo hostnamectl set-hostname gpu01    # gpu01 only; likewise for the other nodes
```

Run on every host:

```bash
sudo ./scripts/05-configure-hosts.sh --apply
getent hosts mgmt01 gpu01 gpu02 gpu03
```

`05` only manages the block in `/etc/hosts` carrying the `robot-platform` marker. It stops in two cases:

- an entry with the same name already exists outside the block → it stops and asks for manual handling;
- the managed block exists but differs from the target (for example a node was added later) → it stops with `existing managed /etc/hosts block differs`.

**The script does not make incremental edits.** When the topology changes, delete the old block first and regenerate it — see [Adding a GPU node to an existing cluster](docs/09-add-gpu-node.md#3-rebuild-the-managed-etchosts-block).

## 1. Prepare the QNAP

Complete [QNAP NAS setup](docs/01-qnap-nas.md) first. At minimum these must exist:

```text
/mnt/robot_platform/datasets
/mnt/robot_platform/jobs
/mnt/robot_platform/mlflow-artifacts
```

The QNAP NFS allow list must contain **every** node IP: `192.168.100.202`, `192.168.100.215`, `192.168.100.216`, `192.168.100.217`. This step is easy to forget when adding a node; the symptom is that the new node fails to mount or mounts read-only. While the pilot uses `all_squash`, give the QNAP guest account read/write permission on the shared directory.

## 2. Install mgmt01

Run the following on `mgmt01` only:

```bash
./scripts/00-audit-host.sh management
sudo ./scripts/10-install-management.sh --apply
sudo ./deploy/management/bootstrap.sh --apply
sudo ./scripts/25-install-training-environment.sh --apply
```

Ubuntu 22.04 ships Python 3.10, while the current LeRobot needs Python 3.12. If the machine has no `python3.12` yet, install it first following [Management node installation](docs/02-management-node.md).

`10` generates the cluster-unique `/etc/munge/munge.key`. Do not generate a second one, and do not put it in Git, on the NAS, or in a chat log.

## 3. Install every GPU worker

Run the following on `gpu01`, `gpu02` and `gpu03` separately:

```bash
./scripts/00-audit-host.sh gpu
sudo ./scripts/20-install-gpu-node.sh --apply
sudo ./scripts/25-install-training-environment.sh --apply
```

Confirm the local directories exist:

```bash
sudo -u robot-train test -d /cache/datasets
sudo -u robot-train test -d /cache/exports
sudo -u robot-train test -d /work/runs
sudo -u robot-train test -w /var/lib/robot-platform/cache
```

The last one is `LELAB_JOB_CACHE_ROOT`, which **must exist on every worker and be writable by `robot-train`**. Slurm points `HOME` at the `robot-train` home directory, which does not exist on the workers, so a job that caches into `~` fails on that node. Create it if missing:

```bash
sudo install -d -o robot-train -g robotdata -m 0750 /var/lib/robot-platform/cache
```

## 4. Install and configure Slurm

The old Slurm in the Ubuntu 22.04 repositories is not suitable as the final version for this project. After the base role scripts finish, install the same Slurm 26.05.2 DEBs on every host — see [Slurm 26.05.2 installation](docs/Slurm-INSTALL.md).

Then complete [Slurm cluster finalization](docs/06-cluster-finalization.md):

1. collect the real `sudo slurmd -C` output on every host;
2. on `mgmt01`, fill in `config/slurm/nodes.conf`, one line per machine;
3. render and review `config/slurm/slurm.conf.generated`;
4. install the controller config on `mgmt01`;
5. securely copy the Munge key and the generated config to every worker;
6. install the worker config on each worker using that machine's real paths;
7. verify that every node is `idle` and run a single-GPU smoke test on each node.

Two details that go wrong repeatedly:

- **`/etc/slurm/slurm.conf` must be byte-for-byte identical across the whole cluster.** When the topology changes, the stale config on existing nodes has to be replaced too; updating only the new node breaks the existing ones.
- Paths like `/secure/temp/munge.key` are documentation placeholders and are not created for you. The first two arguments passed to `install-worker-config.sh` must be **files that already exist on that worker and are readable by root**.

## 5. Install leLab

Once every node shows `idle` in `sinfo -N -l` and the shared training environment passes its GPU test on every machine, run on `mgmt01` only:

```bash
sudo ./scripts/15-install-lelab-platform.sh --apply
```

The installation also requires:

- Python 3.12 on `mgmt01`;
- Node.js 20.19 or newer plus npm for the ordinary user invoking `sudo`;
- `datasets` readable and `jobs` writable on the NAS;
- working Slurm commands.

The installer creates `/etc/robot-platform/lelab.env` on first run and does not overwrite it afterwards. With the current remote SSH users it should be configured as:

```bash
LELAB_CLUSTER_NODES=mgmt01=192.168.100.202,gpu01=snorlax@192.168.100.215,gpu02=yang@192.168.100.216,gpu03=snorlax@192.168.100.217
```

This line gets long. **Edit it in an editor, not with a one-line `sed`**: a long command wraps when pasted into a terminal, and `sed` reports `unterminated 's' command` once it receives an incomplete expression. The file is read by systemd as an `EnvironmentFile`, so after editing it you must run `sudo systemctl restart lelab-platform` for the change to take effect.

For SSH keys, host key verification and check commands, see [leLab cluster web](docs/07-lelab-cluster-web.md).

## 6. Acceptance

Per-role acceptance:

```bash
# mgmt01
./scripts/90-validate-deployment.sh management

# every GPU worker
./scripts/90-validate-deployment.sh gpu
```

Key checks on the management node:

```bash
scontrol ping
sinfo -N -l
scontrol show nodes

curl --noproxy '*' -fsS http://127.0.0.1:8000/health
curl --noproxy '*' -fsS http://127.0.0.1:8000/cluster/status | jq
curl --noproxy '*' -fsS http://127.0.0.1:8000/cluster/templates | jq
```

Final acceptance is not "the service is active", it is:

1. every node can be allocated one GPU by Slurm, and `nvidia-smi -L` returns a **different** GPU UUID on each (a repeated UUID means `NodeAddr` is wrong and two NodeNames point at the same physical machine);
2. leLab sees every node and its VRAM, with `eligible` set to `true`;
3. a small LeRobot dataset on the NAS shows up in the UI;
4. a short training job can be submitted and produces logs and a checkpoint;
5. after an interruption, the job can resume from the shared checkpoint.

## Script behavior and security boundary

- `00-audit-host.sh` and `90-validate-deployment.sh` are read-only checks;
- every other installation script requires an explicit `--apply`;
- the role installers create accounts, directories, systemd services and the NFS mount;
- `15` and `25` reach out to Python/Git/npm package sources and take a while;
- `config/site.env`, `deploy/management/.env`, the Munge key, and the leLab SSH private key and tokens must never be committed;
- after a failure you can fix the cause and re-run the same step, but once the self-built Slurm DEBs are in use, read the version-protection notes in [Slurm installation](docs/Slurm-INSTALL.md) before re-running `10`/`20`.

## Developing LeRobot in the `lerobot/` submodule

`lerobot/` is a Git submodule pointing at [Yingminj/lerobot_dev](https://github.com/Yingminj/lerobot_dev.git), a fork of `huggingface/lerobot`. It is a development checkout for reading and modifying the framework — editing it changes nothing on the cluster until you deploy it. Training jobs run out of `/opt/robot-platform/train-venv`, which `25-install-training-environment.sh` installs from `LEROBOT_GIT_URL@LEROBOT_GIT_REF` and later updates in place from this checkout with `--sync-lerobot`. Keep the submodule on the same commit as `LEROBOT_GIT_REF` (currently `dev`), otherwise you will write code against APIs the deployed environment does not have.

Clone with the submodule, or fill it in afterwards:

```bash
git clone --recurse-submodules https://github.com/Yingminj/robot_data_platform.git
# already cloned without it:
git submodule update --init lerobot
```

Inside the submodule, `origin` is the fork and `upstream` is `huggingface/lerobot`:

```bash
git -C lerobot remote -v
git -C lerobot fetch upstream      # pick up new upstream releases
```

### The parent repository stores a commit ID, not files

`git add lerobot` in the parent stages a single 40-character SHA — the submodule's current `HEAD`. It never stages the contents of files under `lerobot/`, and `git commit` in the parent never pushes anything to `lerobot_dev`. Two repositories, two independent pushes.

Two consequences are worth remembering, because neither announces itself:

- Editing a file under `lerobot/` and then running `git add lerobot && git commit` in the parent records **nothing**. The submodule's `HEAD` has not moved, so the SHA is unchanged and the commit is empty.
- Committing inside the submodule without pushing, then committing the pointer in the parent, produces a parent commit that references a SHA existing only on that one machine. Everyone else — including you on another node — gets `fatal: reference is not a tree` from `git submodule update`.

So always work in this order: contents, publish, pointer.

```bash
# 1. contents, inside the submodule
git -C lerobot add <files>
git -C lerobot commit -m "..."

# 2. publish them (the first push of a new branch needs -u)
git -C lerobot push -u origin dev

# 3. only now record the new pointer in the parent
git add lerobot && git commit -m "bump lerobot to ..."
```

Let Git enforce step 2 instead of remembering it:

```bash
git config --global push.recurseSubmodules check   # or on-demand, to push the submodule automatically
```

### Reading the status output

`git status` in the parent describes the submodule with a one-character code, and the two codes mean very different things:

| Output | Meaning | Does `git add lerobot` record anything? |
|---|---|---|
| `? lerobot` | untracked files inside the submodule | no — `HEAD` has not moved |
| `M lerobot` | the submodule `HEAD` differs from the recorded pointer | yes — the new SHA |

`git submodule status` says the same thing more precisely: a leading `+` means the checkout has moved off the recorded commit, `-` means the submodule is not initialized, and a leading space means it matches.

### Getting a change onto the cluster

A commit in `lerobot_dev` changes nothing on the nodes by itself. `/opt` is **not** shared between nodes, so every node Slurm can schedule needs its own update — list them with `sinfo -N -l`. Skip one and a job that lands there dies on `import`, usually much later than the deployment you thought was finished.

The two roles do different amounts of work: the management node publishes the code and owns the leLab configuration, the workers only pull and update their environment.

#### On the management node (mgmt01)

**1. Publish the code.** Contents, publish, pointer — the order from [The parent repository stores a commit ID, not files](#the-parent-repository-stores-a-commit-id-not-files):

```bash
git -C lerobot add <files>
git -C lerobot commit -m "..."
git -C lerobot push origin dev              # first push of a new branch needs -u
git add lerobot && git commit -m "bump lerobot to ..." && git push
```

**2. Update the training environment.**

```bash
sudo ./scripts/25-install-training-environment.sh --sync-lerobot --apply
```

This rsyncs `lerobot/src/lerobot/` over the installed package (with `--delete`, so a file you removed in the fork stops shadowing the new code), stamps `SUBMODULE_REVISION`, re-checks `import lerobot`, and verifies the venv still satisfies the extras declared in the submodule's `pyproject.toml`.

It copies source files and resolves **no** dependencies. If your change adds a third-party import, the script refuses to call the node ready and prints the exact `pip install` to run; add the dependency to an extra in `pyproject.toml` too, so a fresh install gets it without the manual step.

**3. Register a new policy in leLab** — only when you added a new `policy_type`. Append an entry to `/etc/robot-platform/model-templates.json`, which no script maintains after `15-install-lelab-platform.sh` seeds it on first install:

```json
{
  "id": "act_delta",
  "label": "ACT (relative actions)",
  "policy_type": "act_delta",
  "python_executable": "/opt/robot-platform/train-venv/bin/python",
  "partition": "train",
  "min_gpu_memory_mb": 8000,
  "cpus_per_task": 8,
  "memory_gb": 48,
  "description": "..."
}
```

`id` must equal `policy_type`, and leLab rejects the whole file if any entry breaks that rule or if the JSON is malformed — which takes the model list down for everyone, so check the endpoint afterwards. The file is re-read on every request, so no restart is needed. Mirror the entry into the version-controlled `apps/lelab/config/model-templates.json.example` as well, or a node installed later will not have it.

Two limits are worth knowing before planning an experiment around the web UI: leLab emits a fixed set of policy flags (`--policy.type`, `.device`, `.use_amp`, `.push_to_hub`, `.repo_id` — see `apps/lelab/lelab/train.py`), so a policy variant that needs any other `--policy.*` flag has to be launched with `sbatch` or the CLI; and since `id` must equal `policy_type`, one policy cannot be offered as several presets.

#### On every GPU worker (gpu01, gpu02, …)

Pull, then update the environment. Nothing else — the workers hold no leLab configuration.

```bash
git pull
git submodule update --init --recursive
sudo ./scripts/25-install-training-environment.sh --sync-lerobot --apply
```

`--sync-lerobot` copies from the checkout **on that node**, so the pull is what actually carries your change across; skip it and the script cheerfully re-syncs the old revision. A worker with no checkout can install from GitHub instead:

```bash
sudo ./scripts/25-install-training-environment.sh --apply
```

which rebuilds the venv from `LEROBOT_GIT_URL@LEROBOT_GIT_REF` and re-resolves every dependency — much slower, but it needs no local clone and it does pick up new dependencies on its own.

#### Verify, on each node

```bash
/opt/robot-platform/train-venv/bin/python -c \
  'import lerobot, pathlib; print((pathlib.Path(lerobot.__file__).parent / "SUBMODULE_REVISION").read_text())'
/opt/robot-platform/train-venv/bin/python -c \
  'from lerobot.policies.factory import get_policy_class; print(get_policy_class("<policy_type>"))'
```

The first prints the revision the installed files actually came from — pip's own metadata still describes whatever it last installed, so it is not evidence. The second fails loudly when a new policy is not registered, which is otherwise only discovered by a training job that dies at startup.

#### Two alternatives to this workflow

- **New policy, no fork — use a plugin.** Package it as a `lerobot_policy_<name>` distribution and `pip install -e` it into each node's `train-venv`. `lerobot-train` auto-imports every installed distribution with that prefix, so edits take effect with no reinstall and no fork of LeRobot at all. It still needs the leLab template entry from step 3.
- **Framework change — re-deploy the pin.** Tag the fork, point `LEROBOT_GIT_URL` and `LEROBOT_GIT_REF` in `config/site.env` at that tag, and re-run `25-install-training-environment.sh --apply` on every node. Move the submodule pointer to the same tag so the checkout and the deployed environment stay in agreement.

## Not included yet

The repository does not yet contain a complete data governance workflow:

- the H5 validator and the automatic QC worker;
- the time-range annotation frontend;
- a runnable upload agent;
- production training data and team-specific models.

The collection-related scripts `30`/`40` only prepare accounts, directories and systemd templates; they do not mean the collection pipeline is ready for use.

## Acknowledgements

This platform is built on the following open source projects, with thanks:

- [huggingface/leLab](https://github.com/huggingface/leLab) — the web interface in `apps/lelab` comes from this project (Apache-2.0). This repository adds Slurm cluster scheduling, multi-node GPU probing and NAS-shared datasets on top of it; the original license and copyright notice are kept in `apps/lelab/LICENSE`.
- [huggingface/lerobot](https://github.com/huggingface/lerobot) — the underlying framework for training, inference and the LeRobot v3 dataset format (Apache-2.0), which is what the conversions in `tool/` target.
- [SchedMD/slurm](https://github.com/SchedMD/slurm) — cluster job scheduling (GPL-2.0).
- [mlflow/mlflow](https://github.com/mlflow/mlflow) — training metrics and artifact tracking (Apache-2.0).
- [ros2/rosbag2](https://github.com/ros2/rosbag2) — the recording format on the collection side and the input to the conversions in `tool/`.

Model and dataset names visible in the screenshots are local pilot data and are not part of the projects above.
