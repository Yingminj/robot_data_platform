# Management node `mgmt01` installation

**English** | [简体中文](02-management-node.zh-CN.md)

`mgmt01` runs PostgreSQL, Redis, MLflow, the Slurm controller, a Slurm worker and leLab all at once, and contributes one RTX 4090 to Slurm. Unless stated otherwise, every command in this document is run from the repository root on `mgmt01`.

Adding a GPU node to an already running cluster does not require redoing this document — see [Adding a GPU node to an existing cluster](09-add-gpu-node.md).

## 1. Pre-installation checks

Confirm the host identity and the site configuration:

```bash
hostname -s
ip -br address
cp config/site.env.example config/site.env
editor config/site.env
```

Expected:

```text
hostname: mgmt01
MANAGEMENT_IP=192.168.100.202
GPU_NODE_NAMES="mgmt01 gpu01 gpu02 gpu03"
GPU_NODE_IPS="192.168.100.202 192.168.100.215 192.168.100.216 192.168.100.217"
```

The two lists correspond positionally, must have the same length, and **must have exactly the same values on every host**.

If the hostname needs to change:

```bash
sudo hostnamectl set-hostname mgmt01
```

Log in again afterwards, then run:

```bash
sudo ./scripts/05-configure-hosts.sh --apply
./scripts/00-audit-host.sh management
nvidia-smi
timedatectl show --property=NTPSynchronized --value
```

Before continuing, also confirm:

- the QNAP allows NFS access from **all** node IPs;
- there is enough disk space for `/`, Docker and PostgreSQL;
- the NVIDIA driver works;
- `DATA_GID` and `TRAIN_UID` are not already taken by other accounts;
- every host uses the same cluster fields in `config/site.env`.

## 2. Prepare Python 3.12 and the frontend toolchain

The current LeRobot and leLab use Python 3.12. Check first:

```bash
python3.12 --version
```

Ubuntu 24.04 can use the system package directly. When Ubuntu 22.04 does not have the command, install it from a team-approved package source; the current environment uses:

```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-dev python3.12-venv
```

The leLab frontend requires Node.js 20.19 or newer and npm. They must be installed in the environment of the ordinary user who runs `sudo ./scripts/15-...`:

```bash
node --version
npm --version
```

If you use nvm, the installer sources that user's `~/.nvm/nvm.sh` when it drops privileges to build the frontend. Do not install Node.js for root only.

## 3. Install the management base components

```bash
sudo ./scripts/10-install-management.sh --apply
```

The script:

- installs NFS, Chrony, Munge, SSH, Docker and the NVIDIA Container Toolkit;
- creates `robotdata`, `robot-ingest` and `robot-train`;
- creates the cache, runtime and Slurm state directories;
- mounts the QNAP persistently at `/mnt/robot_platform`;
- generates `/etc/munge/munge.key` on first run;
- prepares the machine for the Slurm controller and the local worker.

The script does not touch the NVIDIA driver. After the first run, check:

```bash
findmnt /mnt/robot_platform
getent passwd robot-train
getent group robotdata
sudo -u robot-train test -r /mnt/robot_platform/datasets
sudo -u robot-train test -w /mnt/robot_platform/jobs
```

`/etc/munge/munge.key` is the single authentication key for the whole cluster. Only ever copy this one securely to the workers; do not generate a new one on `gpu01`.

### Slurm version ordering

The `slurm-wlm` from Ubuntu 22.04 only exists so the role scripts can finish the base preparation; everything should end up on Slurm 26.05.2, which supports the current cgroup v2 configuration. The recommended order is:

```text
run the 10/20 base role scripts first
→ install the same Slurm 26.05.2 DEBs on every host
→ install the controller/worker configs last
```

Once the self-built DEBs are in use, do not casually reinstall Ubuntu's `slurm-wlm`, or 26.05.2 gets replaced by the old version. See [Slurm 26.05.2 installation](Slurm-INSTALL.md).

## 4. Start PostgreSQL, Redis and MLflow

First confirm the NAS directory exists and is writable by `robot-ingest`:

```bash
sudo -u robot-ingest test -w /mnt/robot_platform/mlflow-artifacts
```

Then run:

```bash
sudo ./deploy/management/bootstrap.sh --apply
sudo docker compose \
  --env-file deploy/management/.env \
  -f deploy/management/compose.yaml \
  ps
curl --noproxy '*' -fsS http://127.0.0.1:5000/health
```

Data locations:

| Content | Location |
|---|---|
| PostgreSQL | `/var/lib/robot-platform/postgres`, local SSD |
| MLflow artifacts | `/mnt/robot_platform/mlflow-artifacts`, NAS |
| Database password | `deploy/management/.env`, mode `0600` |

Do not put the PostgreSQL data directory on NFS, and do not commit `.env`.

## 5. Install the shared training environment

```bash
sudo ./scripts/25-install-training-environment.sh --apply
sudo -u robot-train /opt/robot-platform/train-venv/bin/python -c \
  'import torch, lerobot; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

`torch.cuda.is_available()` is expected to be `True`. The `LEROBOT_GIT_REF`, the training environment path and the Python major version must be identical on **all** workers.

## 6. Finish Slurm

Do not install leLab yet. Complete Slurm following these documents:

1. install [Slurm 26.05.2](Slurm-INSTALL.md) on every host;
2. render the config and distribute the Munge key following [Slurm cluster finalization](06-cluster-finalization.md);
3. run the GPU smoke test from `mgmt01` against each node separately.

In the end the management node should have all of these working:

```bash
systemctl is-active munge slurmctld slurmd
scontrol ping
sinfo -N -l
```

## 7. Install leLab

Only run this once every node is `idle` and the GPU smoke test has succeeded:

```bash
bash -n scripts/15-install-lelab-platform.sh
sudo ./scripts/15-install-lelab-platform.sh --apply
```

The script:

- builds the React frontend as the ordinary user who invoked sudo;
- installs `/opt/robot-platform/lelab`;
- creates `/opt/robot-platform/lelab-venv`;
- creates `/etc/robot-platform/lelab.env` and the model templates on first run;
- starts `lelab-platform.service`.

The installer does not overwrite an existing `/etc/robot-platform/lelab.env`. The SSH probing configuration must be done separately, following [leLab cluster web](07-lelab-cluster-web.md).

Check:

```bash
systemctl is-active lelab-platform
curl --noproxy '*' -fsS http://127.0.0.1:8000/health
curl --noproxy '*' -fsS http://127.0.0.1:8000/cluster/status | jq
```

## 8. Final management node acceptance

```bash
./scripts/90-validate-deployment.sh management
```

If `http_proxy`/`https_proxy` are set on this machine, always use `curl --noproxy '*'` when reaching local services; otherwise even a `127.0.0.1` request can be sent to the proxy and come back as 502.

## 9. Ports and backups

The current cluster needs at least:

| Port | Source | Destination |
|---|---|---|
| TCP 22 | administrators, leLab on `mgmt01` | all hosts |
| TCP 2049 | all hosts | QNAP |
| TCP 6817 | all workers | `mgmt01` |
| TCP 6818 | `mgmt01` | all workers |
| TCP 8000 | team intranet | `mgmt01` |
| TCP 5000 | intranet during the pilot | `mgmt01` |

When adding a node, all of these rules must cover the new node, and its IP must also be added to the QNAP NFS allow list.

PostgreSQL 5432, Redis 6379 and the Docker TCP API should not be exposed to the intranet.

Back up at least:

- a daily logical backup of PostgreSQL;
- `deploy/management/.env` into a controlled password vault;
- the configuration in `/etc/slurm` and `/etc/robot-platform` that contains no private keys;
- the current repository commit, the Slurm package version and `LEROBOT_GIT_REF`.
