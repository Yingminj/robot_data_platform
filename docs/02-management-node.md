# 管理机 `mgmt01` 安装

`mgmt01` 同时运行 PostgreSQL、Redis、MLflow、Slurm Controller、Slurm Worker 和 leLab，并提供一张 RTX 4090 给 Slurm。本文中的命令除特别说明外，都在 `mgmt01` 的仓库根目录执行。

向已经运行的集群加 GPU 节点不需要重做本文，见[向已有集群增加 GPU 节点](09-add-gpu-node.md)。

## 1. 安装前检查

确认主机身份和站点配置：

```bash
hostname -s
ip -br address
cp config/site.env.example config/site.env
editor config/site.env
```

期望：

```text
hostname: mgmt01
MANAGEMENT_IP=192.168.100.202
GPU_NODE_NAMES="mgmt01 gpu01 gpu02 gpu03"
GPU_NODE_IPS="192.168.100.202 192.168.100.215 192.168.100.216 192.168.100.217"
```

两个列表按位置一一对应，长度必须相同，且**在所有主机上取值完全一致**。

如果需要修改主机名：

```bash
sudo hostnamectl set-hostname mgmt01
```

修改后重新登录，再执行：

```bash
sudo ./scripts/05-configure-hosts.sh --apply
./scripts/00-audit-host.sh management
nvidia-smi
timedatectl show --property=NTPSynchronized --value
```

在继续前还要确认：

- QNAP 已允许**全部**节点 IP 访问 NFS；
- `/`、Docker 和 PostgreSQL 所在磁盘空间足够；
- NVIDIA 驱动正常；
- `DATA_GID` 和 `TRAIN_UID` 未被其他账号占用；
- 所有主机使用相同的 `config/site.env` 集群字段。

## 2. 准备 Python 3.12 和前端工具链

当前 LeRobot 和 leLab 使用 Python 3.12。先检查：

```bash
python3.12 --version
```

Ubuntu 24.04 可直接使用系统包。Ubuntu 22.04 没有该命令时，使用团队批准的软件源安装；当前环境采用：

```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-dev python3.12-venv
```

leLab 前端要求 Node.js 20.19 或更高版本和 npm。它们应安装在执行 `sudo ./scripts/15-...` 的普通用户环境中：

```bash
node --version
npm --version
```

如果使用 nvm，安装脚本会在降权构建前端时加载该用户的 `~/.nvm/nvm.sh`。不要只给 root 安装 Node.js。

## 3. 安装管理机基础组件

```bash
sudo ./scripts/10-install-management.sh --apply
```

该脚本会：

- 安装 NFS、Chrony、Munge、SSH、Docker 和 NVIDIA Container Toolkit；
- 创建 `robotdata`、`robot-ingest`、`robot-train`；
- 创建缓存、运行和 Slurm 状态目录；
- 将 QNAP 持久挂载到 `/mnt/robot_platform`；
- 首次生成 `/etc/munge/munge.key`；
- 为后续 Slurm Controller 和本机 Worker 做准备。

脚本不会修改 NVIDIA 驱动。第一次运行后检查：

```bash
findmnt /mnt/robot_platform
getent passwd robot-train
getent group robotdata
sudo -u robot-train test -r /mnt/robot_platform/datasets
sudo -u robot-train test -w /mnt/robot_platform/jobs
```

`/etc/munge/munge.key` 是整个集群唯一的认证密钥。后续只把这一份安全复制到 Worker，不要在 `gpu01` 重新生成。

### Slurm 版本顺序

Ubuntu 22.04 的 `slurm-wlm` 只用于让角色脚本完成基础准备，最终应统一升级为支持当前 cgroup v2 配置的 Slurm 26.05.2。推荐顺序是：

```text
先运行 10/20 基础角色脚本
→ 所有主机安装相同的 Slurm 26.05.2 DEB
→ 最后安装 Controller/Worker 配置
```

使用自建 DEB 后不要随意重新安装 Ubuntu 的 `slurm-wlm`，避免把 26.05.2 替换回旧版。详见 [Slurm 26.05.2 安装](Slurm-INSTALL.md)。

## 4. 启动 PostgreSQL、Redis 和 MLflow

先确认 NAS 目录存在且 `robot-ingest` 可写：

```bash
sudo -u robot-ingest test -w /mnt/robot_platform/mlflow-artifacts
```

然后执行：

```bash
sudo ./deploy/management/bootstrap.sh --apply
sudo docker compose \
  --env-file deploy/management/.env \
  -f deploy/management/compose.yaml \
  ps
curl --noproxy '*' -fsS http://127.0.0.1:5000/health
```

数据位置：

| 内容 | 位置 |
|---|---|
| PostgreSQL | `/var/lib/robot-platform/postgres`，本地 SSD |
| MLflow artifacts | `/mnt/robot_platform/mlflow-artifacts`，NAS |
| 数据库密码 | `deploy/management/.env`，权限 `0600` |

不要把 PostgreSQL 数据目录放到 NFS，也不要提交 `.env`。

## 5. 安装统一训练环境

```bash
sudo ./scripts/25-install-training-environment.sh --apply
sudo -u robot-train /opt/robot-platform/train-venv/bin/python -c \
  'import torch, lerobot; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

期望 `torch.cuda.is_available()` 为 `True`。**所有** Worker 的 `LEROBOT_GIT_REF`、训练环境路径和 Python 主版本必须一致。

## 6. 完成 Slurm

此时先不要安装 leLab。按以下文档完成 Slurm：

1. 所有主机安装 [Slurm 26.05.2](Slurm-INSTALL.md)；
2. 按 [Slurm 集群收尾](06-cluster-finalization.md)渲染配置、分发 Munge 密钥；
3. 从 `mgmt01` 对每个节点分别运行 GPU smoke test。

管理机最终应同时运行：

```bash
systemctl is-active munge slurmctld slurmd
scontrol ping
sinfo -N -l
```

## 7. 安装 leLab

只有在全部节点均为 `idle` 且 GPU smoke test 成功后执行：

```bash
bash -n scripts/15-install-lelab-platform.sh
sudo ./scripts/15-install-lelab-platform.sh --apply
```

脚本会：

- 以发起 sudo 的普通用户构建 React 前端；
- 安装 `/opt/robot-platform/lelab`；
- 创建 `/opt/robot-platform/lelab-venv`；
- 首次创建 `/etc/robot-platform/lelab.env` 和模型模板；
- 启动 `lelab-platform.service`。

安装脚本不会覆盖已经存在的 `/etc/robot-platform/lelab.env`。SSH 探测配置必须按 [leLab 集群 Web](07-lelab-cluster-web.md)单独完成。

检查：

```bash
systemctl is-active lelab-platform
curl --noproxy '*' -fsS http://127.0.0.1:8000/health
curl --noproxy '*' -fsS http://127.0.0.1:8000/cluster/status | jq
```

## 8. 管理机最终验收

```bash
./scripts/90-validate-deployment.sh management
```

若本机设置了 `http_proxy`/`https_proxy`，访问本机服务时统一使用 `curl --noproxy '*'`，否则本地 `127.0.0.1` 请求也可能被送到代理并返回 502。

## 9. 端口和备份

当前集群最少需要：

| 端口 | 来源 | 目标 |
|---|---|---|
| TCP 22 | 管理员、`mgmt01` leLab | 所有主机 |
| TCP 2049 | 所有主机 | QNAP |
| TCP 6817 | 所有 Worker | `mgmt01` |
| TCP 6818 | `mgmt01` | 所有 Worker |
| TCP 8000 | 小组内网 | `mgmt01` |
| TCP 5000 | 试点期内网 | `mgmt01` |

加节点时这几条都要覆盖新节点，QNAP 的 NFS 白名单也要加上它的 IP。

PostgreSQL 5432、Redis 6379 和 Docker TCP API 不应对内网开放。

至少备份：

- PostgreSQL 每日逻辑备份；
- `deploy/management/.env` 到受控密码库；
- `/etc/slurm`、`/etc/robot-platform` 中不含私钥的配置；
- 当前仓库提交号、Slurm 包版本和 `LEROBOT_GIT_REF`。
