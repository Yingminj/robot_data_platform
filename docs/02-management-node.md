# 管理机安装

管理机 `192.168.100.202` 同时运行平台控制面和第 5 个单 GPU Slurm Worker。它必须使用静态 IP、持续开机、禁止自动休眠，建议接 UPS。

## 1. 前置检查

```bash
cp config/site.env.example config/site.env
editor config/site.env
./scripts/00-audit-host.sh management
```

确认：

- `/`、Docker 和 PostgreSQL 数据盘空间满足预期；
- QNAP 已建立平台目录并允许 5 台主机 NFS 读写；
- 管理机能访问 NAS TCP 2049；
- 主机名准备设为 `mgmt01`；
- NVIDIA Driver 和 `nvidia-smi` 正常。

如需修改主机名：

```bash
sudo hostnamectl set-hostname mgmt01
```

## 2. 安装宿主机组件

```bash
sudo ./scripts/10-install-management.sh --apply
```

脚本将：

- 安装 NFS、Chrony、Munge 和 Slurm；
- 安装 NVIDIA Container Toolkit，并配置本机 `slurmd` 所需目录；
- 检查或安装 Docker/Compose；
- 创建服务账号和本地状态目录；
- 将 NAS 持久挂载到 `/mnt/robot_platform`；
- 生成管理机 Munge 密钥；
- 暂停 Slurm Controller/Worker，等待 5 台真实节点配置。

脚本不会修改 NVIDIA 驱动，也不会清理已有 Docker 镜像。

## 3. 启动 PostgreSQL、Redis 和 MLflow

确认以下目录已经在 NAS 上创建，并允许平台服务写入：

```text
/mnt/robot_platform/mlflow-artifacts
```

然后执行：

```bash
sudo ./deploy/management/bootstrap.sh --apply
```

生成的数据库密码位于：

```text
deploy/management/.env
```

该文件权限为 `0600`，不要提交或通过聊天工具发送。基础服务检查：

```bash
cd deploy/management
sudo docker compose ps
curl -fsS http://192.168.100.202:5000/health
```

PostgreSQL 数据保存在管理机本地 SSD：

```text
/var/lib/robot-platform/postgres
```

MLflow artifacts 保存在 NAS。不要将 PostgreSQL 数据目录改到 NFS。

## 4. 安装统一训练环境

管理机也参与训练，因此执行：

```bash
sudo ./scripts/25-install-training-environment.sh --apply
```

该脚本一次性把固定版本 LeRobot 安装到 `/opt/robot-platform/train-venv`。正式任务不再逐次配置 Python 环境。

## 5. 安装 leLab 集群 Web

完成 5 节点 Slurm 配置后：

```bash
sudo ./scripts/15-install-lelab-platform.sh --apply
curl -fsS http://192.168.100.202:8000/health
```

安装脚本要求 Python 3.12 和 Node.js 20.19 或更高版本；Ubuntu 22.04 若没有 Python 3.12，应先通过团队认可的软件源统一安装，避免用系统 Python 版本硬凑依赖。

详细说明见 [leLab 集群 Web](07-lelab-cluster-web.md)。

## 6. 其他业务应用接入点

leLab 已提供训练 Web/API。数据采集与治理功能仍可在后续加入：

- `upload-worker`：H5、manifest、SHA-256、幂等和 quarantine；
- `qc-worker`：自动质检和预览生成；
- `annotation`：时间范围标注与审核；
- `reverse-proxy`：统一 HTTPS 入口。

PostgreSQL 内已经预留 `platform` 数据库，Redis 可作为异步任务队列。正式上线前 MLflow 也应放在 HTTPS 反向代理后，不能长期以裸 `5000` 端口运行。

## 7. 备份

至少安排以下任务：

- 每日 PostgreSQL 逻辑备份到 NAS；
- 定期验证备份可恢复到一个新实例；
- 备份 `deploy/management/.env` 到受控密码库，而不是公开 NAS 目录；
- 保存 Slurm 配置、应用版本和 `LEROBOT_GIT_REF`。

## 8. 防火墙原则

仅开放：

- TCP 22：管理员来源；
- TCP 443：小组用户和采集节点；
- TCP 6817：4 台 GPU 节点；
- TCP 6818：管理机到 4 台远程 Worker；
- TCP 8000：小组内网访问 leLab；
- TCP 5000：仅试点期内网访问，正式上线由 443 代理。

PostgreSQL 5432、Redis 6379 和 Docker API 不对内网开放。本部署包不会自动修改防火墙，避免切断现有远程连接。
