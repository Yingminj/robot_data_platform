# 管理机安装

管理机 `192.168.100.202` 运行平台控制面。它必须使用静态 IP、持续开机、禁止自动休眠，建议接 UPS。

## 1. 前置检查

```bash
cp config/site.env.example config/site.env
editor config/site.env
./scripts/00-audit-host.sh management
```

确认：

- `/`、Docker 和 PostgreSQL 数据盘空间满足预期；
- QNAP 已建立平台目录和 ACL；
- 管理机能访问 NAS TCP 2049；
- 主机名准备设为 `mgmt01`；
- `DATA_GID` 和服务 UID 没有冲突。

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
- 检查或安装 Docker/Compose；
- 创建服务账号和本地状态目录；
- 将 NAS 持久挂载到 `/mnt/robot-platform`；
- 生成管理机 Munge 密钥；
- 暂停 Slurm Controller，等待真实节点配置。

脚本不会修改 NVIDIA 驱动，也不会清理已有 Docker 镜像。

## 3. 启动 PostgreSQL、Redis 和 MLflow

确认以下目录已经由 QNAP 管理员创建，并对 `robot-ingest` 数字 UID 可写：

```text
/mnt/robot-platform/robot-platform/mlflow-artifacts
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

## 4. 业务应用接入点

获得平台源码后，在 Compose 中加入：

- `platform-api`：用户、Episode、DatasetVersion 和权限 API；
- `upload-worker`：H5、manifest、SHA-256、幂等和 quarantine；
- `qc-worker`：自动质检和预览生成；
- `web`：数据目录和管理页面；
- `annotation`：时间范围标注与审核；
- `reverse-proxy`：统一 HTTPS 入口。

PostgreSQL 内已经预留 `platform` 数据库，Redis 可作为异步任务队列。正式上线前 MLflow 也应放在 HTTPS 反向代理后，不能长期以裸 `5000` 端口运行。

## 5. 备份

至少安排以下任务：

- 每日 PostgreSQL 逻辑备份到 NAS；
- 定期验证备份可恢复到一个新实例；
- 备份 `deploy/management/.env` 到受控密码库，而不是公开 NAS 目录；
- 保存 Slurm 配置、应用版本和容器 digest。

## 6. 防火墙原则

仅开放：

- TCP 22：管理员来源；
- TCP 443：小组用户和采集节点；
- TCP 6817：4 台 GPU 节点；
- TCP 5000：仅试点期内网访问，正式上线由 443 代理。

PostgreSQL 5432、Redis 6379 和 Docker API 不对内网开放。本部署包不会自动修改防火墙，避免切断现有远程连接。

