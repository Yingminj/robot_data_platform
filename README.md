# 机器人数据平台部署包

本目录提供 QNAP NAS、管理机、GPU 节点、采集节点以及 leLab 集群 Web 的安装说明和基础设施脚本。目标环境为 Ubuntu 22.04/24.04、QNAP NFSv4、1 台“管理 + GPU”节点和 4 台 GPU 节点，共 5 张可调度 GPU。

已知站点参数：

| 角色 | 地址 |
|---|---|
| 管理机兼 GPU 节点 `mgmt01` | `192.168.100.202` |
| QNAP NAS | `192.168.100.184:/kmd_data_file` |
| 其余 GPU 节点 | `.123`、`.206`、`.208`、`.209` |
| 统一挂载点 | `/mnt/robot_platform` |

## 重要边界

本部署包可以安装和配置：

- NFS 客户端和持久挂载；
- Docker、PostgreSQL、Redis 和 MLflow；
- Munge、Slurm Controller、Slurm Worker；
- GPU 节点缓存和任务目录；
- 基于 `Yingminj/leLab` fork 的集群 Web、NAS 数据集浏览、GPU 探测和 Slurm Runner；
- 采集节点服务账号和本地 Spool；
- 环境审计及部署后验证。

当前工作区仍没有以下数据采集/治理业务源码：

- H5 validator 和自动 QC Worker；
- 关键动作时间范围标注前端；
- 采集端 Upload Agent；
- 正式训练数据和团队定制模型代码。

集群训练 Web 源码直接纳入本仓库的 `apps/lelab/`，与部署脚本、配置和文档使用同一个版本提交，避免部署代码与 Web 功能版本错配。

## 安全约定

所有安装脚本默认不执行修改，必须显式传入 `--apply`。执行前先阅读脚本并确认 `config/site.env`。脚本不会自动安装或升级 NVIDIA 驱动，也不会自动格式化磁盘。

不要提交以下文件：

- `config/site.env` 中的站点私有调整；
- `deploy/management/.env` 中的数据库密码；
- Munge 密钥、上传令牌和任何访问 Token。

## 目录

```text
config/
  site.env.example              统一站点参数
  lelab.env.example             leLab 集群运行参数
  slurm/                        Slurm 模板和节点资源清单
deploy/
  management/                   PostgreSQL、Redis、MLflow Compose
  systemd/                      leLab 与 Upload Agent systemd 模板
docs/
  01-qnap-nas.md
  02-management-node.md
  03-gpu-node.md
  04-collector-node.md
  05-combined-node.md
  06-cluster-finalization.md
scripts/
  00-audit-host.sh              只读环境检查
  05-configure-hosts.sh         安装统一主机名解析
  10-install-management.sh
  15-install-lelab-platform.sh
  20-install-gpu-node.sh
  25-install-training-environment.sh
  30-install-collector.sh
  40-install-combined-node.sh
  50-configure-nfs-mount.sh
  90-validate-deployment.sh
  cluster/                      Slurm 渲染和安装脚本
```

## 总体部署顺序

### 1. 准备统一配置

在需要执行脚本的每台 Linux 主机上复制本目录，然后：

```bash
cp config/site.env.example config/site.env
editor config/site.env
```

重点确认：

- 管理机和 NAS 地址；
- 5 个 GPU Worker（包含管理机）的地址和主机名；
- `robot-train`/`robotdata` 的 UID/GID 在五个 Slurm Worker 上没有冲突；它们仅用于 Slurm 身份一致性，不用于 QNAP 权限管理；
- 本地缓存、Spool 和工作目录位于预期磁盘。

所有 Linux 主机确认配置后可安装统一的静态主机名解析：

```bash
sudo ./scripts/05-configure-hosts.sh --apply
```

### 2. 配置 QNAP

先按照 [QNAP NAS 配置](docs/01-qnap-nas.md)创建目录，并允许 5 台主机通过 NFS 读写。试点阶段不要求 QNAP 按 Linux 数字 UID/GID 建立 ACL，也不配置成员级权限。

### 3. 安装管理机兼 GPU 节点

```bash
./scripts/00-audit-host.sh management
sudo ./scripts/10-install-management.sh --apply
sudo ./deploy/management/bootstrap.sh --apply
```

该主机也必须执行训练环境安装：

```bash
sudo ./scripts/25-install-training-environment.sh --apply
```

详细步骤见[管理机安装](docs/02-management-node.md)。

### 4. 安装 GPU 节点

每台 GPU 主机执行：

```bash
./scripts/00-audit-host.sh gpu
sudo ./scripts/20-install-gpu-node.sh --apply
sudo ./scripts/25-install-training-environment.sh --apply
```

然后在包括管理机在内的 5 台主机收集 `slurmd -C`，生成全节点 `slurm.conf`。详细步骤见 [GPU 节点安装](docs/03-gpu-node.md)和[集群收尾](docs/06-cluster-finalization.md)。

### 5. 安装 leLab 集群 Web

完成 Slurm 后，在管理机执行：

```bash
sudo ./scripts/15-install-lelab-platform.sh --apply
```

Web 默认监听 `http://192.168.100.202:8000`。详细配置见 [leLab 集群 Web](docs/07-lelab-cluster-web.md)。

### 6. 安装采集节点

```bash
./scripts/00-audit-host.sh collector
sudo ./scripts/30-install-collector.sh --apply
```

采集节点不应直接获得 NAS `raw` 写权限。详细步骤见[采集节点安装](docs/04-collector-node.md)。

### 7. 合并节点

同一台机器同时采集和训练时：

```bash
./scripts/00-audit-host.sh combined
sudo ./scripts/40-install-combined-node.sh --apply
```

还必须配置资源隔离和采集窗口，见[采集与 GPU 合并节点](docs/05-combined-node.md)。

### 8. 验收

```bash
./scripts/90-validate-deployment.sh management
./scripts/90-validate-deployment.sh gpu
./scripts/90-validate-deployment.sh collector
./scripts/90-validate-deployment.sh combined
```

第一阶段验收以“NAS 选数据 → 选择登记模型模板 → 自动选择空闲 GPU → Slurm 单机单卡训练 → checkpoint 中断续训”的真实短任务闭环为准；单纯“服务为 active”不等于平台完成。上传、QC、标注和 DatasetVersion 属于后续数据治理阶段。

## 当前环境注意事项

- `.209` 在前次检查中网络不可达，恢复前无法完成 5 节点验收。
- QNAP 当前导出白名单缺少 `.123`，必须补充。
- 现有 `/home/kewei/NAS` 是手工挂载；应迁移为统一的 `/mnt/robot_platform` 自动挂载。
- 当前阶段 NAS 权限以五台主机可读写为目标；进入正式数据治理阶段后再收紧。
- 管理机 Docker 镜像和构建缓存占用较大，正式构建前应为 Docker/数据库规划独立 SSD 空间。
