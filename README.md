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
  07-lelab-cluster-web.md
scripts/
  00-audit-host.sh              只读环境检查（安装前运行，不修改系统）
  05-configure-hosts.sh         安装统一 /etc/hosts 主机名解析
  10-install-management.sh      管理机基础组件（Slurm/Munge/Docker/NFS/账号）
  15-install-lelab-platform.sh  leLab 集群 Web（须在 Slurm 集群就绪后）
  20-install-gpu-node.sh        GPU Worker 基础组件
  25-install-training-environment.sh  共享训练 venv（LeRobot + PyTorch CUDA）
  30-install-collector.sh       采集节点账号、Spool 目录和 systemd 模板
  40-install-combined-node.sh   合并节点（内部依次调用 20 + 30）
  50-configure-nfs-mount.sh     NAS fstab 持久挂载（由 10/20 自动调用，勿单独运行）
  90-validate-deployment.sh     部署后逐项验收（只读）
  cluster/                      Slurm 配置渲染和安装脚本
```

## 总体部署顺序

### 脚本编号规则与依赖关系

`scripts/` 下脚本的编号即执行顺序。两类脚本不修改系统、可随时运行：`00-audit-host.sh`（安装前只读检查）和 `90-validate-deployment.sh`（安装后只读验收）。其余均为安装脚本，必须 `sudo` 并显式传 `--apply`，不带 `--apply` 时仅打印提示并以退出码 2 结束，不做任何修改。

整体依赖关系如下：

```text
config/site.env（每台主机）
   └─ 05-configure-hosts.sh            所有 Linux 主机，统一主机名解析
        └─ QNAP 导出就绪（docs/01）     必须先于 10/20，否则 NFS 挂载失败
             ├─ 管理机：00 audit → 10 → deploy/management/bootstrap.sh → 25
             ├─ GPU 节点：00 audit → 20 → 25
             │    └─ 5 台就绪后 cluster/ 渲染并安装 slurm.conf（集群收尾）
             │         └─ 15-install-lelab-platform.sh（管理机，须最后装）
             ├─ 采集节点：00 audit → 30
             ├─ 合并节点：00 audit → 40（= 20 + 30）
             └─ 90-validate-deployment.sh <role>  逐角色验收
```

关键顺序约束（违反会直接报错或装出不可用的系统）：

- **QNAP 必须先于 10/20**：`10`/`20` 内部会自动调用 `50-configure-nfs-mount.sh rw --apply` 并验证挂载可读，QNAP 导出白名单未包含本机时在此失败；
- **`25` 必须在同节点的 `10`/`20` 之后**：训练账号 `robot-train`（`TRAIN_USER`）由 `10`/`20` 创建，`25` 安装完 venv 后要 `chown` 给该账号；
- **`15` 的编号虽小于 `20`/`25`，实际执行在最后**：它依赖 `sbatch`/`squeue`/`scontrol`/`sinfo` 等 Slurm 命令和已渲染的集群配置，必须在集群收尾完成后、且只在管理机上运行；
- **`bootstrap.sh` 必须在 `10` 之后**：它要求 Docker 已就绪；
- **`40` 不单独做基础安装**：它内部依次调用 `20` 和 `30`，因此合并节点只需跑 `40` 再加资源隔离配置。

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

注意事项：

- 缺少 `config/site.env` 时所有脚本直接退出；`MANAGEMENT_IP`、`NAS_IP`、`TRAIN_UID` 等必填项为空也会立即报错；
- `TRAIN_UID`/`DATA_GID` 若与现有账号冲突，安装脚本会拒绝执行（不会抢占已占用的 UID/GID），需在全部节点统一调整后重跑；
- `TRAIN_PYTHON_BIN` 默认为 `python3.12`：LeRobot v0.6.0 要求 Python ≥ 3.12，而 Ubuntu 22.04 自带 python3 是 3.10，22.04 主机需先自行安装 Python 3.12；
- 训练环境默认使用清华 PyPI 镜像（`PIP_INDEX_URL`）和 gh-proxy 的 LeRobot Git 镜像（`LEROBOT_GIT_URL`），离线或直连环境请按实际情况改写。

所有 Linux 主机确认配置后可安装统一的静态主机名解析：

```bash
sudo ./scripts/05-configure-hosts.sh --apply
```

注意事项：

- 该脚本只管理 `/etc/hosts` 中带 `BEGIN/END robot-platform` 标记的块；若已有标记块内容与期望不一致，脚本会报错退出并要求人工核对，不会自动覆盖；
- 必须在安装任何角色组件之前完成，Munge/Slurm 的节点间认证依赖主机名解析一致。

### 2. 配置 QNAP

先按照 [QNAP NAS 配置](docs/01-qnap-nas.md)创建目录，并允许 5 台主机通过 NFS 读写。试点阶段不要求 QNAP 按 Linux 数字 UID/GID 建立 ACL，也不配置成员级权限。

注意事项：

- 导出白名单必须包含全部 5 台主机（当前缺少 `.123`，需先补充），否则第 3/4 步在挂载校验处失败；
- 当前导出为 all_squash（全部映射为 guest），骨架目录只需 QNAP guest 账号可写；脚本发现 rw 挂载但 root 不可写时会给出警告，提示检查 QNAP 的 squash/ guest 映射设置。

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

注意事项：

- 脚本不会安装或升级 NVIDIA 驱动，必须先自行装好并确认 `nvidia-smi` 可用，否则 `10`/`20`/`25` 都会在中途退出；
- `10` 会自动创建 `robot-train`/`robot-ingest` 账号、缓存目录，并以 `rw` 模式完成 NAS 持久挂载（写入 `/etc/fstab`，修改前自动备份为带时间戳的 `.bak`）；若 `NAS_MOUNT` 已被其他来源挂载，脚本报错退出；
- Docker 缺失时默认自动安装 Ubuntu 仓库的 `docker.io` + Compose 插件；如需使用预装 Docker Engine，在 `site.env` 设 `ALLOW_DOCKER_INSTALL=0`；
- `25` 会在 `TRAIN_ENV_ROOT`（默认 `/opt/robot-platform/train-venv`）创建 venv 并从 Git 安装 LeRobot，耗时较长且需要访问 PyPI/Git 镜像；安装末尾会用 `robot-train` 身份校验 `torch.cuda.is_available()`，CUDA 不可用时判定为失败；
- 重复执行是安全的：账号、目录、fstab 条目均按标记幂等处理。

详细步骤见[管理机安装](docs/02-management-node.md)。

### 4. 安装 GPU 节点

每台 GPU 主机执行：

```bash
./scripts/00-audit-host.sh gpu
sudo ./scripts/20-install-gpu-node.sh --apply
sudo ./scripts/25-install-training-environment.sh --apply
```

然后在包括管理机在内的 5 台主机收集 `slurmd -C`，用 `scripts/cluster/render-slurm-config.sh` 渲染全节点 `slurm.conf`，再分别在管理机执行 `cluster/install-controller-config.sh`、在各 Worker 执行 `cluster/install-worker-config.sh`。详细步骤见 [GPU 节点安装](docs/03-gpu-node.md)和[集群收尾](docs/06-cluster-finalization.md)。

注意事项：

- `install-controller-config.sh` 依赖 `10` 装好的 `slurmctld`，`install-worker-config.sh` 依赖 `20` 装好的 `slurmd`，顺序不能颠倒；
- 全部 5 台主机时间必须同步（角色安装脚本已启用 chrony，验收脚本会检查 `NTPSynchronized`），否则 Munge 认证失败；
- 5 台 Worker 全部就绪前不要进入下一步安装 leLab。

### 5. 安装 leLab 集群 Web

完成 Slurm 后，在管理机执行：

```bash
sudo ./scripts/15-install-lelab-platform.sh --apply
```

Web 默认监听 `http://192.168.100.202:8000`。详细配置见 [leLab 集群 Web](docs/07-lelab-cluster-web.md)。

注意事项：

- 前置条件缺一不可：仓库内 `apps/lelab/` 源码、Python 3.12（`LELAB_PYTHON_BIN`）、Node.js ≥ 20.19 与 npm、可用的 Slurm 客户端命令；
- 安装过程会校验 `robot-train` 对 NAS `datasets/` 可读、`jobs/` 可写；all_squash 下若失败，需在 QNAP 侧放开 guest 账号权限；
- 不要在 Slurm 集群收尾（第 4 步）完成前执行本步。

### 6. 安装采集节点

```bash
./scripts/00-audit-host.sh collector
sudo ./scripts/30-install-collector.sh --apply
```

注意事项：

- 脚本只安装 OS 前置组件、`robot-collector` 账号、Spool 状态目录（`recording` → `ready-to-upload` → `uploading` → `uploaded`/`failed`）和 systemd 模板；ROS2、H5 转换/校验和 Upload Agent 业务程序不在本仓库，需另行部署到 `/opt/robot-platform/bin/robot-upload-agent` 并配置上传令牌后服务才会启动；
- 采集节点不应直接获得 NAS `raw` 写权限，数据一律经管理机 API 上传。

详细步骤见[采集节点安装](docs/04-collector-node.md)。

### 7. 合并节点

同一台机器同时采集和训练时：

```bash
./scripts/00-audit-host.sh combined
sudo ./scripts/40-install-combined-node.sh --apply
```

注意事项：

- `40` 只是依次调用 `20` + `30`，因此合并节点不要再单独跑 `20`/`30`；
- 安装后还必须配置资源隔离（cgroup）和采集窗口；采集需要独占 GPU 或高磁盘带宽时，应先 `scontrol drain` 该 Slurm 节点，见[采集与 GPU 合并节点](docs/05-combined-node.md)；
- 合并节点上的采集服务同样只走管理机 API，不得授予 NAS `raw` 写权限。

### 8. 验收

```bash
./scripts/90-validate-deployment.sh management
./scripts/90-validate-deployment.sh gpu
./scripts/90-validate-deployment.sh collector
./scripts/90-validate-deployment.sh combined
```

验收脚本逐项检查时间同步、管理机/NAS 可达性、NFS 挂载及可写性、Docker/Munge/Slurm 服务、`nvidia-smi`、训练环境、leLab 与 MLflow 健康端点、`sinfo` 节点上报等，任一 FAIL 都应先修复再复跑。

第一阶段验收以“NAS 选数据 → 选择登记模型模板 → 自动选择空闲 GPU → Slurm 单机单卡训练 → checkpoint 中断续训”的真实短任务闭环为准；单纯“服务为 active”不等于平台完成。上传、QC、标注和 DatasetVersion 属于后续数据治理阶段。

### 通用注意事项

- **重复执行**：所有安装脚本按幂等设计（账号/目录/挂载按标记识别），失败后修正问题可直接重跑同一脚本；
- **不会做的事**：脚本不会安装 NVIDIA 驱动、不会格式化磁盘、不会覆盖人工修改过的 `/etc/hosts` 托管块和异源挂载；
- **失败排查顺序**：先看脚本输出的 `ERROR` 行，再对照 `00-audit-host.sh <role>` 的 MISSING/INACTIVE 项；多数失败源于 QNAP 白名单、驱动未装、UID/GID 冲突或主机名解析不一致；
- **私密信息**：`config/site.env`、`deploy/management/.env`、Munge 密钥和上传令牌一律不得提交到版本库。

## 当前环境注意事项

- `.209` 在前次检查中网络不可达，恢复前无法完成 5 节点验收。
- QNAP 当前导出白名单缺少 `.123`，必须补充。
- 现有 `/home/kewei/NAS` 是手工挂载；应迁移为统一的 `/mnt/robot_platform` 自动挂载。
- 当前阶段 NAS 权限以五台主机可读写为目标；进入正式数据治理阶段后再收紧。
- 管理机 Docker 镜像和构建缓存占用较大，正式构建前应为 Docker/数据库规划独立 SSD 空间。
