# Slurm 集群收尾与验收

## 0. 统一地址解析

在管理机和 4 台远程 GPU 节点使用相同的 `config/site.env`，然后执行：

```bash
sudo ./scripts/05-configure-hosts.sh --apply
```

执行前必须确认没有现有 DNS 或 `/etc/hosts` 冲突。脚本发现同名旧条目时会停止，不会覆盖。

## 1. 准备节点资源文件

在管理机：

```bash
cp config/slurm/nodes.conf.example config/slurm/nodes.conf
editor config/slurm/nodes.conf
```

把 5 台主机的 `slurmd -C` 输出填入对应行，包含 `mgmt01`，并保留固定的 `NodeName`、`NodeAddr` 和 `Gres=gpu:1`。`mgmt01` 的 `RealMemory` 应额外为 leLab、Slurm Controller、PostgreSQL、Redis 和 MLflow 预留空间。确认没有 `FILL_ME`：

```bash
./scripts/cluster/render-slurm-config.sh
less config/slurm/slurm.conf.generated
```

所有节点与管理机必须使用同一 Slurm 主版本。Ubuntu 22.04 自带包可能较旧；在启用 `cgroup.conf` 前，应确认所安装版本支持主机当前的 cgroup v2。若版本不满足要求，应统一使用组织批准的较新 Slurm 包或构建产物，不能混用不同主版本。

## 2. 启动 Controller 和管理机 Worker

```bash
sudo ./scripts/cluster/install-controller-config.sh \
  config/slurm/slurm.conf.generated \
  --apply
```

该脚本同时启动 `slurmctld` 和管理机上的 `slurmd`。

## 3. 分发配置和 Munge 密钥

所有节点必须使用完全相同的：

- `/etc/munge/munge.key`；
- `slurm.conf`；
- `cgroup.conf`；
- 主机名和地址解析。

Munge 密钥属于集群机密。应使用已有的安全配置管理工具，或通过管理员控制的临时加密通道分发。不要把它复制进本工作区、Git、NAS 公共目录或聊天记录。

在节点上使用：

```bash
sudo ./scripts/cluster/install-worker-config.sh \
  /secure/temp/munge.key \
  /secure/temp/slurm.conf.generated \
  --apply
```

验证成功后删除临时副本。

## 4. 基础检查

管理机执行：

```bash
munge -n | unmunge
sinfo -N -l
scontrol show nodes
```

每台节点应显示一个 `gpu` GRES。节点为 `INVAL`、`DOWN` 时，优先对比：

- `slurmd -C` 与 `slurm.conf`；
- 系统时间；
- Munge 密钥 checksum、属主和权限；
- `/etc/hosts` 和主机名；
- 6817/6818 防火墙；
- `gres.conf` 与 `/dev/nvidia0`。

## 5. 五节点 GPU smoke test

先确认五台机器都完成 `scripts/25-install-training-environment.sh`，再准备一个使用统一虚拟环境、仅输出 GPU 信息的 sbatch 脚本，然后验证：

- 每个节点单任务；
- 5 个单 GPU 任务同时运行；
- 任务取消；
- 超时；
- 节点 drain/resume；
- Job ID 能写入平台记录。

## 6. 第一阶段训练闭环验收

使用 NAS 中的一份小型 LeRobot 数据集验证：

1. 5 台主机都能读写 NFS；
2. leLab 页面显示 5 台节点 GPU 状态；
3. 手动启动一个 CUDA 进程后，该节点不再显示为空闲；
4. leLab 从 NAS 数据集目录选择数据并提交固定模型模板；
5. Slurm 在一台空闲节点启动统一训练环境；
6. 页面持续显示日志、step、loss 和 checkpoint；
7. 中断任务后，使用 NAS 上的最新 checkpoint 在任意空闲节点继续；
8. 5 个单 GPU smoke job 能同时占用 5 台节点。

采集上传、QC、标注、DatasetVersion、精细权限和 MLflow 深度集成属于后续数据治理范围，不阻塞第一阶段训练调度平台。
