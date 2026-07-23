# Slurm 集群收尾与验收

## 0. 统一地址解析

在管理机和 4 台 GPU 节点使用相同的 `config/site.env`，然后执行：

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

把每台主机的 `slurmd -C` 输出填入对应行，保留固定的 `NodeName`、`NodeAddr` 和 `Gres=gpu:1`。生产配置通常应把 `RealMemory` 比检测总量降低约 5%～10%，为操作系统、Docker 和采集进程保留空间。确认没有 `FILL_ME`：

```bash
./scripts/cluster/render-slurm-config.sh
less config/slurm/slurm.conf.generated
```

所有节点与管理机必须使用同一 Slurm 主版本。Ubuntu 22.04 自带包可能较旧；在启用 `cgroup.conf` 前，应确认所安装版本支持主机当前的 cgroup v2。若版本不满足要求，应统一使用组织批准的较新 Slurm 包或构建产物，不能混用不同主版本。

## 2. 启动 Controller

```bash
sudo ./scripts/cluster/install-controller-config.sh \
  config/slurm/slurm.conf.generated \
  --apply
```

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

## 5. 四节点 GPU smoke test

先准备团队批准的训练镜像和一个仅输出 GPU 信息的 sbatch 脚本，然后验证：

- 每个节点单任务；
- 4 个单 GPU 任务同时运行；
- 任务取消；
- 超时；
- 节点 drain/resume；
- Job ID 能写入平台/MLflow。

不要在验收脚本中使用未经固定 digest 的 `latest` 镜像。

## 6. 数据闭环验收

最终必须用一条短测试 Episode 验证：

1. 采集节点生成唯一 Episode ID；
2. H5 关闭、validator、manifest 和 SHA-256 成功；
3. Upload Agent 上传到 `incoming`；
4. 管理端二次验证后进入 `raw`；
5. PostgreSQL 能查询 NAS 路径和 QC；
6. Web 页面能够查看预览并提交标注；
7. 冻结 DatasetVersion 和 split；
8. GPU 节点把数据预热到本地 cache；
9. Slurm 启动固定镜像；
10. MLflow 保存参数、指标、best/last checkpoint；
11. 数据库和 NAS artifact 完成一次恢复演练。

缺少平台 API、QC/标注应用或训练代码时，只能验收基础设施，不能宣称整个平台完成。
