# 采集与 GPU 合并节点

同一台机器可以同时作为采集节点和 GPU 节点，但需要安装两个角色的并集，并保持账号、目录和数据流隔离。

## 1. 安装

```bash
cp config/site.env.example config/site.env
editor config/site.env
./scripts/00-audit-host.sh combined
sudo ./scripts/40-install-combined-node.sh --apply
```

需要的服务：

| 类别 | 服务 |
|---|---|
| GPU | NVIDIA Driver、Docker、NVIDIA Toolkit、Munge、slurmd |
| 存储 | NFS 只读挂载、本地 Dataset cache、`/work/runs` |
| 采集 | ROS2、相机/机器人驱动、H5 converter、validator、Upload Agent |
| 公共 | Chrony、SSH、监控 |

不需要安装 PostgreSQL、MLflow Server 或 Slurm Controller。

## 2. 权限隔离

- `robot-collector` 只写 `/var/spool/robot-data`；
- `robot-train` 只写 `/cache` 和 `/work/runs`；
- NAS 在该主机仍以只读方式挂载；
- 采集数据通过管理机 API 上传；
- 不要因为机器同时是 GPU 节点就给采集进程 `raw` 写权限。

## 3. 资源冲突

合并节点最容易出现以下问题：

- 训练占满 GPU，影响相机或视觉预处理；
- 数据预热占满磁盘带宽，导致 H5 写入丢帧；
- 训练占满内存，导致 ROS2/H5 converter 被 OOM；
- `/work` 和 Spool 共用磁盘，训练输出挤占未上传数据空间。

至少选择一种运行策略：

1. 采集期间将节点置为 Slurm drain；
2. 为采集安排固定时段，训练只在非采集时段运行；
3. 在 Slurm 中保留 CPU/内存，并将 Spool 放在独立 SSD；
4. 如果采集也需要 GPU，则采集期间禁止该节点接收 GPU Job。

采集前 drain 示例：

```bash
sudo scontrol update NodeName=gpu01 State=DRAIN Reason=collection
```

采集结束并确认上传安全后恢复：

```bash
sudo scontrol update NodeName=gpu01 State=RESUME
```

这些命令需要 Slurm 管理权限，应通过管理机或受控运维流程执行。

## 4. 建议磁盘布局

```text
独立采集盘  /var/spool/robot-data
训练缓存盘  /cache/datasets
训练工作盘  /work/runs
NAS          /mnt/robot_platform（只读）
```

如果只能共用一个 NVMe，应分别设置空间水位，并优先保护未 `COMMITTED` 的采集数据。

