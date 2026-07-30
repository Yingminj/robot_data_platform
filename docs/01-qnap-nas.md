# QNAP NAS 配置

NAS 是原始 H5、数据集版本、标注导出、MLflow artifact 和模型发布包的权威文件存储，不运行 PostgreSQL、MLflow Server 或 Slurm。

## 1. 存储布局

优先在 QNAP 中创建一个独立共享文件夹 `robot-platform`。如果现阶段只能继续使用 `/kmd_data_file`，至少创建专用子目录：

```text
/kmd_data_file/robot-platform/
├── incoming/
├── raw/
├── quarantine/
├── annotations/
├── datasets/
├── jobs/
├── mlflow-artifacts/
├── model-releases/
├── backups/
└── trash/
```

不要把已有共享根目录中的其他项目迁移或改权限。平台目录应与现有数据隔离。

## 2. 第一阶段 NFS 服务

在 QTS 中启用 NFSv4，并为共享文件夹添加主机访问规则：

| IP | 角色 | 试点权限 |
|---|---|---|
| `192.168.100.202` | mgmt01（管理 + GPU） | 读写 |
| `192.168.100.215` | gpu01 | 读写 |
| `192.168.100.206` | gpu02 | 读写 |
| `192.168.100.208` | gpu03 | 读写 |
| `192.168.100.209` | gpu04 | 读写 |

五台机器使用同一个导出和挂载点 `/mnt/robot_platform`。需要至少建立：

```text
datasets/
jobs/
mlflow-artifacts/
```

试点阶段不要求 QNAP 按 Linux 数字 UID/GID 建立 ACL，也不做成员级权限。本部署保留 QNAP 默认的"映射所有用户到 guest"（all_squash）：所有平台账号（`robot-ingest`、`robot-train`、容器内进程）在 NAS 上统一按 `guest` 评估权限，目标是所有节点上的平台服务都能读取 `datasets`、写入各自的 `jobs/<job-id>`。仍建议只对白名单中的五个 IP 开放，而不是整个网段。

all_squash 模式下需要确认：

1. QTS → 控制台 → 权限 → 共享文件夹 → `robot_platform`：授予 `guest` 账号 **RW**（guest 默认常被拒绝，拒绝时所有平台账号的读写都会失败，且现象与 Linux 侧权限无关）。
2. 骨架目录只需 guest 可写。由于所有客户端用户都被映射为 guest，目录属主为 guest 即可满足全部平台服务；可直接在任一挂载点创建：

   ```bash
   sudo mkdir -p /mnt/robot_platform/{incoming,raw,quarantine,annotations,datasets,jobs,mlflow-artifacts,model-releases,backups,trash}
   ```

   若目录已存在且属主是 admin 或 `2200:2200`（guest 无权访问），需在 QNAP 上 SSH 修正，客户端 root 被映射为 guest、无权改动他人属主的目录：

   ```bash
   chmod -R 0777 /share/robot_platform/{incoming,raw,quarantine,annotations,datasets,jobs,mlflow-artifacts,model-releases}
   ```

3. 此模式下 NAS 上所有文件都归 guest 所有，无逐用户审计；数字 UID/GID ACL 和 setgid 约定推迟到数据治理阶段再启用。

注意：Slurm 自身仍要求五个 Worker 上的训练账号 UID/GID 一致。`config/site.env` 中的 `TRAIN_UID` 和 `DATA_GID` 只解决 Slurm 运行身份，不参与本阶段的 NAS 权限设计。

## 3. 数据保护

至少配置：

- 平台目录定时快照；
- 快照保留策略；
- 容量达到 70%、80%、90% 的分级告警；
- 磁盘、RAID 和风扇健康告警；
- PostgreSQL 备份目录的额外保留策略；
- 第二台存储设备或离线介质上的第二副本。

NAS 快照可以恢复误删，但不能替代独立备份。

## 4. NAS 验收

在管理机确认：

```bash
showmount -e 192.168.100.184
findmnt /mnt/robot_platform
df -hT /mnt/robot_platform
```

在五台节点分别确认挂载为 `rw`。使用平台训练账号验证：

```bash
sudo -u robot-train test -r /mnt/robot_platform/datasets
sudo -u robot-train test -w /mnt/robot_platform/jobs
```

QNAP 的 UID/GID 映射、细粒度 ACL 和原始数据保护在后续正式数据治理阶段处理，不作为第一阶段训练平台上线的前置条件。
