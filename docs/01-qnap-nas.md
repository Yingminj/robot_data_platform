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
├── mlflow-artifacts/
├── model-releases/
├── backups/
└── trash/
```

不要把已有共享根目录中的其他项目迁移或改权限。平台目录应与现有数据隔离。

## 2. NFS 服务

在 QTS 中启用 NFSv4，并为共享文件夹添加主机访问规则：

| IP | 角色 | 建议权限 |
|---|---|---|
| `192.168.100.202` | 管理机 | 读写 |
| `192.168.100.123` | gpu01 | 只读 |
| `192.168.100.206` | gpu02 | 只读 |
| `192.168.100.208` | gpu03 | 只读 |
| `192.168.100.209` | gpu04 | 只读 |

采集节点默认不挂载 NAS，而是通过管理机上传 API 写入。如果必须使用 NFS 直传，只能给该节点一个独立的 `incoming/<node-id>` 可写目录，不能开放 `raw`。

不要直接对整个 `192.168.100.0/24` 开放读写。

## 3. UID/GID 和 ACL

当前 NFS 使用 `sec=sys`，权限依赖数字 UID/GID。部署前确认 `config/site.env` 中的 ID 未被占用，并在 QNAP 权限中做对应映射：

```text
robotdata       GID 2200
robot-ingest    UID 2200
robot-collector UID 2201
robot-train     UID 2202
```

这些只是示例默认值；如果 QNAP 或 Linux 已占用，必须在安装前整体更换。

推荐权限：

- `raw`：`robot-ingest` 可写，`robotdata` 只读；
- `datasets`：数据集生成服务可写，`robotdata` 只读；
- `mlflow-artifacts`：管理机 MLflow 服务可写；
- `backups`：仅备份服务和管理员可写；
- 普通 Windows/SMB 用户不能写 `raw`。

如果 QNAP 无法在单一共享文件夹中实现这些 ACL，创建多个共享文件夹并分别导出，不要退回到 `777`。

## 4. 数据保护

至少配置：

- 平台目录定时快照；
- 快照保留策略；
- 容量达到 70%、80%、90% 的分级告警；
- 磁盘、RAID 和风扇健康告警；
- PostgreSQL 备份目录的额外保留策略；
- 第二台存储设备或离线介质上的第二副本。

NAS 快照可以恢复误删，但不能替代独立备份。

## 5. NAS 验收

在管理机确认：

```bash
showmount -e 192.168.100.184
findmnt /mnt/robot-platform
df -hT /mnt/robot-platform
```

在 GPU 节点确认挂载参数包含 `ro`；在管理机确认平台服务账号可以写 `mlflow-artifacts`。不要用普通用户在 `raw` 中创建测试文件。

