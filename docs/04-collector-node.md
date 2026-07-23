# 采集节点安装

采集节点运行 ROS2 采集、H5 转换、本地验证和上传 Agent。默认不挂载 NAS，通过管理机 API 上传。

## 1. 安装基础组件

```bash
cp config/site.env.example config/site.env
editor config/site.env
./scripts/00-audit-host.sh collector
sudo ./scripts/30-install-collector.sh --apply
```

脚本创建：

```text
/var/spool/robot-data/
├── recording/
├── ready-to-upload/
├── uploading/
├── uploaded/
└── failed/
```

以及：

```text
/etc/robot-platform/collector.env
/etc/systemd/system/robot-upload-agent.service
```

## 2. 需要项目提供的组件

当前工作区没有采集应用源码，因此还需要安装：

- ROS2 及机器人/相机驱动；
- H5 converter；
- H5 schema validator；
- manifest 生成器；
- Upload Agent；
- 平台签发的节点上传 Token。

Upload Agent 的预留安装路径：

```text
/opt/robot-platform/bin/robot-upload-agent
```

Token 路径：

```text
/etc/robot-platform/upload.token
```

Token 建议权限：

```bash
sudo chown root:robotdata /etc/robot-platform/upload.token
sudo chmod 0640 /etc/robot-platform/upload.token
```

应用就绪后：

```bash
sudo systemctl enable --now robot-upload-agent
```

## 3. 采集状态规则

- 写入中的 H5 只能位于 `recording`；
- H5 正常关闭后才执行 validator；
- validator、manifest 和 SHA-256 全部成功后移动到 `ready-to-upload`；
- 上传使用 `.partial`/断点续传；
- 服务端返回 `COMMITTED` 前不能清理本地文件；
- 本地空间低于 `COLLECTOR_MIN_FREE_GB` 时停止新采集，而不是删除未提交数据。

## 4. 网络与权限

采集节点只需要访问管理机 HTTPS 443。它不需要 NAS `raw` 写权限，也不需要 PostgreSQL、Redis、MLflow Server、Munge 或 Slurm。

如果采集数据包含人脸、声音或工作区敏感画面，应在进入外部数据仓库前执行脱敏和审批；本地原始数据仍按组织保留规则处理。

