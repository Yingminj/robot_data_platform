# Slurm 26.05.2 DEB 安装说明

本文只负责把两台 Ubuntu 22.04 `amd64` 主机统一到支持 cgroup v2 的 Slurm 26.05.2。集群节点参数、Munge 分发和启动顺序见 [Slurm 双节点集群收尾](06-cluster-finalization.md)。

仓库内构建产物：

```text
artifacts/slurm-26.05.2-jammy-amd64/
├── SHA256SUMS
├── debs/
├── metadata/
└── source/
```

压缩归档及校验：

```text
artifacts/slurm-26.05.2-jammy-amd64.tar.gz
artifacts/slurm-26.05.2-jammy-amd64.tar.gz.sha256
```

## 1. 适用范围和安装顺序

当前推荐顺序：

```text
mgmt01 先执行 scripts/10-install-management.sh
gpu01 先执行 scripts/20-install-gpu-node.sh
→ 两台主机安装本页 Slurm 26.05.2
→ 渲染并安装最终 Slurm 配置
```

原因是角色脚本会准备账号、目录、NFS 和其他系统依赖，但 Ubuntu 22.04 自带的 Slurm 较旧。最终版本必须以本页验证结果为准。

安装 26.05.2 后，避免再次执行会明确安装 Ubuntu `slurm-wlm` 的操作。如确需重跑 `10`/`20`，先用 `apt -s` 检查模拟事务，确认不会移除 `slurm-smd*`：

```bash
sudo apt -s install slurm-wlm
```

如果模拟结果要删除或替换 `slurm-smd*`，不要确认事务。

## 2. 构建信息

- 上游源码：SchedMD Slurm 26.05.2；
- Debian 包版本：`26.05.2-1`；
- 构建目标：Ubuntu 22.04 Jammy，`amd64`；
- 已包含 `cgroup_v2.so`、`task_cgroup.so`、`proctrack_cgroup.so`；
- 使用 MUNGE 认证；
- 包含 NVIDIA NVML GPU 探测插件。

NVML 在运行时加载 NVIDIA 驱动提供的 `libnvidia-ml.so.1`，这些 DEB 不绑定特定驱动分支。

## 3. 把产物放到两台主机

两台主机都需要完整的 `artifacts/slurm-26.05.2-jammy-amd64` 目录。若 `gpu01` 上的仓库副本没有该目录，可从 `mgmt01` 复制归档：

```bash
scp \
  artifacts/slurm-26.05.2-jammy-amd64.tar.gz \
  artifacts/slurm-26.05.2-jammy-amd64.tar.gz.sha256 \
  snorlax@192.168.100.215:~/
```

在 `gpu01` 校验外层归档：

```bash
cd ~
sha256sum -c slurm-26.05.2-jammy-amd64.tar.gz.sha256
tar -xzf slurm-26.05.2-jammy-amd64.tar.gz
```

若两台主机都有完整仓库，可以直接使用仓库内已经解压的目录，无需重复解压。

## 4. 校验 DEB

进入产物目录：

```bash
cd artifacts/slurm-26.05.2-jammy-amd64
sha256sum -c SHA256SUMS
```

如果从 home 解压，则进入：

```bash
cd ~/slurm-26.05.2-jammy-amd64
sha256sum -c SHA256SUMS
```

任何校验失败都应停止安装并重新复制文件。

## 5. 备份旧配置

两台主机都执行：

```bash
sudo systemctl stop slurmctld slurmd 2>/dev/null || true

if sudo test -d /etc/slurm; then
  sudo cp -a \
    /etc/slurm \
    "/etc/slurm.backup.$(date +%Y%m%d-%H%M%S)"
fi
```

不要删除 `/etc/munge/munge.key`。`mgmt01` 上由角色脚本生成的密钥将作为集群唯一来源。

## 6. 安装四个核心包

在两台主机的产物目录中执行同一命令：

```bash
sudo apt update
sudo apt install -y munge \
  ./debs/slurm-smd_26.05.2-1_amd64.deb \
  ./debs/slurm-smd-client_26.05.2-1_amd64.deb \
  ./debs/slurm-smd-slurmctld_26.05.2-1_amd64.deb \
  ./debs/slurm-smd-slurmd_26.05.2-1_amd64.deb
```

使用 `apt install ./debs/...`，不要用 `dpkg -i` 绕过依赖解析。安装事务中如果出现计划删除关键系统组件，应先取消并检查包冲突。

当前集群不需要 `slurmdbd`、`slurmrestd`、开发包、PAM、NSS、PMI、Sview 或 Sackd。

## 7. 安装后版本检查

两台主机都执行：

```bash
/usr/sbin/slurmctld -V
/usr/sbin/slurmd -V
sbatch --version
dpkg-query -S /usr/sbin/slurmd
dpkg-query -S /usr/sbin/slurmctld
```

期望前三项都是 `26.05.2`，二进制归属为 `slurm-smd-slurmd` 和 `slurm-smd-slurmctld`。即使 dpkg 数据库中仍能看到旧包名，也应以实际二进制版本与归属检查为准。

## 8. cgroup v2 和 NVML 检查

两台主机：

```bash
stat -fc %T /sys/fs/cgroup
nvidia-smi
ldconfig -p | grep -E 'libnvidia-ml\.so\.1'
find /usr/lib -type f \
  \( -name cgroup_v2.so -o -name gpu_nvml.so \) \
  -print
```

期望：

- cgroup 文件系统输出 `cgroup2fs`；
- `nvidia-smi` 正常；
- 能找到 `cgroup_v2.so` 和 `gpu_nvml.so`。

仓库最终安装的 cgroup 配置为：

```ini
CgroupPlugin=autodetect
ConstrainCores=yes
ConstrainRAMSpace=yes
ConstrainDevices=yes
ConstrainSwapSpace=yes
```

GPU 配置为单卡 `/dev/nvidia0`：

```ini
AutoDetect=nvml
Name=gpu File=/dev/nvidia0
```

## 9. 此时如何处理服务

刚装完软件包但还没有最终集群配置时，不要尝试用默认配置启动整个集群。

在 `mgmt01` 最终运行：

```text
munge + slurmctld + slurmd
```

在 `gpu01` 最终运行：

```text
munge + slurmd
```

具体启用和重启由以下脚本完成：

```bash
# mgmt01
sudo ./scripts/cluster/install-controller-config.sh \
  config/slurm/slurm.conf.generated \
  --apply

# gpu01
sudo ./scripts/cluster/install-worker-config.sh \
  <本机munge.key路径> \
  <本机slurm.conf.generated路径> \
  --apply
```

不要在 `gpu01` 启用 `slurmctld`。

## 10. 最终验证

配置安装后，两台主机分别执行：

```bash
sudo slurmd -C
sudo slurmd -G
systemctl --no-pager --full status munge slurmd
```

只在 `mgmt01`：

```bash
systemctl --no-pager --full status slurmctld
scontrol ping
sinfo -N -l
```

软件版本、cgroup 插件、GRES、Munge 和两节点 smoke test 全部通过，才算 Slurm 安装完成。
