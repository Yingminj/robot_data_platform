# GPU Worker `gpu01` 安装

`gpu01` 运行 NVIDIA Driver、Docker/NVIDIA Runtime、Munge、`slurmd` 和统一 LeRobot 训练环境。本文命令除特别说明外，都在 `gpu01` 的仓库根目录执行。

当前主机信息：

| 项目 | 值 |
|---|---|
| Slurm NodeName | `gpu01` |
| IP | `192.168.100.215` |
| 管理员 SSH 目标 | `snorlax@192.168.100.215` |
| Slurm 作业用户 | `robot-train` |

`snorlax` 只用于登录和 leLab 的只读 GPU 探测；Slurm 的节点名仍然必须是 `gpu01`。

## 1. 安装前检查

```bash
hostname -s
ip -br address
cp config/site.env.example config/site.env
editor config/site.env
```

如果主机名还不是 `gpu01`：

```bash
sudo hostnamectl set-hostname gpu01
```

重新登录后执行：

```bash
sudo ./scripts/05-configure-hosts.sh --apply
./scripts/00-audit-host.sh gpu
nvidia-smi
timedatectl show --property=NTPSynchronized --value
getent hosts mgmt01 gpu01
```

## 2. 准备 Python 3.12

```bash
python3.12 --version
```

Ubuntu 22.04 当前环境的安装方式：

```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-dev python3.12-venv
```

Worker 不需要 Node.js 或 npm。

## 3. 安装 Worker 基础组件

```bash
sudo ./scripts/20-install-gpu-node.sh --apply
```

脚本会安装或准备：

- NFS、Chrony、Munge、SSH；
- Docker 和 NVIDIA Container Toolkit；
- `robotdata` 与 `robot-train`；
- `/cache/datasets`、`/cache/exports`、`/work/runs`；
- `/var/lib/robot-platform/slurmd`；
- QNAP 的持久读写挂载。

首次执行时不会使用未知 Munge 密钥启动 `slurmd`。这是正常的，集群配置在后面安装。

检查本地目录和 NFS：

```bash
sudo -u robot-train test -d /cache/datasets
sudo -u robot-train test -d /cache/exports
sudo -u robot-train test -d /work/runs
sudo -u robot-train test -r /mnt/robot_platform/datasets
sudo -u robot-train test -w /mnt/robot_platform/jobs
findmnt /mnt/robot_platform
```

如果前三个本地目录因早期安装中断而缺失，可补建：

```bash
sudo install -d -o robot-train -g robotdata -m 0750 \
  /cache/datasets \
  /cache/exports \
  /work/runs \
  /var/lib/robot-platform/huggingface
```

## 4. 安装统一训练环境

```bash
sudo ./scripts/25-install-training-environment.sh --apply
sudo -u robot-train /opt/robot-platform/train-venv/bin/python -c \
  'import torch, lerobot; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

期望 CUDA 为 `True`。不要在作业启动时临时安装 Python 包；所有 Worker 应使用同一路径和同一版本。

## 5. 安装 Slurm 26.05.2

两台机器必须安装完全相同的 Slurm 主版本和包构建。按 [Slurm 26.05.2 DEB 安装说明](Slurm-INSTALL.md)执行后检查：

```bash
slurmd -V
stat -fc %T /sys/fs/cgroup
test -e /usr/lib/x86_64-linux-gnu/slurm-wlm/cgroup_v2.so \
  || test -e /usr/lib/x86_64-linux-gnu/slurm/cgroup_v2.so
```

期望：

```text
slurm 26.05.2
cgroup2fs
```

## 6. 收集真实硬件参数

在 `gpu01`：

```bash
sudo slurmd -C
```

把输出中的 `CPUs`、`Boards`、`SocketsPerBoard`、`CoresPerSocket`、`ThreadsPerCore` 和 `RealMemory` 填到管理机的 `config/slurm/nodes.conf`。不要照抄示例。

`RealMemory` 应略低于物理内存，给操作系统和后台服务留余量。

## 7. 接收并安装集群配置

先在 `mgmt01` 完成配置渲染。以下是一套明确的安全传输示例。

### 7.1 在 mgmt01 暂存并复制

```bash
stage_dir="$(mktemp -d)"
sudo install -o "$USER" -g "$(id -gn)" -m 0600 \
  /etc/munge/munge.key \
  "$stage_dir/munge.key"
install -m 0644 \
  config/slurm/slurm.conf.generated \
  "$stage_dir/slurm.conf.generated"

ssh snorlax@192.168.100.215 \
  'install -d -m 0700 ~/robot-platform-secure'
scp \
  "$stage_dir/munge.key" \
  "$stage_dir/slurm.conf.generated" \
  snorlax@192.168.100.215:~/robot-platform-secure/

rm -f "$stage_dir/munge.key" "$stage_dir/slurm.conf.generated"
rmdir "$stage_dir"
```

### 7.2 在 gpu01 安装

从 `gpu01` 仓库根目录执行：

```bash
sudo ./scripts/cluster/install-worker-config.sh \
  /home/snorlax/robot-platform-secure/munge.key \
  /home/snorlax/robot-platform-secure/slurm.conf.generated \
  --apply
```

这里的两个文件路径必须在 `gpu01` 本机真实存在。`/secure/temp/...` 只是旧文档中的占位写法，不是预设目录。

验证后删除 Worker 上的临时密钥：

```bash
rm -f \
  /home/snorlax/robot-platform-secure/munge.key \
  /home/snorlax/robot-platform-secure/slurm.conf.generated
rmdir /home/snorlax/robot-platform-secure
```

检查：

```bash
systemctl is-active munge slurmd
sudo slurmd -G
sha256sum /etc/slurm/slurm.conf /etc/slurm/cgroup.conf /etc/slurm/gres.conf
journalctl -u slurmd -n 50 --no-pager
```

## 8. 从管理机做调度测试

不要把直接 SSH 运行 Python 当作 Slurm 验收。回到 `mgmt01` 执行：

```bash
srun \
  --partition=debug \
  --nodes=1 \
  --nodelist=gpu01 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --mem=1G \
  --gres=gpu:1 \
  --time=00:02:00 \
  /opt/robot-platform/train-venv/bin/python -c \
  'import os, socket, torch; print(socket.gethostname()); print(os.environ.get("CUDA_VISIBLE_DEVICES")); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

## 9. Worker 运行约定

- 正式训练通过 Slurm，不通过个人 SSH 会话启动；
- `/mnt/robot_platform/jobs` 保存日志和 checkpoint，是共享持久数据；
- `/cache` 和 `/work` 是本地空间，可重建，不保存唯一副本；
- 手动 CUDA 进程会让 leLab 把该节点标记为不可调度；
- CPU 日常进程不影响 leLab 的 GPU 空闲判断；
- 不开放 Docker TCP API；
- `gpu01` 不运行 `slurmctld`、PostgreSQL、Redis、MLflow 或 leLab。
