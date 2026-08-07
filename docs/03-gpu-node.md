# GPU Worker 安装

GPU Worker 运行 NVIDIA Driver、Docker/NVIDIA Runtime、Munge、`slurmd` 和统一 LeRobot 训练环境。本文以 `gpu01` 为例，**每台 GPU Worker 的步骤完全相同**，只需替换主机名、IP 和 SSH 账号。命令除特别说明外，都在该 Worker 的仓库根目录执行。

当前各 Worker：

| Slurm NodeName | IP | 管理员 SSH 目标 |
|---|---|---|
| `gpu01` | `192.168.100.215` | `snorlax@192.168.100.215` |
| `gpu02` | `192.168.100.216` | `yang@192.168.100.216` |
| `gpu03` | `192.168.100.217` | `snorlax@192.168.100.217` |

Slurm 作业统一以 `robot-train` 运行。

**SSH 账号只用于登录和 leLab 的只读 GPU 探测，Slurm 的节点名与它无关。** 上表中 `gpu02` 的账号是 `yang`，与另两台不同，这是允许的——账号不必与 NodeName 相同，各节点之间也不必相同。`hostname -s` 必须等于 Slurm NodeName。

向已经运行的集群加节点时，除本文外还有一批集群级改动，见[向已有集群增加 GPU 节点](09-add-gpu-node.md)。

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
sudo -u robot-train test -w /var/lib/robot-platform/cache
sudo -u robot-train test -r /mnt/robot_platform/datasets
sudo -u robot-train test -w /mnt/robot_platform/jobs
findmnt /mnt/robot_platform
```

如果这些本地目录因早期安装中断而缺失，可补建：

```bash
sudo install -d -o robot-train -g robotdata -m 0750 \
  /cache/datasets \
  /cache/exports \
  /work/runs \
  /var/lib/robot-platform/huggingface \
  /var/lib/robot-platform/cache
```

`/var/lib/robot-platform/cache` 对应 leLab 的 `LELAB_JOB_CACHE_ROOT`，**必须在每台 Worker 上存在且 `robot-train` 可写**。Slurm 会把 `HOME` 指向 `robot-train` 的家目录，而它在 Worker 上并不存在，缓存到 `~` 的作业会在该节点失败。各节点用相同的本地路径即可，不需要共享存储。

NFS 挂载失败时先确认 QNAP 的白名单包含**这台**新节点的 IP，加节点时最容易漏掉这一项。

## 4. 安装统一训练环境

```bash
sudo ./scripts/25-install-training-environment.sh --apply
sudo -u robot-train /opt/robot-platform/train-venv/bin/python -c \
  'import torch, lerobot; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

期望 CUDA 为 `True`。不要在作业启动时临时安装 Python 包；所有 Worker 应使用同一路径和同一版本。

## 5. 安装 Slurm 26.05.2

全集群必须安装完全相同的 Slurm 主版本和包构建。按 [Slurm 26.05.2 DEB 安装说明](Slurm-INSTALL.md)执行后检查：

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

此时 `sbatch --version` 会报错，**这是正常的，不要据此判断安装失败**：

```text
sbatch: error: resolve_ctls_from_dns_srv: res_nsearch error: Unknown host
sbatch: error: fetch_config: DNS SRV lookup failed
sbatch: fatal: Could not establish a configuration source
```

新版 Slurm 在打印版本前会先尝试加载配置。本机此时还没有 `/etc/slurm/slurm.conf`，于是回退到本集群不使用的 configless DNS SRV 发现。第 7 步装入配置后该报错消失。查版本用 `/usr/sbin/slurmd -V`，不受影响。

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

> **整段一次执行完，不要只粘贴后半段。** 下面每处 `${stage_dir:?}` 的 `:?` 都不能省略：如果漏掉第一行 `stage_dir="$(mktemp -d)"`，变量为空，`"$stage_dir/munge.key"` 会变成 `/munge.key`，而带 `sudo` 的那条会**静默成功**，把集群唯一的认证密钥写到文件系统根目录。加上 `:?` 后 bash 会立即报 `stage_dir: parameter null or not set` 并终止。

```bash
stage_dir="$(mktemp -d)"
sudo install -o "$USER" -g "$(id -gn)" -m 0600 \
  /etc/munge/munge.key \
  "${stage_dir:?}/munge.key"
install -m 0644 \
  config/slurm/slurm.conf.generated \
  "${stage_dir:?}/slurm.conf.generated"

ssh snorlax@192.168.100.215 \
  'install -d -m 0700 ~/robot-platform-secure'
scp \
  "${stage_dir:?}/munge.key" \
  "${stage_dir:?}/slurm.conf.generated" \
  snorlax@192.168.100.215:~/robot-platform-secure/

shred -u "${stage_dir:?}/munge.key"
rm -f "${stage_dir:?}/slurm.conf.generated"
rmdir "${stage_dir:?}"
```

如果已经误写到根目录，检查并销毁：

```bash
ls -l /munge.key && sudo shred -u /munge.key
```

### 7.2 在 gpu01 安装

从 `gpu01` 仓库根目录执行：

```bash
sudo ./scripts/cluster/install-worker-config.sh \
  /home/snorlax/robot-platform-secure/munge.key \
  /home/snorlax/robot-platform-secure/slurm.conf.generated \
  --apply
```

这里的两个文件路径必须在**执行命令的这台机器**上真实存在。`/secure/temp/...` 只是旧文档中的占位写法，不是预设目录。脚本只输出 usage 时，说明这两个文件之一不可读，或第三个参数不是 `--apply`。

验证后销毁 Worker 上的临时密钥：

```bash
shred -u /home/snorlax/robot-platform-secure/munge.key
rm -f /home/snorlax/robot-platform-secure/slurm.conf.generated
rmdir /home/snorlax/robot-platform-secure
```

检查：

```bash
systemctl is-active munge slurmd
sudo slurmd -G
sha256sum /etc/slurm/slurm.conf /etc/slurm/cgroup.conf /etc/slurm/gres.conf
journalctl -u slurmd -n 50 --no-pager
```

三个 checksum 必须与 `mgmt01` 上的完全一致，这是判断配置是否真的装对的唯一可靠方式。

`slurmd -G` 会输出一条 GRES 类型提示，**这是正常的**：

```text
gres/gpu: _normalize_sys_gres_types: Could not find an unused configuration record
with a GRES type that is a substring of system device `nvidia_geforce_rtx_4090`.
Setting system GRES type to NULL
```

`gres.conf` 声明的是不带型号的 `Name=gpu`，NVML 报告的设备型号是 `nvidia_geforce_rtx_4090`，于是 Slurm 把类型置为 NULL，与 `nodes.conf` 中同样不带型号的 `Gres=gpu:1` 一致。紧随其后的这行才是结论：

```text
Gres Name=gpu Type=(null) Count=1 Index=0 File=/dev/nvidia0 Flags=HAS_FILE,ENV_NVML
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
- 手动 CUDA 进程会让 leLab 把该节点标记为不可调度（`eligible: false`）；
- CPU 日常进程不影响 leLab 的 GPU 空闲判断；
- 不开放 Docker TCP API；
- Worker 不运行 `slurmctld`、PostgreSQL、Redis、MLflow 或 leLab。

远程桌面类工具（RustDesk、TeamViewer、向日葵等）会占用 GPU 并被计为 compute process，使该节点长期 `eligible: false`。leLab 已内置一份图形进程名白名单（见 `apps/lelab/lelab/cluster.py` 的 `graphics_patterns`），本机在用的工具若不在其中，把进程名补进去，不要靠关闭工具绕开。
