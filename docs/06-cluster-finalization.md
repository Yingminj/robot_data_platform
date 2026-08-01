# Slurm 双节点集群收尾与验收

本文是当前 `mgmt01 + gpu01` 集群的权威配置顺序。所有“在管理机执行”的命令都从 `mgmt01` 仓库根目录运行；“在 Worker 执行”的命令都从 `gpu01` 仓库根目录运行。

## 0. 完成条件

进入本阶段前：

- `mgmt01` 已执行 `10` 和 `25`；
- `gpu01` 已执行 `20` 和 `25`；
- 两台机器已安装相同的 Slurm 26.05.2；
- 两台机器都使用 cgroup v2；
- 两台机器主机名、时间和 `/etc/hosts` 正确；
- QNAP 在相同绝对路径 `/mnt/robot_platform` 挂载；
- `robot-train` 和 `robotdata` 的数字 UID/GID 一致。

快速核对：

```bash
hostname -s
slurmd -V
stat -fc %T /sys/fs/cgroup
id robot-train
getent group robotdata
getent hosts mgmt01 gpu01
findmnt /mnt/robot_platform
```

## 1. 收集节点资源

两台主机分别执行：

```bash
sudo slurmd -C
```

在 `mgmt01` 复制模板：

```bash
cp config/slurm/nodes.conf.example config/slurm/nodes.conf
editor config/slurm/nodes.conf
```

每行都应：

- 使用固定 `NodeName=mgmt01` 或 `NodeName=gpu01`；
- 使用对应 `NodeAddr`；
- 保留本机 `slurmd -C` 的 CPU 拓扑；
- 设置略低于物理内存的 `RealMemory`；
- 追加 `Gres=gpu:1 State=UNKNOWN`。

当前结构示例：

```ini
NodeName=mgmt01 NodeAddr=192.168.100.202 CPUs=... RealMemory=... Gres=gpu:1 State=UNKNOWN
NodeName=gpu01 NodeAddr=192.168.100.215 CPUs=... RealMemory=... Gres=gpu:1 State=UNKNOWN
```

不要保留 `FILL_ME`，也不要把一台机器的硬件行复制给另一台。

## 2. 渲染并审查配置

在 `mgmt01`：

```bash
./scripts/cluster/render-slurm-config.sh
sed -n '1,240p' config/slurm/slurm.conf.generated
```

确认：

```bash
! grep -E 'FILL_ME|@@' config/slurm/slurm.conf.generated
grep '^NodeName=' config/slurm/slurm.conf.generated
grep '^PartitionName=' config/slurm/slurm.conf.generated
```

预期包含两个节点和 `debug`、`train`、`eval` 三个分区。

## 3. 安装 Controller 和 mgmt01 Worker

在 `mgmt01`：

```bash
sudo ./scripts/cluster/install-controller-config.sh \
  config/slurm/slurm.conf.generated \
  --apply
```

该脚本会把以下文件装入 `/etc/slurm`：

```text
slurm.conf
cgroup.conf
gres.conf
```

并重启 `munge`、`slurmctld` 和本机 `slurmd`。检查：

```bash
systemctl is-active munge slurmctld slurmd
munge -n | unmunge | sed -n '1,12p'
sudo slurmd -G
scontrol ping
```

## 4. 安装 gpu01 Worker

Worker 必须获得与管理机完全相同的：

- `/etc/munge/munge.key`；
- 生成的 `slurm.conf`；
- 仓库中的 `cgroup.conf` 和 `gres.conf`。

Munge 密钥只能经管理员控制的临时通道传输，不能放入 Git、NAS 公共目录或聊天记录。

具体的 `scp` 暂存命令见 [GPU Worker 安装：接收并安装集群配置](03-gpu-node.md#7-接收并安装集群配置)。

在 `gpu01` 的调用形式是：

```bash
sudo ./scripts/cluster/install-worker-config.sh \
  <gpu01本机的munge.key路径> \
  <gpu01本机的slurm.conf.generated路径> \
  --apply
```

前两个参数是位置参数，顺序不能交换；两个文件必须已经存在于 `gpu01`。如果传入不存在的 `/secure/temp/...`，脚本只会输出 usage。

## 5. 对比两台机器

两台机器分别执行：

```bash
slurmd -V
stat -fc %T /sys/fs/cgroup
sha256sum \
  /etc/slurm/slurm.conf \
  /etc/slurm/cgroup.conf \
  /etc/slurm/gres.conf
sudo sha256sum /etc/munge/munge.key
```

三份 Slurm 配置和 Munge key 的 checksum 必须分别一致。不要发送 Munge key 内容。

在 `mgmt01`：

```bash
scontrol ping
sinfo -N -l
sinfo -o '%P|%N|%T|%c|%m|%G'
scontrol show nodes
squeue
```

期望两台节点：

- 状态为 `idle`；
- GRES 包含 `gpu:1`；
- 地址分别是 `.202`、`.215`；
- CPU 和内存与各自配置一致。

## 6. cgroup v2 和 GPU 隔离检查

两台机器分别检查：

```bash
stat -fc %T /sys/fs/cgroup
sudo slurmd -G
journalctl -u slurmd -b --no-pager | \
  grep -Ei 'cgroup|gres|gpu|error|fatal'
```

`config/slurm/cgroup.conf` 当前启用：

```ini
CgroupPlugin=autodetect
ConstrainCores=yes
ConstrainRAMSpace=yes
ConstrainDevices=yes
ConstrainSwapSpace=yes
```

`gres.conf` 使用 NVML 自动探测 `/dev/nvidia0`。如果 `slurmd -G` 报 GPU 数量或设备不匹配，先修正 NVIDIA 驱动和 `gres.conf`，不要直接把节点强制 RESUME。

## 7. 两节点 GPU smoke test

只在 `mgmt01` 执行：

```bash
for node in mgmt01 gpu01; do
  echo "[$node]"
  srun \
    --partition=debug \
    --nodes=1 \
    --nodelist="$node" \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=1G \
    --gres=gpu:1 \
    --time=00:02:00 \
    /opt/robot-platform/train-venv/bin/python -c \
    'import os, socket, torch, lerobot; print("host=", socket.gethostname()); print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES")); print("cuda=", torch.cuda.is_available()); print("gpu=", torch.cuda.get_device_name(0)); print("lerobot=ok")'
done
```

两个任务都必须：

- 在指定节点运行；
- 只看到分配的 GPU；
- `cuda=True`；
- 成功导入 LeRobot。

再做并行占用测试：

```bash
srun --partition=debug --nodelist=mgmt01 --gres=gpu:1 --time=00:02:00 \
  bash -c 'nvidia-smi -L; sleep 20' &
srun --partition=debug --nodelist=gpu01 --gres=gpu:1 --time=00:02:00 \
  bash -c 'nvidia-smi -L; sleep 20' &
wait
```

## 8. 常见非正常状态

| 现象 | 优先检查 |
|---|---|
| `DOWN` / `NOT_RESPONDING` | `slurmd` 服务、6818、防火墙、主机名、时间 |
| `INVAL` | `slurmd -C` 与 NodeName 行、RealMemory、CPU 拓扑 |
| `Invalid generic resource` | `sudo slurmd -G`、`gres.conf`、NVML、`/dev/nvidia0` |
| Munge 认证失败 | key checksum、`0400 munge:munge`、时间同步 |
| cgroup 插件加载失败 | Slurm 版本、`cgroup2fs`、`cgroup_v2.so` |
| 作业停在 `PENDING` | `scontrol show job <id>` 的 `Reason` |
| 远端作业提示无法进入提交目录 | 确认作业脚本和输出目录位于共享绝对路径；详见排障文档 |

日志：

```bash
# mgmt01
journalctl -u slurmctld -u slurmd -u munge -n 150 --no-pager

# gpu01
journalctl -u slurmd -u munge -n 150 --no-pager
```

## 9. Slurm 完成后的下一步

只有本页所有检查通过后，才在 `mgmt01` 安装 leLab：

```bash
sudo ./scripts/15-install-lelab-platform.sh --apply
```

接着完成 [leLab SSH 探测和 API 验收](07-lelab-cluster-web.md)，最后用 NAS 中的小型 LeRobot 数据集提交第一条短任务。
