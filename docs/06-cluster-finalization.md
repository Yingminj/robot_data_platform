# Slurm 集群收尾与验收

本文是当前集群（`mgmt01` + `gpu01` + `gpu02` + `gpu03`）的权威配置顺序。所有“在管理机执行”的命令都从 `mgmt01` 仓库根目录运行；“在 Worker 执行”的命令都从对应 Worker 的仓库根目录运行。

**向已经运行的集群加节点不要走本文流程**，看[向已有集群增加 GPU 节点](09-add-gpu-node.md)。

## 0. 完成条件

进入本阶段前：

- `mgmt01` 已执行 `10` 和 `25`；
- 每台 GPU Worker 已执行 `20` 和 `25`；
- 所有机器已安装相同的 Slurm 26.05.2；
- 所有机器都使用 cgroup v2；
- 所有机器主机名、时间和 `/etc/hosts` 正确；
- QNAP 在相同绝对路径 `/mnt/robot_platform` 挂载，且白名单包含全部节点 IP；
- `robot-train` 和 `robotdata` 的数字 UID/GID 在所有机器上一致。

在**每台**机器上快速核对：

```bash
hostname -s
slurmd -V
stat -fc %T /sys/fs/cgroup
id robot-train
getent group robotdata
getent hosts mgmt01 gpu01 gpu02 gpu03
findmnt /mnt/robot_platform
```

`hostname -s` 的输出必须与将要写入 `nodes.conf` 的 NodeName 完全一致。

此时 Worker 上执行 `sbatch --version` 会报 `DNS SRV lookup failed`，**这是正常的**：本机还没有 `/etc/slurm/slurm.conf`，新版 Slurm 回退到本集群不使用的 configless 发现。用 `/usr/sbin/slurmd -V` 查版本不受影响，装完配置后该报错消失。

## 1. 收集节点资源

**每台**主机分别执行：

```bash
sudo slurmd -C
```

在 `mgmt01` 复制模板：

```bash
cp config/slurm/nodes.conf.example config/slurm/nodes.conf
editor config/slurm/nodes.conf
```

每行都应：

- 使用固定 `NodeName=`，与该机 `hostname -s` 相同；
- 使用对应 `NodeAddr`；
- 保留**该机自己** `slurmd -C` 的 CPU 拓扑；
- 设置不高于 `slurmd -C` 实测值的 `RealMemory`；
- 追加 `Gres=gpu:1 State=UNKNOWN`。

当前结构示例：

```ini
NodeName=mgmt01 NodeAddr=192.168.100.202 CPUs=... RealMemory=61912 Gres=gpu:1 State=UNKNOWN
NodeName=gpu01 NodeAddr=192.168.100.215 CPUs=... RealMemory=61920 Gres=gpu:1 State=UNKNOWN
NodeName=gpu02 NodeAddr=192.168.100.216 CPUs=... RealMemory=61919 Gres=gpu:1 State=UNKNOWN
NodeName=gpu03 NodeAddr=192.168.100.217 CPUs=... RealMemory=61914 Gres=gpu:1 State=UNKNOWN
```

不要保留 `FILL_ME`。**各机 `RealMemory` 通常有几 MB 差异，不要为了整齐把一台的值复制给另一台**：填报值高于实测值会让节点进入 `INVAL`。

行数必须与 `config/site.env` 中 `GPU_NODE_NAMES` 的节点数相同，否则渲染脚本报：

```text
expected 4 nodes, found 2
```

## 2. 渲染并审查配置

在 `mgmt01`：

```bash
./scripts/cluster/render-slurm-config.sh
sed -n '1,240p' config/slurm/slurm.conf.generated
```

确认：

```bash
! grep -E 'FILL_ME|@@' config/slurm/slurm.conf.generated
grep -c '^NodeName=' config/slurm/slurm.conf.generated
grep '^PartitionName=' config/slurm/slurm.conf.generated
```

预期节点数与 `GPU_NODE_NAMES` 相同，且 `debug`、`train`、`eval` 三个分区都列出全部节点：

```ini
PartitionName=debug Nodes=mgmt01,gpu01,gpu02,gpu03 Default=YES MaxTime=01:00:00 State=UP
```

## 3. 安装 Controller 和 mgmt01 Worker

在 `mgmt01`：

```bash
sudo ./scripts/cluster/install-controller-config.sh \
  config/slurm/slurm.conf.generated \
  --apply
```

> **参数顺序与编号脚本相反。** 这里配置文件路径是第一个参数，`--apply` 是第二个；而 `10`/`20`/`25` 等编号脚本把 `--apply` 放在第一位。只写 `sudo ./scripts/cluster/install-controller-config.sh --apply` 会把 `--apply` 当成配置文件名，然后输出与不带任何参数时相同的提示：
>
> ```text
> This script changes the host. Re-run it with --apply after reviewing config/site.env.
> ```
>
> 看到这句话不代表 `--apply` 写错了位置以外的问题，补上配置文件路径即可。

该脚本会重启 `slurmctld` **和 mgmt01 本机的 `slurmd`**，执行前先看 `squeue`：本机正在运行的作业会被中断。

它会把以下文件装入 `/etc/slurm`：

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

## 4. 安装每台 Worker

**每一台** Worker 都必须获得与管理机完全相同的：

- `/etc/munge/munge.key`；
- 生成的 `slurm.conf`；
- 仓库中的 `cgroup.conf` 和 `gres.conf`。

> **拓扑变化时，已有节点也要重新分发 `slurm.conf`。** Slurm 要求全集群配置逐字节一致。加节点后只更新新节点，已有节点手里仍是不含新节点的旧配置，控制器重新加载后它会失效。加节点的完整流程见[向已有集群增加 GPU 节点](09-add-gpu-node.md)。

Munge 密钥只能经管理员控制的临时通道传输，不能放入 Git、NAS 公共目录或聊天记录。具体的 `scp` 暂存命令见 [GPU Worker 安装：接收并安装集群配置](03-gpu-node.md#7-接收并安装集群配置)。

在每台 Worker 上的调用形式是：

```bash
sudo ./scripts/cluster/install-worker-config.sh \
  <该Worker本机的munge.key路径> \
  <该Worker本机的slurm.conf.generated路径> \
  --apply
```

前两个参数是位置参数，顺序不能交换；两个文件必须已经存在于**执行命令的那台机器**上。如果传入不存在的 `/secure/temp/...`，脚本只会输出 usage。

安装过程中 `slurmd -G` 会输出一条 GRES 类型提示，**这是正常的**：

```text
gres/gpu: _normalize_sys_gres_types: Could not find an unused configuration record
with a GRES type that is a substring of system device `nvidia_geforce_rtx_4090`.
Setting system GRES type to NULL
```

`gres.conf` 声明的是不带型号的 `Name=gpu`，NVML 报告的设备型号是 `nvidia_geforce_rtx_4090`，于是 Slurm 把类型置为 NULL，与 `nodes.conf` 中同样不带型号的 `Gres=gpu:1` 一致。紧随其后的这行才是结论：

```text
Gres Name=gpu Type=(null) Count=1 Index=0 File=/dev/nvidia0 Flags=HAS_FILE,ENV_NVML
```

只有需要按型号申请（`--gres=gpu:rtx4090:1`）时才要改 `gres.conf`。

## 5. 对比所有机器

每台机器分别执行：

```bash
slurmd -V
stat -fc %T /sys/fs/cgroup
sha256sum \
  /etc/slurm/slurm.conf \
  /etc/slurm/cgroup.conf \
  /etc/slurm/gres.conf
sudo sha256sum /etc/munge/munge.key
```

三份 Slurm 配置和 Munge key 的 checksum 必须在**所有**机器上分别一致。**只比对 checksum，不要发送 Munge key 内容。**

在 `mgmt01`：

```bash
scontrol ping
sinfo -N -l
sinfo -o '%P|%N|%T|%c|%m|%G'
scontrol show nodes
squeue
```

期望每个节点：

- 在 `debug`、`train`、`eval` 三个分区中都是 `idle`；
- GRES 包含 `gpu:1`；
- 地址与 `nodes.conf` 中的 `NodeAddr` 一致；
- CPU 和内存与各自配置一致。

## 6. cgroup v2 和 GPU 隔离检查

每台机器分别检查：

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

## 7. 逐节点 GPU smoke test

只在 `mgmt01` 执行：

```bash
for node in mgmt01 gpu01 gpu02 gpu03; do
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

每个任务都必须：

- 在指定节点运行；
- 只看到分配的 GPU；
- `cuda=True`；
- 成功导入 LeRobot。

再确认各节点是**不同的物理机**：

```bash
for node in mgmt01 gpu01 gpu02 gpu03; do
  echo -n "$node: "
  srun --partition=debug --nodelist="$node" --gres=gpu:1 --time=00:02:00 nvidia-smi -L
done
```

返回的 GPU UUID 必须两两不同。**UUID 重复说明 `NodeAddr` 写错**，两个 NodeName 指向了同一台物理机——这时 `sinfo` 一切正常，只有 UUID 能发现问题。

再做并行占用测试：

```bash
for node in mgmt01 gpu01 gpu02 gpu03; do
  srun --partition=debug --nodelist="$node" --gres=gpu:1 --time=00:02:00 \
    bash -c 'nvidia-smi -L; sleep 20' &
done
wait
```

过程中的这条警告可以忽略：

```text
error: couldn't chdir to `/home/kewei/YING/robot_data_platform': No such file or directory: going to /tmp instead
```

`srun` 会把提交端的当前目录传给远端，而该仓库路径只存在于 `mgmt01`。leLab 提交的作业使用绝对路径，不受影响；要消除警告可加 `--chdir=/tmp`。

## 8. 常见非正常状态

| 现象 | 优先检查 |
|---|---|
| `DOWN` / `NOT_RESPONDING` | `slurmd` 服务、6818、防火墙、主机名、时间 |
| `INVAL` | `slurmd -C` 与 NodeName 行、RealMemory、CPU 拓扑 |
| `Invalid generic resource` | `sudo slurmd -G`、`gres.conf`、NVML、`/dev/nvidia0` |
| Munge 认证失败 | key checksum、`0400 munge:munge`、时间同步 |
| cgroup 插件加载失败 | Slurm 版本、`cgroup2fs`、`cgroup_v2.so` |
| 作业停在 `PENDING` | `scontrol show job <id>` 的 `Reason` |
| 远端作业提示无法进入提交目录 | 属于正常警告，见第 7 节 |
| 加节点后已有节点变 `DOWN` | 已有节点的 `slurm.conf` 未同步更新，见第 4 节 |

以下三类输出**不是**故障，不要据此重装：

| 输出 | 出现时机 | 说明 |
|---|---|---|
| `DNS SRV lookup failed` | Worker 装配置前执行 `sbatch --version` | 本机尚无 `slurm.conf`，回退到未使用的 configless 发现 |
| `_normalize_sys_gres_types ... Setting system GRES type to NULL` | 每次 `slurmd -G` | `gres.conf` 用不带型号的 `Name=gpu`，与 `Gres=gpu:1` 一致 |
| `couldn't chdir to ...: going to /tmp instead` | `srun` 从 `mgmt01` 仓库目录提交 | 提交端目录不存在于 Worker，作业本身不受影响 |

日志：

```bash
# mgmt01
journalctl -u slurmctld -u slurmd -u munge -n 150 --no-pager

# 每台 Worker
journalctl -u slurmd -u munge -n 150 --no-pager
```

## 9. Slurm 完成后的下一步

只有本页所有检查通过后，才在 `mgmt01` 安装 leLab：

```bash
sudo ./scripts/15-install-lelab-platform.sh --apply
```

接着完成 [leLab SSH 探测和 API 验收](07-lelab-cluster-web.md)，最后用 NAS 中的小型 LeRobot 数据集提交第一条短任务。
