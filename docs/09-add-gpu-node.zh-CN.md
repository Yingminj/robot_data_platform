# 向已有集群增加 GPU 节点

[English](09-add-gpu-node.md) | **简体中文**

本文用于**已经在运行的集群**增加一台 GPU Worker，例如从 `mgmt01 + gpu01` 扩到 `mgmt01 + gpu01 + gpu02 + gpu03`。

首次部署不要看本文，看 [README](../README.zh-CN.md) 的完整顺序。

## 0. 本文要解决的三个易错点

初次扩容最容易在这三处失败，且失败现象都不指向真正原因：

1. **拓扑字段必须在所有主机上同时更新。** `GPU_NODE_NAMES` 只在新机器上改，`05-configure-hosts.sh` 和渲染脚本都会通过，但集群行为不一致。
2. **`slurm.conf` 必须全集群一致。** 只把新配置发给新节点，**已有的 `gpu01` 会因为持有旧的、不含新节点的配置而失效**。这一步最容易漏。
3. **`/etc/hosts` 托管块不能增量修改。** `05-configure-hosts.sh` 发现已有块与目标不一致时会直接停止，必须先手工删除旧块。

全流程顺序（不要跳步）：

```text
1. 新节点装系统、驱动、Python 3.12
2. 所有主机同步更新 site.env 的 GPU_NODE_NAMES / GPU_NODE_IPS
3. 所有主机重建 /etc/hosts 托管块
4. 新节点执行 20 + 25，安装 Slurm 26.05.2
5. 新节点 slurmd -C，把结果填入 mgmt01 的 nodes.conf
6. mgmt01 渲染新的 slurm.conf.generated
7. mgmt01 安装 Controller 配置
8. 分发到「所有」Worker，包括已有的 gpu01
9. 验证 sinfo 中全部节点 idle
10. 配置 leLab 的 SSH 探测和节点映射
11. 验收
```

## 1. 确定新节点信息

本文以新增两台为例：

| Slurm NodeName | IP | SSH 登录账号 |
|---|---|---|
| `gpu02` | `192.168.100.216` | `yang` |
| `gpu03` | `192.168.100.217` | `snorlax` |

**SSH 账号不必与 Slurm NodeName 相同，也不必在各节点之间相同。** 上表中 `gpu02` 用 `yang`、`gpu03` 用 `snorlax`，这是允许的：Slurm 只认 NodeName，leLab 的 `LELAB_CLUSTER_NODES` 负责把 NodeName 映射到 SSH 目标。

在新节点上：

```bash
sudo hostnamectl set-hostname gpu02
```

改完重新登录，确认 `hostname -s` 与将要写入 `nodes.conf` 的 NodeName 完全一致。

## 2. 在所有主机同步 site.env

**这一步要在 `mgmt01`、`gpu01` 和每台新节点上分别执行**，改成完全相同的值：

```bash
editor config/site.env
```

```bash
GPU_NODE_NAMES="mgmt01 gpu01 gpu02 gpu03"
GPU_NODE_IPS="192.168.100.202 192.168.100.215 192.168.100.216 192.168.100.217"
```

两个列表按位置一一对应，长度必须相同，否则 `05-configure-hosts.sh` 会报 `GPU_NODE_NAMES and GPU_NODE_IPS have different lengths`。

新节点如果是全新的仓库副本，先 `cp config/site.env.example config/site.env`，再核对 [README 中必须一致的字段清单](../README.zh-CN.md#0-准备统一配置)。`DATA_GID` 和 `TRAIN_UID` 尤其重要：数字不一致会导致作业在新节点上写不了 NAS。

## 3. 重建 /etc/hosts 托管块

`05-configure-hosts.sh` **不会增量修改已有的托管块**。它比较现有块与目标块，不一致就停止：

```text
existing managed /etc/hosts block differs; review it manually
```

这是有意设计，防止脚本覆盖人工调整过的内容。扩容时的正确做法是先删除旧块，再重新生成。在**每台**主机上：

```bash
sudo cp -a /etc/hosts /etc/hosts.bak.$(date +%Y%m%d%H%M%S)
sudo sed -i '/^# BEGIN robot-platform managed hosts$/,/^# END robot-platform managed hosts$/d' /etc/hosts
sudo ./scripts/05-configure-hosts.sh --apply
getent hosts mgmt01 gpu01 gpu02 gpu03
```

最后一条必须四行全部解析成功。

## 4. 新节点安装基础组件和训练环境

在新节点上，与 [GPU 节点安装](03-gpu-node.zh-CN.md) 完全相同：

```bash
./scripts/00-audit-host.sh gpu
sudo ./scripts/20-install-gpu-node.sh --apply
sudo ./scripts/25-install-training-environment.sh --apply
```

然后按 [Slurm 26.05.2 安装](Slurm-INSTALL.zh-CN.md) 安装与现有节点**完全相同**的包版本：

```bash
/usr/sbin/slurmd -V     # 期望 slurm 26.05.2
stat -fc %T /sys/fs/cgroup   # 期望 cgroup2fs
```

此时执行 `sbatch --version` 会报错，**这是正常的**：

```text
sbatch: error: resolve_ctls_from_dns_srv: res_nsearch error: Unknown host
sbatch: error: fetch_config: DNS SRV lookup failed
sbatch: fatal: Could not establish a configuration source
```

新版 Slurm 在打印版本前会先尝试加载配置，本机此时还没有 `/etc/slurm/slurm.conf`，于是回退到本集群不使用的 configless DNS SRV 发现。第 8 步装入配置后该报错消失。用 `/usr/sbin/slurmd -V` 检查版本不受影响。

## 5. 收集新节点硬件参数

在**每台新节点**上：

```bash
sudo slurmd -C
```

把输出中的 `CPUs`、`Boards`、`SocketsPerBoard`、`CoresPerSocket`、`ThreadsPerCore`、`RealMemory` 抄到 `mgmt01` 的 `config/slurm/nodes.conf`，**追加**新行，不要改动已有行：

```ini
NodeName=gpu02 NodeAddr=192.168.100.216 CPUs=32 Boards=1 SocketsPerBoard=1 CoresPerSocket=16 ThreadsPerCore=2 RealMemory=61919 Gres=gpu:1 State=UNKNOWN
NodeName=gpu03 NodeAddr=192.168.100.217 CPUs=32 Boards=1 SocketsPerBoard=1 CoresPerSocket=16 ThreadsPerCore=2 RealMemory=61914 Gres=gpu:1 State=UNKNOWN
```

各机 `RealMemory` 通常有几 MB 差异，**不要为了整齐把一台的值复制给另一台**。填报值高于 `slurmd -C` 实测值会让节点进入 `INVAL`。

`nodes.conf` 中的行数必须与 `GPU_NODE_NAMES` 的节点数相同，渲染脚本会校验：

```text
expected 4 nodes, found 2
```

## 6. 渲染并审查配置

在 `mgmt01`：

```bash
./scripts/cluster/render-slurm-config.sh
grep -c '^NodeName=' config/slurm/slurm.conf.generated    # 期望 4
grep '^PartitionName=' config/slurm/slurm.conf.generated
```

三个分区都应列出全部节点：

```ini
PartitionName=debug Nodes=mgmt01,gpu01,gpu02,gpu03 Default=YES MaxTime=01:00:00 State=UP
PartitionName=train Nodes=mgmt01,gpu01,gpu02,gpu03 MaxTime=7-00:00:00 State=UP
PartitionName=eval Nodes=mgmt01,gpu01,gpu02,gpu03 MaxTime=1-00:00:00 State=UP
```

渲染脚本遇到 `FILL_ME` 会直接失败，这是防止未填硬件参数就部署。

## 7. 安装 Controller 配置

在 `mgmt01`：

```bash
sudo ./scripts/cluster/install-controller-config.sh \
  config/slurm/slurm.conf.generated \
  --apply
```

**配置文件路径是第一个参数，`--apply` 是第二个参数。** 这与编号脚本（`10`/`20`/`25` 等把 `--apply` 放在第一位）不同。只写 `sudo ./scripts/cluster/install-controller-config.sh --apply` 会把 `--apply` 当成配置文件名，然后报同一句提示：

```text
This script changes the host. Re-run it with --apply after reviewing config/site.env.
```

该脚本会重启 `slurmctld` **和 mgmt01 本机的 `slurmd`**。执行前先看 `squeue`，本机正在跑的作业会被中断。

## 8. 分发到所有 Worker（含已有节点）

> **本步最容易漏掉已有的 `gpu01`。** Slurm 要求全集群 `slurm.conf` 完全一致。`gpu01` 手里还是旧的两节点配置，控制器重新加载后它会失效。

Munge 密钥对已有节点没有变化，只有**新节点**需要密钥；`slurm.conf` 则**所有** Worker 都要更新。

在 `mgmt01` 暂存并分发：

```bash
stage_dir="$(mktemp -d)"
sudo install -o "$USER" -g "$(id -gn)" -m 0600 \
  /etc/munge/munge.key "${stage_dir:?}/munge.key"
install -m 0644 \
  config/slurm/slurm.conf.generated "${stage_dir:?}/slurm.conf.generated"

for target in snorlax@192.168.100.215 yang@192.168.100.216 snorlax@192.168.100.217; do
  ssh "$target" 'install -d -m 0700 ~/robot-platform-secure'
  scp "${stage_dir:?}"/munge.key "${stage_dir:?}"/slurm.conf.generated \
    "$target:~/robot-platform-secure/"
done

shred -u "${stage_dir:?}/munge.key"
rm -f "${stage_dir:?}/slurm.conf.generated"
rmdir "${stage_dir:?}"
```

`${stage_dir:?}` 的 `:?` 不能省略。如果只复制粘贴了后半段命令而漏掉 `stage_dir="$(mktemp -d)"`，变量为空，`"$stage_dir/munge.key"` 会变成 `/munge.key`，而前面带 `sudo` 的那条会**静默成功**，把集群唯一的认证密钥写到文件系统根目录。加上 `:?` 后 bash 会立即报 `stage_dir: parameter null or not set` 并终止。

如果已经发生，删除它：

```bash
sudo shred -u /munge.key
```

在**每台** Worker 上安装：

```bash
sudo ./scripts/cluster/install-worker-config.sh \
  ~/robot-platform-secure/munge.key \
  ~/robot-platform-secure/slurm.conf.generated \
  --apply
```

装完清理：

```bash
shred -u ~/robot-platform-secure/munge.key
rm -f ~/robot-platform-secure/slurm.conf.generated
rmdir ~/robot-platform-secure
```

`slurmd -G` 会输出一条 GRES 类型提示，**这是正常的**：

```text
gres/gpu: _normalize_sys_gres_types: Could not find an unused configuration record
with a GRES type that is a substring of system device `nvidia_geforce_rtx_4090`.
Setting system GRES type to NULL
```

`gres.conf` 声明的是不带型号的 `Name=gpu`，NVML 报告的设备型号是 `nvidia_geforce_rtx_4090`，于是 Slurm 把类型置为 NULL。这与 `nodes.conf` 中同样不带型号的 `Gres=gpu:1` 一致。紧随其后的这行才是结论：

```text
Gres Name=gpu Type=(null) Count=1 Index=0 File=/dev/nvidia0 Flags=HAS_FILE,ENV_NVML
```

只有需要按型号申请（`--gres=gpu:rtx4090:1`）时才要改 `gres.conf`。

## 9. 校验全集群一致性

在每台主机执行，四份输出必须两两相同：

```bash
sha256sum /etc/slurm/slurm.conf /etc/slurm/cgroup.conf /etc/slurm/gres.conf
sudo sha256sum /etc/munge/munge.key
```

**不要把 Munge key 的内容贴到任何地方，只比对 checksum。**

在 `mgmt01`：

```bash
scontrol ping
sinfo -N -l
```

期望每个节点在 `debug`、`train`、`eval` 三个分区中都是 `idle`。

逐节点 smoke test：

```bash
for node in mgmt01 gpu01 gpu02 gpu03; do
  echo "[$node]"
  srun --partition=debug --nodelist="$node" --gres=gpu:1 --time=00:02:00 \
    nvidia-smi -L
done
```

每台应返回**各不相同**的 GPU UUID。UUID 重复说明 `NodeAddr` 写错，两个 NodeName 指向了同一台物理机。

过程中的这条警告可以忽略：

```text
error: couldn't chdir to `/home/kewei/YING/robot_data_platform': No such file or directory: going to /tmp instead
```

`srun` 会把提交端的当前目录传给远端，而该路径只存在于 `mgmt01`。leLab 提交的作业使用绝对路径，不受影响。要消除警告可加 `--chdir=/tmp`。

## 10. 配置 leLab

leLab 只装在 `mgmt01`，扩容时不需要重装，只需三件事。

### 10.1 SSH 公钥

把 `mgmt01` 的 leLab 公钥装到每台新节点的登录账号：

```bash
# 在 mgmt01 查看
cat /etc/robot-platform/lelab_ssh_key.pub
```

在新节点上，以对应账号（`gpu02` 是 `yang`，`gpu03` 是 `snorlax`）执行：

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
editor ~/.ssh/authorized_keys      # 粘贴上面那一行
chmod 600 ~/.ssh/authorized_keys
ssh-keygen -lf ~/.ssh/authorized_keys
```

公钥是很长的一行。**终端粘贴时如果被折行，会得到一个永远匹配不上的密钥，且没有任何报错。** 用最后一条命令确认新增的密钥能被正确解析。

也可以用 `ssh-copy-id`，注意 `-f` 必须保留（否则它会尝试读取只有 `robot-train` 能读的同名私钥）：

```bash
ssh-copy-id -f -i /etc/robot-platform/lelab_ssh_key.pub yang@192.168.100.216
```

### 10.2 host key

发起连接的是 `mgmt01` 上的 `robot-train`，所以要装到它的 `known_hosts`。**逐台扫描并人工核对指纹，不要用 `-H`**：加了 `-H` 的输出主机名是哈希过的，无法与节点上看到的指纹对应，也无法去重。

在 `mgmt01` 扫描：

```bash
ssh-keyscan -t ed25519 192.168.100.216 2>/dev/null > /tmp/kh216
ssh-keyscan -t ed25519 192.168.100.217 2>/dev/null > /tmp/kh217
ssh-keygen -lf /tmp/kh216
ssh-keygen -lf /tmp/kh217
```

在**每台新节点的控制台**查看真实指纹：

```bash
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
```

**只比对 `SHA256:` 字段。** 末尾注释必然不同：扫描结果标注的是被查询的 IP，节点本机标注的是密钥注释（如 `root@gpu02`）。指纹不一致就停止，说明应答该 IP 的不是你以为的机器。

核对通过后追加（**不要覆盖**已有内容，`gpu01` 的条目还在里面）：

```bash
sudo -u robot-train tee -a /home/robot-train/.ssh/known_hosts < /tmp/kh216 >/dev/null
sudo -u robot-train tee -a /home/robot-train/.ssh/known_hosts < /tmp/kh217 >/dev/null
rm -f /tmp/kh216 /tmp/kh217
```

这里用 `sudo -u robot-train tee -a` 而不是 `sudo ... >>`：重定向由当前 shell 执行，身份是你自己而不是 `robot-train`，会生成属主错误的文件，SSH 随后直接忽略它。

确认写入：

```bash
sudo -u robot-train ssh-keygen -F 192.168.100.216 -f /home/robot-train/.ssh/known_hosts
```

### 10.3 节点映射

编辑活动配置（**不是**仓库里的 `config/lelab.env.example`）：

```bash
sudo editor /etc/robot-platform/lelab.env
```

```bash
LELAB_CLUSTER_NODES=mgmt01=192.168.100.202,gpu01=snorlax@192.168.100.215,gpu02=yang@192.168.100.216,gpu03=snorlax@192.168.100.217
```

这是很长的一行，**用编辑器改，不要用 `sed` 一行命令**。终端粘贴长命令时会在任意位置插入换行，`sed` 会看到不完整的表达式并报错：

```text
sed: -e expression #1, char 85: unterminated `s' command
```

必须用命令行时，把值拆成短片段拼装，每段都不会被折行：

```bash
N='mgmt01=192.168.100.202'
N="$N,gpu01=snorlax@192.168.100.215"
N="$N,gpu02=yang@192.168.100.216"
N="$N,gpu03=snorlax@192.168.100.217"
echo "$N"
sudo sed -i "s|^LELAB_CLUSTER_NODES=.*|LELAB_CLUSTER_NODES=$N|" /etc/robot-platform/lelab.env
```

执行 `sed` 前先看 `echo "$N"` 是否输出完整的一行。注意 `sed` 脚本必须用双引号，`$N` 需要展开。

该文件由 systemd 以 `EnvironmentFile` 读取，**只在服务重启时生效**：

```bash
sudo systemctl restart lelab-platform
```

## 11. 新节点的运行期前置条件

这几项在提交作业前不会报错，作业调度到新节点后才失败。在**每台新节点**确认：

```bash
# LELAB_JOB_CACHE_ROOT 必须存在且 robot-train 可写。
# Slurm 把 HOME 指向 robot-train 的家目录，而它在 Worker 上并不存在。
sudo install -d -o robot-train -g robotdata -m 0750 /var/lib/robot-platform/cache
sudo -u robot-train test -w /var/lib/robot-platform/cache && echo cache OK

# NAS 必须以相同绝对路径挂载
findmnt /mnt/robot_platform
sudo -u robot-train test -r /mnt/robot_platform/datasets && echo datasets OK
sudo -u robot-train test -w /mnt/robot_platform/jobs && echo jobs OK

# 训练环境
sudo -u robot-train /opt/robot-platform/train-venv/bin/python -c \
  'import torch, lerobot; print(torch.cuda.is_available())'

# 视频解码后端。torchcodec 自带 libtorchcodec_core*.so，但 libav* 来自系统 ffmpeg，
# 且没有任何 pip 依赖会装它。缺失时前面所有检查都通过，直到训练第一个 batch 才报
# "Could not load libtorchcodec"（见[排障 12b-2](08-troubleshooting.zh-CN.md)）。
sudo -u robot-train /opt/robot-platform/train-venv/bin/python -c \
  'from torchcodec.decoders import VideoDecoder; print("decoder OK")'
```

`25-install-training-environment.sh` 现在会安装 `ffmpeg` 并在安装末尾做这项导入检查，
但**早于该改动装好的节点需要手工补装**：

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

同时确认 QNAP 的 NFS 白名单已包含新节点 IP，否则挂载会失败或只读。

## 12. 验收

在 `mgmt01`：

```bash
sinfo -N -l
curl --noproxy '*' -fsS http://127.0.0.1:8000/cluster/status | \
  jq -c '.nodes[] | {name,reachable,slurm_state,eligible,memory_free_mb,reason}'
```

每个新节点都应是：

```json
{"name":"gpu02","reachable":true,"slurm_state":"idle","eligible":true,"memory_free_mb":47752,"reason":null}
```

常见的两种未完成状态：

| `reason` | 含义 |
|---|---|
| `Permission denied (publickey,password)` | host key 已通过，公钥未装入远端 `authorized_keys`（见 10.1） |
| `Host key verification failed` | 公钥无关，`robot-train` 的 known_hosts 缺该节点（见 10.2） |
| `GPU has a compute process outside or inside Slurm` | 该节点 GPU 上有 Slurm 外的 CUDA 进程，见[排障 10](08-troubleshooting.zh-CN.md) |

最后从 leLab 页面把节点选为 `auto`，提交一条很短的训练任务，确认它落到新节点并生成 `slurm.out` 与 checkpoint。

## 13. 需要同步更新的文件清单

扩容后应一并检查的内容：

| 文件 | 位置 | 是否纳入 Git |
|---|---|---|
| `config/site.env` | 所有主机 | 否，被忽略 |
| `config/slurm/nodes.conf` | 仅 `mgmt01` | 否，被忽略 |
| `config/slurm/slurm.conf.generated` | 仅 `mgmt01`，由脚本生成 | 否，被忽略 |
| `/etc/slurm/slurm.conf` | 所有主机，必须一致 | 否 |
| `/etc/hosts` 托管块 | 所有主机 | 否 |
| `/etc/robot-platform/lelab.env` | 仅 `mgmt01` | 否 |
| `/home/robot-train/.ssh/known_hosts` | 仅 `mgmt01` | 否 |
| `config/site.env.example` | 仓库 | 是 |
| `config/lelab.env.example` | 仓库 | 是 |
| `config/slurm/nodes.conf.example` | 仓库 | 是 |
| `README.md` 拓扑表 | 仓库 | 是 |

前七项是本地活动配置，**换机器或重装后不会自动恢复**，扩容记录只能靠三个 `.example` 模板和 README 保留。
