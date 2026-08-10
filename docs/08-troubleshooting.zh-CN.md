# 安装与运行排障

[English](08-troubleshooting.md) | **简体中文**

先根据故障所在层级定位，不要一看到 Web 异常就重装全部组件：

```text
主机/网络
→ NFS、账号、时间
→ Munge
→ Slurm Controller/Worker
→ 统一训练环境
→ leLab systemd
→ leLab SSH GPU 探测
→ 数据集与训练任务
```

## 0. 这些输出不是故障

排查前先排除三条**正常但看起来像报错**的输出，它们曾多次导致无谓的重装：

| 输出 | 出现时机 | 说明 |
|---|---|---|
| `DNS SRV lookup failed` / `Could not establish a configuration source` | Worker 装 Slurm 配置**之前**执行 `sbatch --version` | 新版 Slurm 打印版本前先加载配置，本机尚无 `/etc/slurm/slurm.conf`，回退到本集群不使用的 configless 发现。装完配置即消失。查版本用 `/usr/sbin/slurmd -V` |
| `_normalize_sys_gres_types ... Setting system GRES type to NULL` | 每次 `slurmd -G` | `gres.conf` 用不带型号的 `Name=gpu`，NVML 报 `nvidia_geforce_rtx_4090`，于是类型置 NULL，与 `Gres=gpu:1` 一致。下一行 `Gres Name=gpu Type=(null) Count=1` 才是结论 |
| `couldn't chdir to ...: going to /tmp instead` | 从 `mgmt01` 仓库目录 `srun` | 提交端当前目录不存在于 Worker。leLab 作业用绝对路径，不受影响。要消除可加 `--chdir=/tmp` |

## 1. 先收集最小状态

在 `mgmt01`：

```bash
hostname -s
slurmd -V
stat -fc %T /sys/fs/cgroup
systemctl is-active munge slurmctld slurmd lelab-platform
scontrol ping
sinfo -N -l
squeue
findmnt /mnt/robot_platform
curl --noproxy '*' -fsS http://127.0.0.1:8000/cluster/status | \
  jq -c '.nodes[] | {name,reachable,slurm_state,eligible,reason}'
```

在**每台** Worker：

```bash
hostname -s
slurmd -V
stat -fc %T /sys/fs/cgroup
systemctl is-active munge slurmd
nvidia-smi
sudo slurmd -G
findmnt /mnt/robot_platform
sha256sum /etc/slurm/slurm.conf
```

最后一条的输出必须与 `mgmt01` 上完全相同。**全集群 `slurm.conf` 不一致是加节点后最常见的故障根因**，且现象（某个节点 `DOWN`）不指向配置。

## 2. Worker 安装脚本只输出 usage

报错：

```text
ERROR: usage: ./scripts/cluster/install-worker-config.sh \
<secure-copy-of-munge.key> <slurm.conf.generated> --apply
```

含义不是 `--apply` 换行错误，而是以下至少一项不成立：

- 第一个文件在当前 Worker 本机不存在或不可读；
- 第二个文件在当前 Worker 本机不存在或不可读；
- 第三个参数不是 `--apply`。

在 `gpu01` 检查：

```bash
sudo test -r /home/snorlax/robot-platform-secure/munge.key
sudo test -r /home/snorlax/robot-platform-secure/slurm.conf.generated
```

正确调用：

```bash
sudo ./scripts/cluster/install-worker-config.sh \
  /home/snorlax/robot-platform-secure/munge.key \
  /home/snorlax/robot-platform-secure/slurm.conf.generated \
  --apply
```

`/secure/temp/...` 是占位符，不是脚本自动创建的目录。

## 2b. Controller 安装脚本只输出 usage

```text
This script changes the host. Re-run it with --apply after reviewing config/site.env.
```

**`install-controller-config.sh` 的参数顺序与编号脚本相反**：配置文件路径是第一个参数，`--apply` 是第二个。只写 `--apply` 会被当成配置文件名，于是输出与不带参数时相同的提示。

```bash
# 错误：--apply 被当作配置文件名
sudo ./scripts/cluster/install-controller-config.sh --apply

# 正确
sudo ./scripts/cluster/install-controller-config.sh \
  config/slurm/slurm.conf.generated \
  --apply
```

`install-worker-config.sh` 同理：两个文件路径在前，`--apply` 在第三位。而 `10`/`20`/`25` 等编号脚本把 `--apply` 放在第一位。

## 2c. 渲染脚本报节点数不符

```text
expected 4 nodes, found 2
```

`config/slurm/nodes.conf` 的 `NodeName=` 行数与 `config/site.env` 中 `GPU_NODE_NAMES` 的节点数不一致。加节点时通常是改了 `site.env` 但忘了往 `nodes.conf` 追加新行。

相关的还有：

| 报错 | 原因 |
|---|---|
| `missing gpu02 in .../nodes.conf` | `nodes.conf` 缺该节点的行，或 NodeName 拼写与 `GPU_NODE_NAMES` 不一致 |
| `nodes.conf still contains FILL_ME placeholders` | 新节点的硬件参数还没填，先在该节点跑 `sudo slurmd -C` |
| `GPU_NODE_NAMES and GPU_NODE_IPS have different lengths` | 两个列表元素数不同，它们按位置一一对应 |

## 2d. 加节点后已有节点变成 DOWN

Slurm 要求全集群 `slurm.conf` 逐字节一致。只把新配置发给新节点时，已有节点手里仍是不含新节点的旧配置，控制器重新加载后它会失效。

```bash
# 在每台节点比对，必须全部相同
sha256sum /etc/slurm/slurm.conf
```

修复方式是把新渲染的 `slurm.conf.generated` 重新分发到**所有** Worker（Munge 密钥不变，无需重发），然后各自 `sudo systemctl restart slurmd`。完整流程见[向已有集群增加 GPU 节点](09-add-gpu-node.zh-CN.md)。

## 2e. /etc/hosts 托管块拒绝更新

```text
existing managed /etc/hosts block differs; review it manually
```

`05-configure-hosts.sh` **不做增量修改**：它比较现有托管块与目标块，不一致就停止，防止覆盖人工调整。拓扑变更时先删除旧块再重新生成，在**每台**主机执行：

```bash
sudo cp -a /etc/hosts /etc/hosts.bak.$(date +%Y%m%d%H%M%S)
sudo sed -i '/^# BEGIN robot-platform managed hosts$/,/^# END robot-platform managed hosts$/d' /etc/hosts
sudo ./scripts/05-configure-hosts.sh --apply
getent hosts mgmt01 gpu01 gpu02 gpu03
```

另一种停止情况是块**外**已存在同名条目，提示 `/etc/hosts already contains gpu02`，需要先手工清理那条。

## 3. Slurm cgroup v2 插件错误

先检查版本和系统：

```bash
slurmd -V
stat -fc %T /sys/fs/cgroup
find /usr/lib -type f -name cgroup_v2.so -print
```

当前期望：

```text
slurm 26.05.2
cgroup2fs
```

若仍是 Ubuntu 22.04 自带旧版，按 [Slurm 26.05.2 安装](Slurm-INSTALL.zh-CN.md)升级所有机器。不能只升级 Controller 或只升级部分 Worker。

## 4. Slurm 节点为 DOWN、INVAL 或 UNKNOWN

在 `mgmt01`：

```bash
scontrol show node <NodeName>
journalctl -u slurmctld -n 150 --no-pager
```

在故障 Worker：

```bash
sudo slurmd -C
sudo slurmd -G
journalctl -u slurmd -u munge -n 150 --no-pager
```

逐项对比：

1. NodeName 与 `hostname -s`；
2. `NodeAddr` 与固定 IP；
3. CPU 拓扑和 `RealMemory`（**填报值高于 `slurmd -C` 实测值会导致 `INVAL`**，各机内存通常差几 MB，不要互相复制）；
4. `slurm.conf`、`cgroup.conf`、`gres.conf` checksum 在**所有**节点一致；
5. Munge key checksum、`munge:munge` 和 `0400`；
6. 各机器时间；
7. TCP 6817/6818；
8. `sudo slurmd -G` 是否识别 `gpu:1`。

第 4 项是加节点后最常见的原因，见 2d 节。

只有原因已经修复时才恢复节点：

```bash
sudo scontrol update NodeName=gpu02 State=RESUME
```

## 5. NFS 已挂载但服务账号不能写

检查：

```bash
findmnt /mnt/robot_platform
sudo -u robot-train test -r /mnt/robot_platform/datasets
sudo -u robot-train test -w /mnt/robot_platform/jobs
sudo -u robot-ingest test -w /mnt/robot_platform/mlflow-artifacts
```

当前 QNAP 使用 `all_squash` 时，Linux 本地 `chown` 通常不能解决问题，因为所有客户端账号都映射为 QNAP guest。应在 QNAP：

- 确认客户端 IP 在 NFS 白名单；
- 确认共享是 RW；
- 给 guest 账号共享目录和子目录的读写权限。

## 6. leLab 安装显示 Python 包成功，但最后 EOF

曾出现：

```text
unexpected EOF while looking for matching `"'
```

先检查当前仓库脚本：

```bash
bash -n scripts/15-install-lelab-platform.sh
```

语法通过后重新执行：

```bash
sudo ./scripts/15-install-lelab-platform.sh --apply
```

`Successfully installed LeLab ...` 只证明 Python 安装阶段完成；还应确认：

```bash
test -r /etc/robot-platform/lelab.env
test -r /etc/systemd/system/lelab-platform.service
systemctl is-active lelab-platform
```

## 7. curl 本机 API 返回 502

如果设置了 `http_proxy` 或 `https_proxy`，`curl` 可能把 `127.0.0.1` 请求发给代理。

检查：

```bash
env | grep -Ei '^(http|https|all|no)_proxy='
```

访问本机服务时使用：

```bash
curl --noproxy '*' -fsS http://127.0.0.1:8000/health
```

命令换行时，反斜杠 `\` 后面不能再有空格。

## 8. ssh-copy-id 无法打开私钥

报错：

```text
failed to open ID file '/etc/robot-platform/lelab_ssh_key': Permission denied
```

公钥存在但普通用户无权读取同名私钥。明确要求只复制公钥：

```bash
ssh-copy-id \
  -f \
  -i /etc/robot-platform/lelab_ssh_key.pub \
  snorlax@192.168.100.215
```

私钥应继续保持：

```text
robot-train:robotdata 0600
```

不要为了让 `ssh-copy-id` 安静而放宽私钥权限。

## 9. leLab 报 Host key verification failed

这条与 `Permission denied` 无关：**缺的是 `mgmt01` 上 `robot-train` 的 known_hosts 条目**，不是远端公钥。为真正发起连接的服务账号配置 known_hosts，完整指纹核对步骤见 [leLab 主机指纹配置](07-lelab-cluster-web.zh-CN.md#5-验证并安装-worker-主机指纹)。

要点：逐台扫描到独立文件，**不要加 `-H`**（哈希后无法与节点上看到的指纹对应，也无法去重），只比对 `SHA256:` 字段，核对通过后用 `sudo -u robot-train tee -a` **追加**而不是覆盖。

验证必须使用与 systemd 相同的身份：

```bash
sudo -H -u robot-train ssh \
  -o BatchMode=yes \
  -i /etc/robot-platform/lelab_ssh_key \
  snorlax@192.168.100.215 \
  nvidia-smi -L
```

普通用户自己能 SSH 不代表 `robot-train` 能 SSH。

## 9b. leLab 报 Permission denied (publickey,password)

```json
{"name":"gpu02","reachable":false,"reason":"yang@192.168.100.216: Permission denied (publickey,password)."}
```

**这条报错说明 host key 已经通过**，不要再动 known_hosts。缺的是远端 `authorized_keys` 里的 leLab 公钥。

在 `mgmt01`：

```bash
cat /etc/robot-platform/lelab_ssh_key.pub
ssh-copy-id -f -i /etc/robot-platform/lelab_ssh_key.pub yang@192.168.100.216
```

`-f` 不能省，否则 `ssh-copy-id` 会尝试读取只有 `robot-train` 能读的同名私钥。手工粘贴时注意公钥不能折行，用 `ssh-keygen -lf ~/.ssh/authorized_keys` 验证。

装好后不需要重启 leLab，`/cluster/status` 每次请求都会实时探测。

与 `Host key verification failed` 的区别：

| 报错 | 缺的是 |
|---|---|
| `Permission denied (publickey,password)` | 远端 `authorized_keys` 中的公钥 |
| `Host key verification failed` | `mgmt01` 上 `robot-train` 的 known_hosts 条目 |

## 10. 节点 reachable 但 eligible 为 false

示例：

```json
{
  "slurm_state": "idle",
  "reachable": true,
  "compute_processes": 1,
  "eligible": false,
  "reason": "GPU has a compute process outside or inside Slurm"
}
```

这说明 SSH 和 GPU 探测已经成功，但 GPU 上存在 CUDA compute process。因为 Slurm 同时显示 `idle`，通常是 Slurm 外的手动进程。

定位：

```bash
sudo -H -u robot-train ssh \
  -o BatchMode=yes \
  -i /etc/robot-platform/lelab_ssh_key \
  snorlax@192.168.100.215 \
  nvidia-smi \
    --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits
```

到该节点查看 PID：

```bash
ps -o user,pid,ppid,lstart,cmd -p <PID>
```

不要终止未知进程。确认是过期任务后，由进程所有者停止。进程消失后重新请求 API，不需要重启 leLab。

如果占用者是远程桌面类工具（RustDesk、TeamViewer、向日葵等），正确做法不是关掉它，而是把进程名加入 `apps/lelab/lelab/cluster.py` 中 `_probe_node` 的 `graphics_patterns` 白名单。`rustdesk` 已在其中，所以 `mgmt01` 即使运行它，`compute_processes` 也是 0。

### 与调度的关系

`eligible: false` 的节点**不会被 `auto` 选中**，但仍可在页面上手动指定，Slurm 也照常接受 `srun`/`sbatch`。因此 `sinfo` 显示 `idle` 与 leLab 显示不可用并不矛盾——前者看 Slurm 分配状态，后者额外看 GPU 上有没有 Slurm 外的进程。

## 11. SSH 地址与 Slurm 节点名混淆

正确：

```bash
LELAB_CLUSTER_NODES=mgmt01=192.168.100.202,gpu01=snorlax@192.168.100.215,gpu02=yang@192.168.100.216,gpu03=snorlax@192.168.100.217
```

左边是 Slurm NodeName，右边是 SSH 目标。注意 `gpu02` 用的是 `yang`，与另两台不同——**各节点的 SSH 用户不必相同**。以下做法错误：

```text
把 Slurm NodeName 改成 snorlax@192.168.100.215
把 /etc/hosts 中 gpu02 映射写成包含用户的字符串
假定 SSH 用户一定与 NodeName 相同
假定所有节点的 SSH 用户都一样
```

## 11b. 长命令粘贴导致的两类事故

终端粘贴长命令时会在任意位置插入换行，这在本项目里造成过两次真实事故，都值得单独记住。

### sed 表达式被截断

```text
sed: -e expression #1, char 85: unterminated `s' command
```

`LELAB_CLUSTER_NODES` 这类很长的单行值，**用编辑器改，不要用 `sed` 一行命令**。必须用命令行时拆成短片段拼装，执行前先 `echo` 确认是完整一行：

```bash
N='mgmt01=192.168.100.202'
N="$N,gpu01=snorlax@192.168.100.215"
N="$N,gpu02=yang@192.168.100.216"
N="$N,gpu03=snorlax@192.168.100.217"
echo "$N"
sudo sed -i "s|^LELAB_CLUSTER_NODES=.*|LELAB_CLUSTER_NODES=$N|" /etc/robot-platform/lelab.env
```

`sed` 脚本这里必须用双引号，`$N` 需要展开。

同类问题还有 SSH 公钥：`authorized_keys` 中被折行的公钥**永远匹配不上，且没有任何报错**，现象是 `Permission denied (publickey)`。用 `ssh-keygen -lf ~/.ssh/authorized_keys` 确认每行都能解析。

### 变量未设置导致写入根目录

分发 Munge 密钥时，如果只粘贴了后半段而漏掉 `stage_dir="$(mktemp -d)"`：

```bash
sudo install -o "$USER" -g "$(id -gn)" -m 0600 /etc/munge/munge.key "$stage_dir/munge.key"
```

`$stage_dir` 为空，路径变成 `/munge.key`，而这条带 `sudo`，会**静默成功**，把集群唯一的认证密钥写到文件系统根目录。紧接着不带 `sudo` 的那条才会报 `install: cannot create regular file '/slurm.conf.generated': Permission denied`——报错的是后一条，出问题的是前一条。

检查并销毁：

```bash
ls -l /munge.key && sudo shred -u /munge.key
```

文档中的分发命令一律写成 `"${stage_dir:?}/munge.key"`。`:?` 会让 bash 在变量为空时立即报 `stage_dir: parameter null or not set` 并终止，不要删掉它。

## 12. Slurm 远端提示无法进入提交目录

```text
error: couldn't chdir to `/home/kewei/YING/robot_data_platform': No such file or directory: going to /tmp instead
```

**这是警告不是失败。** `srun`/`sbatch` 默认继承提交端当前目录，而该仓库路径只存在于 `mgmt01`。

临时 smoke test 可显式指定所有 Worker 都存在的目录：

```bash
srun --chdir=/tmp <其他参数> <命令>
```

正式 leLab 任务的脚本、日志和输出都用绝对路径：

```text
/mnt/robot_platform/jobs/<job-id>
```

该绝对路径必须在**每台** Worker 上一致。若训练实际失败，检查 `job.sbatch`、`slurm.out` 和 `scontrol show job <id>`，不要只根据 chdir 警告判断失败原因。

leLab 自己提交的作业现在带 `--chdir=/mnt/robot_platform/jobs/<job-id>`，不会再打印这条警告；服务的
`WorkingDirectory=/opt/robot-platform/lelab` 只存在于 `mgmt01`。若仍然看到它，说明服务还没重启到新版本。

## 12b. 作业提交成功但在某个节点启动即失败

作业调度到新加的节点后立刻失败，而在旧节点正常，通常是该节点缺少运行期目录。这类问题**提交时不报错**，只在落到该节点时才暴露。

在该节点检查：

```bash
sudo -u robot-train test -w /var/lib/robot-platform/cache && echo cache OK
findmnt /mnt/robot_platform
sudo -u robot-train test -w /mnt/robot_platform/jobs && echo jobs OK
sudo -u robot-train /opt/robot-platform/train-venv/bin/python -c \
  'import torch, lerobot; print(torch.cuda.is_available())'
sudo -u robot-train /opt/robot-platform/train-venv/bin/python -c \
  'from torchcodec.decoders import VideoDecoder; print("decoder OK")'
```

最后一项失败见 12b-2：缺的是系统 `ffmpeg` 包，而不是 Python 依赖。

`/var/lib/robot-platform/cache` 是 `LELAB_JOB_CACHE_ROOT`。Slurm 把 `HOME` 指向 `robot-train` 的家目录，而它在 Worker 上并不存在，缓存到 `~` 的作业会失败。补建：

```bash
sudo install -d -o robot-train -g robotdata -m 0750 /var/lib/robot-platform/cache
```

NAS 未挂载或只读时，先确认 QNAP 的 NFS 白名单包含**这台**节点的 IP。

### 12b-1. `Cannot create the job cache under '/var/lib/robot-platform/cache'`，但该目录明明存在且可写

报错信息指向 `LELAB_JOB_CACHE_ROOT`，实际失败的却是上一行 `mkdir` 打印的另一个路径：

```text
mkdir: 无法创建目录 "/var/lib/robot-platform/huggingface": 权限不够
Cannot create the job cache under '/var/lib/robot-platform/cache' on gpu03;
```

`sbatch` 默认 `--export=ALL`，作业会继承 leLab 服务自己的 `HF_HOME`
（`/etc/robot-platform/lelab.env` 里的 `/var/lib/robot-platform/huggingface`）。那是**管理节点本地**
的缓存路径，新 Worker 上没有，而其父目录 `/var/lib/robot-platform` 属 root、`robot-train` 无法创建 ——
于是 `mkdir -p` 整条失败。`cache` 目录本身没有问题，它下面的 `torch`、`xdg`、`home` 都已建好。

**先看 `mkdir` 那一行报的具体路径，不要只看第二行的汇总信息。**

修复已在 `apps/lelab/lelab/runners/slurm.py` 里：`LELAB_JOB_CACHE_ROOT` 一旦设置就覆盖继承来的
`HF_HOME`，缓存全部落在 `$LELAB_JOB_CACHE_ROOT/` 下。重启服务生效：

```bash
sudo systemctl restart lelab-platform
```

若暂时不能重启服务，在该节点补建目录也能绕过：

```bash
sudo install -d -o robot-train -g robotdata -m 0750 /var/lib/robot-platform/huggingface
```

### 12b-2. 训练开始后立刻 `RuntimeError: Could not load libtorchcodec`

日志已经打印到 `Start offline training on a fixed dataset`、`num_learnable_params` 等行，说明配置、
数据集、GPU 都正常，随后第一个 batch 在 DataLoader worker 里抛出：

```text
RuntimeError: Caught RuntimeError in DataLoader worker process 0.
  ...
RuntimeError: Could not load libtorchcodec.
[start of libtorchcodec loading traceback]
FFmpeg version 8:
OSError: libavutil.so.60: cannot open shared object file: No such file or directory
...
FFmpeg version 4:
OSError: libavdevice.so.58: cannot open shared object file: No such file or directory
```

torchcodec 会依次尝试 FFmpeg 4/5/6/7/8 五个版本。**Ubuntu 22.04 只有 FFmpeg 4.4，所以只有最后一段
`FFmpeg version 4` 的报错是真实原因，前四段必然失败、可以忽略。** 报错提到的 PyTorch 版本不兼容
（第 2 条）通常也是误导。

真实原因是该节点缺少系统 `ffmpeg` 包。torchcodec 自带 `libtorchcodec_core*.so`，但其依赖的
`libav*.so` 来自系统，没有任何 pip 依赖会安装它。棘手之处在于其他包会顺带装上
`libavcodec58`、`libavformat58`、`libavutil56`，唯独 `libavdevice58` 只由 `ffmpeg` 提供 ——
节点看起来"有 ffmpeg 库"，实际缺的就是这一个。

确认（在出问题的节点上，与正常节点对比）：

```bash
ls /usr/lib/x86_64-linux-gnu/libavdevice.so.58   # 缺失即是此问题
dpkg -l ffmpeg
```

修复：

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
sudo -u robot-train /opt/robot-platform/train-venv/bin/python -c \
  'from torchcodec.decoders import VideoDecoder; print("decoder OK")'
```

`25-install-training-environment.sh` 已把 `ffmpeg` 加入安装包并在末尾做这项导入检查，
`90-validate-deployment.sh` 也增加了 `video decoder loads FFmpeg` 一项；早于该改动装好的节点需手工补装。

## 13. 日志位置

```bash
# leLab
journalctl -u lelab-platform -n 200 --no-pager

# Slurm Controller
journalctl -u slurmctld -n 200 --no-pager

# Worker 与认证
journalctl -u slurmd -u munge -n 200 --no-pager

# 管理基础设施
sudo docker compose \
  --env-file deploy/management/.env \
  -f deploy/management/compose.yaml \
  ps
```

排障时优先保存命令输出、时间点、节点名和 Job ID，不要发送密钥、数据库密码或 Token。
