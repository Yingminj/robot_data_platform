# leLab 集群 Web 安装与配置

[English](07-lelab-cluster-web.md) | **简体中文**

leLab 只安装在 `mgmt01`。它通过本机 Slurm 命令提交训练，通过 SSH 到每台 Worker 执行只读 `nvidia-smi`，发现 Slurm 外的 CUDA 进程。

**加节点不需要重装 leLab**，只需第 3、4、5 节的三件事：节点映射、SSH 公钥、host key。完整扩容流程见[向已有集群增加 GPU 节点](09-add-gpu-node.zh-CN.md)。

```text
浏览器
  → mgmt01:8000 / leLab FastAPI
  → 扫描 /mnt/robot_platform/datasets
  → sinfo 获取 Slurm 节点状态
  → 本机/SSH nvidia-smi 获取 GPU 与计算进程
  → sbatch --nodes=1 --gres=gpu:1
  → /mnt/robot_platform/jobs/<job-id>
```

## 1. 安装前置条件

在 `mgmt01` 检查：

```bash
python3.12 --version
node --version
npm --version
sbatch --version
sinfo -N -l
sudo -u robot-train test -r /mnt/robot_platform/datasets
sudo -u robot-train test -w /mnt/robot_platform/jobs
bash -n scripts/15-install-lelab-platform.sh
```

要求：

- Python 3.12；
- Node.js 20.19 或更高版本和 npm；
- **所有**节点都已在 Slurm 中处于 `idle`；
- **每台** Worker 已安装 `/opt/robot-platform/train-venv`；
- NAS 数据集目录可读，任务目录可写。

Node/npm 应在发起 sudo 的普通用户环境中可用。安装脚本会降权到该用户，在临时目录构建前端；root 不需要单独安装 Node。

## 2. 安装

只在 `mgmt01` 执行：

```bash
sudo ./scripts/15-install-lelab-platform.sh --apply
```

如果直接从 root shell 或自动化系统执行，显式指定拥有 Node/npm 的非 root 用户：

```bash
sudo LELAB_BUILD_USER=kewei \
  ./scripts/15-install-lelab-platform.sh --apply
```

安装位置：

| 内容 | 路径 |
|---|---|
| 应用 | `/opt/robot-platform/lelab` |
| Python venv | `/opt/robot-platform/lelab-venv` |
| 运行配置 | `/etc/robot-platform/lelab.env` |
| 模型模板 | `/etc/robot-platform/model-templates.json` |
| systemd 服务 | `/etc/systemd/system/lelab-platform.service` |

安装脚本只在文件不存在时复制 `lelab.env` 和模型模板。重跑安装不会覆盖已有运行配置；修改模板或节点列表时，应直接编辑 `/etc/robot-platform` 下的活动文件。

检查：

```bash
systemctl is-active lelab-platform
journalctl -u lelab-platform -n 100 --no-pager
curl --noproxy '*' -fsS http://127.0.0.1:8000/health
```

如果安装曾在末尾报 shell 引号或 EOF 错误，先确认当前脚本语法通过，再重跑同一安装命令。Python 包已经安装并不等于 systemd 和 `/etc/robot-platform` 已完成。

## 3. 配置节点映射

编辑**活动**配置（不是仓库里的 `config/lelab.env.example`，那只是模板）：

```bash
sudo editor /etc/robot-platform/lelab.env
```

当前应至少包含：

```bash
LELAB_CLUSTER_ENABLED=1
LELAB_CLUSTER_NODES=mgmt01=192.168.100.202,gpu01=snorlax@192.168.100.215,gpu02=yang@192.168.100.216,gpu03=snorlax@192.168.100.217
LELAB_SSH_CONNECT_TIMEOUT=3
LELAB_SSH_IDENTITY_FILE=/etc/robot-platform/lelab_ssh_key

LELAB_NAS_DATASETS_ROOT=/mnt/robot_platform/datasets
LELAB_OUTPUT_ROOT=/mnt/robot_platform/jobs
LELAB_MODEL_TEMPLATES=/etc/robot-platform/model-templates.json
HF_HOME=/var/lib/robot-platform/huggingface
LELAB_JOB_CACHE_ROOT=/var/lib/robot-platform/cache
```

每个节点项的格式是：

```text
Slurm节点名=SSH目标
```

因此：

- 左侧必须与 `sinfo` 中的 NodeName 相同；
- `mgmt01` 会按节点名识别为本机，不实际 SSH；
- 右侧可以包含 SSH 用户；
- **各节点的 SSH 用户不必相同**，上例中 `gpu02` 是 `yang`，另两台是 `snorlax`；
- 不要把 Slurm NodeName 改成 `snorlax@...`。

节点数量没有上限，leLab 按逗号拆分该变量，探测时最多并发 8 路。增删节点只改这一行，不需要改代码或重装。

> **`LELAB_CLUSTER_NODES` 是很长的一行，用编辑器改，不要用 `sed` 一行命令。** 终端粘贴长命令时会在任意位置插入换行，`sed` 收到被截断的表达式后报：
>
> ```text
> sed: -e expression #1, char 85: unterminated `s' command
> ```
>
> 必须用命令行时，把值拆成短片段拼装，每段都短到不会被折行，并在执行前用 `echo` 确认：
>
> ```bash
> N='mgmt01=192.168.100.202'
> N="$N,gpu01=snorlax@192.168.100.215"
> N="$N,gpu02=yang@192.168.100.216"
> N="$N,gpu03=snorlax@192.168.100.217"
> echo "$N"
> sudo sed -i "s|^LELAB_CLUSTER_NODES=.*|LELAB_CLUSTER_NODES=$N|" /etc/robot-platform/lelab.env
> ```
>
> `sed` 脚本这里必须用双引号，`$N` 需要展开。

该文件由 systemd 以 `EnvironmentFile` 读取，**只在服务重启时生效**，改完必须：

```bash
grep -n LELAB_CLUSTER_NODES /etc/robot-platform/lelab.env
sudo systemctl restart lelab-platform
```

### LELAB_JOB_CACHE_ROOT

`LELAB_JOB_CACHE_ROOT` 指向的目录必须在**每台 Worker** 上存在且 `robot-train` 可写：

```bash
sudo install -d -o robot-train -g robotdata -m 0750 /var/lib/robot-platform/cache
```

Slurm 把 `HOME` 指向 `robot-train` 的家目录，而它在 Worker 上并不存在，缓存到 `~` 的作业（torch hub backbone 权重、HF、wandb）会在该节点失败。各节点用相同的本地路径即可，不需要共享存储。未设置时回退到 `HF_HOME`。

**这一项在提交作业时不报错，只在作业被调度到缺失该目录的节点后才失败**，加节点时容易漏。

## 4. 生成 leLab SSH 密钥

以下命令只在 `mgmt01` 执行。若文件已存在，不要覆盖，先检查是否为当前已经授权的密钥。

```bash
sudo test -e /etc/robot-platform/lelab_ssh_key || \
  sudo ssh-keygen \
    -q \
    -t ed25519 \
    -N '' \
    -C lelab-gpu-probe \
    -f /etc/robot-platform/lelab_ssh_key

sudo chown \
  robot-train:robotdata \
  /etc/robot-platform/lelab_ssh_key \
  /etc/robot-platform/lelab_ssh_key.pub
sudo chmod 0600 /etc/robot-platform/lelab_ssh_key
sudo chmod 0644 /etc/robot-platform/lelab_ssh_key.pub
```

将公钥安装到**每台** Worker 的登录账号（注意各台账号可能不同）：

```bash
ssh-copy-id -f -i /etc/robot-platform/lelab_ssh_key.pub snorlax@192.168.100.215
ssh-copy-id -f -i /etc/robot-platform/lelab_ssh_key.pub yang@192.168.100.216
ssh-copy-id -f -i /etc/robot-platform/lelab_ssh_key.pub snorlax@192.168.100.217
```

这里保留 `-f` 很重要。`ssh-copy-id` 默认可能尝试打开同名私钥，而私钥只允许 `robot-train` 读取，从普通用户执行时会出现 `Permission denied`；`-f` 明确只安装给出的公钥。

如果 `ssh-copy-id` 不可用，在目标节点手工粘贴：

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
editor ~/.ssh/authorized_keys      # 粘贴 lelab_ssh_key.pub 的那一行
chmod 600 ~/.ssh/authorized_keys
ssh-keygen -lf ~/.ssh/authorized_keys
```

公钥是很长的一行。**终端粘贴时若被折行，会得到一个永远匹配不上的密钥，且没有任何报错**，现象是后续 `Permission denied (publickey)`。用最后一条命令确认新增密钥能被正确解析。

如果当前登录用户不能穿过 `/etc/robot-platform` 目录读取公钥，先把公钥复制到仅当前用户可读的临时文件，安装后删除：

```bash
sudo install -o "$USER" -g "$(id -gn)" -m 0600 \
  /etc/robot-platform/lelab_ssh_key.pub \
  /tmp/lelab_ssh_key.pub
ssh-copy-id -f -i /tmp/lelab_ssh_key.pub snorlax@192.168.100.215
rm -f /tmp/lelab_ssh_key.pub
```

## 5. 验证并安装 Worker 主机指纹

SSH 公钥授权和服务器 host key 信任是两件事。`ssh-copy-id` 成功后，systemd 服务仍可能因为 `Host key verification failed` 无法连接。

两种失败现象要分清，它们的修法完全不同：

| 报错 | 原因 | 看哪一节 |
|---|---|---|
| `Permission denied (publickey,password)` | host key 已通过，公钥未装入远端 `authorized_keys` | 第 4 节 |
| `Host key verification failed` | 公钥无关，`robot-train` 的 known_hosts 缺该节点 | 本节 |

### 5.1 在 mgmt01 逐台扫描

**逐台扫描到独立文件，且不要加 `-H`。** `-H` 会把主机名哈希，且每次用不同的盐，结果是：同一台机器重复扫描会产生看起来不同的多行，无法去重；哈希后的行也无法与节点上看到的指纹对应，等于放弃了人工核对。

```bash
ssh-keyscan -T 5 -t ed25519 192.168.100.215 2>/dev/null > /tmp/kh215
ssh-keyscan -T 5 -t ed25519 192.168.100.216 2>/dev/null > /tmp/kh216
ssh-keyscan -T 5 -t ed25519 192.168.100.217 2>/dev/null > /tmp/kh217

ssh-keygen -lf /tmp/kh215
ssh-keygen -lf /tmp/kh216
ssh-keygen -lf /tmp/kh217
```

每条输出形如：

```text
256 SHA256:EWn7POFCGoIXDCnTAHRp4LQw1b68yiwLUj0HlpRII7Q 192.168.100.216 (ED25519)
```

`ssh-keyscan` 的提示信息（`# 192.168.100.216:22 SSH-2.0-OpenSSH_8.9p1 ...`）走 stderr，不会进入文件，看到它们是正常的。

### 5.2 在每台 Worker 的控制台查看真实指纹

```bash
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
```

```text
256 SHA256:EWn7POFCGoIXDCnTAHRp4LQw1b68yiwLUj0HlpRII7Q root@gpu02 (ED25519)
```

**只比对 `SHA256:` 字段。** 末尾注释必然不同：扫描结果标注被查询的 IP，节点本机标注密钥注释。开头的 `256` 和 `(ED25519)` 也应相同。

指纹不一致就停止，说明应答该 IP 的不是你以为的机器。不要跳过对比。

### 5.3 为 robot-train 安装 known_hosts

首次准备目录：

```bash
sudo install -d -o robot-train -g robotdata -m 0750 /home/robot-train
sudo install -d -o robot-train -g robotdata -m 0700 /home/robot-train/.ssh
```

核对通过后**追加**（`-a`），不要覆盖已有节点的条目：

```bash
sudo -u robot-train tee -a /home/robot-train/.ssh/known_hosts < /tmp/kh215 >/dev/null
sudo -u robot-train tee -a /home/robot-train/.ssh/known_hosts < /tmp/kh216 >/dev/null
sudo -u robot-train tee -a /home/robot-train/.ssh/known_hosts < /tmp/kh217 >/dev/null
rm -f /tmp/kh215 /tmp/kh216 /tmp/kh217
```

这里用 `sudo -u robot-train tee -a`，而**不能**用 `sudo ... >>`：重定向由当前 shell 执行，身份是你自己而不是 `robot-train`，会写不进去或生成属主错误的文件，SSH 随后直接忽略它。

确认每台都已写入：

```bash
for ip in 192.168.100.215 192.168.100.216 192.168.100.217; do
  sudo -u robot-train ssh-keygen -F "$ip" -f /home/robot-train/.ssh/known_hosts
done
```

这些命令都在 `mgmt01` 执行，因为发起 SSH 的进程是 `mgmt01` 上的 `robot-train`。

## 6. 以服务账号验证 SSH

仍在 `mgmt01`，对**每台** Worker 执行一次。这条命令与 leLab 服务实际发起的连接完全一致，普通用户自己能 SSH 不代表 `robot-train` 能：

```bash
for target in snorlax@192.168.100.215 yang@192.168.100.216 snorlax@192.168.100.217; do
  echo "[$target]"
  sudo -H -u robot-train ssh \
    -o BatchMode=yes \
    -i /etc/robot-platform/lelab_ssh_key \
    "$target" \
    nvidia-smi --query-gpu=name,memory.total,memory.free \
      --format=csv,noheader,nounits
done
```

每台都必须无密码返回 GPU 信息。`BatchMode=yes` 不允许任何交互，因此 host key 未知会直接失败而不是提示确认——这是有意的。若失败，先不要重启 leLab；根据错误检查：

| 报错 | 原因 |
|---|---|
| `Identity file ... not accessible` | 私钥路径、属主或权限 |
| `Permission denied (publickey,password)` | **host key 已通过**，公钥未装入该节点的 `authorized_keys`（第 4 节） |
| `Host key verification failed` | 公钥无关，`robot-train` 的 known_hosts 缺该节点或指纹变化（第 5 节） |
| timeout | IP、SSH 服务或防火墙 |

前两者容易混淆：只要看到 `Permission denied`，就说明 host key 部分已经正确，不必再动 known_hosts。

## 7. 检查集群 API

```bash
sudo systemctl restart lelab-platform

curl --noproxy '*' -fsS http://127.0.0.1:8000/cluster/status | \
  jq -c '.nodes[] | {name,address,reachable,slurm_state,eligible,memory_free_mb,reason}'
curl --noproxy '*' -fsS \
  http://127.0.0.1:8000/cluster/templates | jq
```

每个节点应是：

```json
{"name":"gpu02","address":"yang@192.168.100.216","reachable":true,"slurm_state":"idle","eligible":true,"memory_free_mb":47752,"reason":null}
```

`/cluster/status` 每次请求都会实时探测，改完 `authorized_keys` 或结束 GPU 进程后**不需要重启 leLab**，重新请求即可。只有改 `/etc/robot-platform/lelab.env` 才需要重启。

每个节点的字段含义：

| 字段 | 含义 |
|---|---|
| `slurm_state` | `sinfo` 中的节点状态 |
| `reachable` | 本机或 SSH `nvidia-smi` 是否成功 |
| `compute_processes` | 当前 GPU 计算进程数 |
| `eligible` | 是否允许 leLab 选择该节点 |
| `reason` | 不可选择的直接原因 |

节点只有同时满足以下条件才 `eligible: true`：

- Slurm 状态为 `idle`；
- GPU 探测可达；
- 没有 CUDA compute process；
- 空闲显存满足所选模板。

如果 Slurm 为 `idle`，但 `compute_processes` 大于 0，通常表示有人通过 SSH 或桌面会话在 Slurm 外运行 CUDA。**该节点会一直 `eligible: false`，`auto` 调度永远不会选它。** 定位时在 `mgmt01` 执行（把 SSH 目标换成对应节点）：

```bash
sudo -H -u robot-train ssh \
  -o BatchMode=yes \
  -i /etc/robot-platform/lelab_ssh_key \
  snorlax@192.168.100.215 \
  nvidia-smi \
    --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits
```

再到该节点查看对应 PID：

```bash
ps -o user,pid,ppid,lstart,cmd -p <PID>
```

确认进程用途和所有者后再决定是否由所有者停止。不要直接终止未知进程。进程退出后无需重启 leLab，重新请求 `/cluster/status` 即可。

如果占用者是远程桌面类工具（RustDesk、TeamViewer、向日葵等），正确做法不是关掉它，而是把进程名加入 leLab 的图形进程白名单——`apps/lelab/lelab/cluster.py` 中 `_probe_node` 的 `graphics_patterns`。`rustdesk` 已在其中，所以 `mgmt01` 即使运行它也报 0。

## 8. 模型模板

活动模板：

```text
/etc/robot-platform/model-templates.json
```

模板限制用户可选择的：

- LeRobot policy 类型；
- Python 训练环境；
- Slurm partition；
- 最低空闲显存；
- CPU 和内存申请。

第一阶段模板 `id` 应与 `policy_type` 相同，避免把任意命令暴露给 Web。修改后：

```bash
sudo systemctl restart lelab-platform
curl --noproxy '*' -fsS \
  http://127.0.0.1:8000/cluster/templates | jq
```

## 9. NAS 数据集和任务目录

leLab 识别包含 `meta/info.json` 的 LeRobot 数据集：

```text
/mnt/robot_platform/datasets/team/pick-cube/
├── meta/info.json
├── data/
└── videos/
```

检查 API 是否发现：

```bash
curl --noproxy '*' -fsS \
  http://127.0.0.1:8000/datasets | jq
```

每条 Slurm 任务写入：

```text
/mnt/robot_platform/jobs/<job-id>/
├── job.json
├── job.sbatch
├── log.jsonl
├── slurm.out
└── run/checkpoints/
```

`LELAB_OUTPUT_ROOT` 必须在**每台** Worker 上以相同绝对路径可见，否则该节点不能写日志和 checkpoint。加节点后先确认新节点的 `findmnt /mnt/robot_platform` 正常、QNAP 白名单包含它的 IP。

## 10. 第一条训练任务

正式使用前选一份小型数据集和 ACT 模板，设置很短的训练步数，验证：

1. 页面列出数据集；
2. 至少一个节点 `eligible: true`；
3. 提交后 `squeue` 出现 Job；
4. `slurm.out` 和 `log.jsonl` 持续更新；
5. 输出目录生成 checkpoint；
6. Stop 能触发 `scancel`；
7. 页面能从完整 checkpoint 恢复。

如果网页提交失败，同时查看：

```bash
journalctl -u lelab-platform -f
squeue
scontrol show job <SlurmJobID>
```

## 11. 访问和安全边界

试点地址：

```text
http://192.168.100.202:8000
```

当前 8000 端口没有作为正式公网入口设计。正式上线前应增加反向代理、HTTPS、身份认证和访问控制。leLab 私钥只能用于 GPU 探测；后续应在 `gpu01` 的 `authorized_keys` 中限制允许执行的命令。
