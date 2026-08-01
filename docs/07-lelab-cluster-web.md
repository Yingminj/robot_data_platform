# leLab 集群 Web 安装与配置

leLab 只安装在 `mgmt01`。它通过本机 Slurm 命令提交训练，通过 SSH 到 `gpu01` 执行只读 `nvidia-smi`，发现 Slurm 外的 CUDA 进程。

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
- `mgmt01`、`gpu01` 都已在 Slurm 中处于可用状态；
- 两台 Worker 已安装 `/opt/robot-platform/train-venv`；
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

## 3. 配置双节点映射

编辑活动配置：

```bash
sudo editor /etc/robot-platform/lelab.env
```

当前应至少包含：

```bash
LELAB_CLUSTER_ENABLED=1
LELAB_CLUSTER_NODES=mgmt01=192.168.100.202,gpu01=snorlax@192.168.100.215
LELAB_SSH_CONNECT_TIMEOUT=3
LELAB_SSH_IDENTITY_FILE=/etc/robot-platform/lelab_ssh_key

LELAB_NAS_DATASETS_ROOT=/mnt/robot_platform/datasets
LELAB_OUTPUT_ROOT=/mnt/robot_platform/jobs
LELAB_MODEL_TEMPLATES=/etc/robot-platform/model-templates.json
HF_HOME=/var/lib/robot-platform/huggingface
```

每个节点项的格式是：

```text
Slurm节点名=SSH目标
```

因此：

- 左侧必须与 `sinfo` 中的 NodeName 相同；
- `mgmt01` 会按节点名识别为本机，不实际 SSH；
- `gpu01` 的右侧可以包含 SSH 用户，当前为 `snorlax@192.168.100.215`；
- 不要把 Slurm NodeName 改成 `snorlax@...`。

修改后重启：

```bash
sudo systemctl restart lelab-platform
```

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

将公钥安装到 `gpu01` 的 `snorlax` 账号：

```bash
ssh-copy-id \
  -f \
  -i /etc/robot-platform/lelab_ssh_key.pub \
  snorlax@192.168.100.215
```

这里保留 `-f` 很重要。`ssh-copy-id` 默认可能尝试打开同名私钥，而私钥只允许 `robot-train` 读取，从普通用户执行时会出现 `Permission denied`；`-f` 明确只安装给出的公钥。

如果当前登录用户不能穿过 `/etc/robot-platform` 目录读取公钥，先把公钥复制到仅当前用户可读的临时文件，安装后删除：

```bash
sudo install -o "$USER" -g "$(id -gn)" -m 0600 \
  /etc/robot-platform/lelab_ssh_key.pub \
  /tmp/lelab_ssh_key.pub
ssh-copy-id -f -i /tmp/lelab_ssh_key.pub snorlax@192.168.100.215
rm -f /tmp/lelab_ssh_key.pub
```

## 5. 验证并安装 gpu01 主机指纹

SSH 公钥授权和服务器 host key 信任是两件事。`ssh-copy-id` 成功后，systemd 服务仍可能因为 `Host key verification failed` 无法连接。

### 5.1 在 gpu01 查看真实指纹

登录 `gpu01`：

```bash
sudo ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
```

记下 SHA256 指纹。

### 5.2 在 mgmt01 扫描并人工对比

```bash
ssh-keyscan \
  -T 5 \
  -H \
  -t ed25519 \
  192.168.100.215 \
  > /tmp/gpu01-ssh-host-key
ssh-keygen -lf /tmp/gpu01-ssh-host-key
```

只有两个指纹完全一致时才继续。不要跳过对比。

### 5.3 为 robot-train 安装 known_hosts

```bash
sudo install -d \
  -o robot-train \
  -g robotdata \
  -m 0750 \
  /home/robot-train
sudo install -d \
  -o robot-train \
  -g robotdata \
  -m 0700 \
  /home/robot-train/.ssh
sudo install \
  -o robot-train \
  -g robotdata \
  -m 0600 \
  /tmp/gpu01-ssh-host-key \
  /home/robot-train/.ssh/known_hosts
rm -f /tmp/gpu01-ssh-host-key
```

这些命令都在 `mgmt01` 执行，因为发起 SSH 的进程是 `mgmt01` 上的 `robot-train`。

如果以后添加更多 Worker，不要覆盖已有 `known_hosts`；把每台已经人工验证的扫描结果合并后一次安装。

## 6. 以服务账号验证 SSH

仍在 `mgmt01`：

```bash
sudo -H -u robot-train ssh \
  -o BatchMode=yes \
  -i /etc/robot-platform/lelab_ssh_key \
  snorlax@192.168.100.215 \
  nvidia-smi \
    --query-gpu=name,memory.total,memory.free \
    --format=csv,noheader,nounits
```

这一步必须无密码返回 GPU 信息。若失败，先不要重启 leLab；根据错误检查：

- `Identity file ... not accessible`：私钥路径、属主或权限；
- `Permission denied (publickey)`：公钥未装入 `snorlax` 的 `authorized_keys`；
- `Host key verification failed`：`robot-train` 的 known_hosts 未配置或指纹变化；
- timeout：IP、SSH 服务或防火墙。

## 7. 检查集群 API

```bash
sudo systemctl restart lelab-platform

curl --noproxy '*' -fsS \
  http://127.0.0.1:8000/cluster/status | jq
curl --noproxy '*' -fsS \
  http://127.0.0.1:8000/cluster/templates | jq
```

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

如果 Slurm 为 `idle`，但 `compute_processes` 大于 0，通常表示有人通过 SSH 或桌面会话在 Slurm 外运行 CUDA。定位时在 `mgmt01` 执行：

```bash
sudo -H -u robot-train ssh \
  -o BatchMode=yes \
  -i /etc/robot-platform/lelab_ssh_key \
  snorlax@192.168.100.215 \
  nvidia-smi \
    --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits
```

再到 `gpu01` 查看对应 PID：

```bash
ps -o user,pid,ppid,lstart,cmd -p <PID>
```

确认进程用途和所有者后再决定是否由所有者停止。不要直接终止未知进程。进程退出后无需重启 leLab，重新请求 `/cluster/status` 即可。

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

`LELAB_OUTPUT_ROOT` 必须在两台 Worker 上以相同绝对路径可见，否则远程节点不能写日志和 checkpoint。

## 10. 第一条训练任务

正式使用前选一份小型数据集和 ACT 模板，设置很短的训练步数，验证：

1. 页面列出数据集；
2. 两台节点至少一台 `eligible: true`；
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
