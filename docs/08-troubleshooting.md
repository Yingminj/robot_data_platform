# 安装与运行排障

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
curl --noproxy '*' -fsS http://127.0.0.1:8000/cluster/status | jq
```

在 `gpu01`：

```bash
hostname -s
slurmd -V
stat -fc %T /sys/fs/cgroup
systemctl is-active munge slurmd
nvidia-smi
sudo slurmd -G
findmnt /mnt/robot_platform
```

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

若仍是 Ubuntu 22.04 自带旧版，按 [Slurm 26.05.2 安装](Slurm-INSTALL.md)升级两台机器。不能只升级 Controller 或只升级 Worker。

## 4. Slurm 节点为 DOWN、INVAL 或 UNKNOWN

在 `mgmt01`：

```bash
scontrol show node mgmt01
scontrol show node gpu01
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
3. CPU 拓扑和 `RealMemory`；
4. `slurm.conf`、`cgroup.conf`、`gres.conf` checksum；
5. Munge key checksum、`munge:munge` 和 `0400`；
6. 两台机器时间；
7. TCP 6817/6818；
8. `sudo slurmd -G` 是否识别 `gpu:1`。

只有原因已经修复时才恢复节点：

```bash
sudo scontrol update NodeName=gpu01 State=RESUME
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

先确认 SSH 公钥已经授权，然后为真正发起连接的服务账号 `robot-train` 配置 known_hosts。完整指纹核对步骤见 [leLab 主机指纹配置](07-lelab-cluster-web.md#5-验证并安装-gpu01-主机指纹)。

验证必须使用与 systemd 相同的身份：

```bash
sudo -H -u robot-train ssh \
  -o BatchMode=yes \
  -i /etc/robot-platform/lelab_ssh_key \
  snorlax@192.168.100.215 \
  nvidia-smi -L
```

普通用户自己能 SSH 不代表 `robot-train` 能 SSH。

## 10. gpu01 reachable 但 eligible 为 false

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

到 `gpu01` 查看 PID：

```bash
ps -o user,pid,ppid,lstart,cmd -p <PID>
```

不要终止未知进程。确认是过期任务后，由进程所有者停止。进程消失后重新请求 API，不需要重启 leLab。

## 11. SSH 地址与 Slurm 节点名混淆

正确：

```bash
LELAB_CLUSTER_NODES=mgmt01=192.168.100.202,gpu01=snorlax@192.168.100.215
```

左边是 Slurm NodeName，右边是 SSH 目标。以下做法错误：

```text
把 Slurm NodeName 改成 snorlax@192.168.100.215
把 /etc/hosts 中 gpu01 映射写成包含用户的字符串
假定 SSH 用户一定与 NodeName 相同
```

## 12. Slurm 远端提示无法进入提交目录

`srun`/`sbatch` 默认继承提交端当前目录。如果该仓库路径只存在于 `mgmt01`，远端可能警告无法进入该目录。

临时 smoke test 可显式指定所有 Worker 都存在的目录：

```bash
srun --chdir=/tmp <其他参数> <命令>
```

正式 leLab 任务的脚本、日志和输出应放在：

```text
/mnt/robot_platform/jobs/<job-id>
```

该绝对路径必须在两台 Worker 上一致。若训练实际失败，检查 `job.sbatch`、`slurm.out` 和 `scontrol show job <id>`，不要只根据 chdir 警告判断失败原因。

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
