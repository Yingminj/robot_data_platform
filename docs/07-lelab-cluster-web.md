# leLab 集群 Web

第一阶段 Web 基于 `https://github.com/Yingminj/leLab` 维护。它仍是独立 Git 仓库，克隆位置为：

```text
apps/lelab/
```

父仓库保存部署脚本，leLab fork 保存 FastAPI/React 功能修改。修改后应分别提交。

## 1. 第一阶段工作流

```text
浏览器
  → leLab FastAPI
  → 扫描 /mnt/robot_platform/datasets
  → 读取预登记模型模板
  → sinfo + SSH nvidia-smi 检查 5 台 GPU
  → sbatch --nodes=1 --gres=gpu:1
  → /mnt/robot_platform/jobs/<job-id>
  → 日志、checkpoint、停止与续训
```

SSH 只用于发现组员在 Slurm 外手动启动的 CUDA 进程。训练启动、停止和资源占用仍由 Slurm 管理。

节点满足以下全部条件才显示为可调度：

- Slurm 状态为 `idle`；
- SSH/本机 `nvidia-smi` 可用；
- `nvidia-smi --query-compute-apps` 没有计算进程；
- 空闲显存达到所选模板的最低要求。

提交前 Web 会选择空闲显存最多的节点，并在同一 Web 进程内预留已提交但 Slurm 状态尚未刷新的节点，避免多人快速提交时全部排到同一台机器；batch 脚本在目标节点上再次检查 CUDA 进程，缩小“检查后突然被本地程序占用”的竞态窗口。

## 2. 管理机配置

复制并检查：

```bash
sudo install -d -m 0750 /etc/robot-platform
sudo cp config/lelab.env.example /etc/robot-platform/lelab.env
sudo editor /etc/robot-platform/lelab.env
```

重要变量：

| 变量 | 作用 |
|---|---|
| `LELAB_CLUSTER_NODES` | `Slurm名=SSH地址` 列表 |
| `LELAB_NAS_DATASETS_ROOT` | Web 扫描的数据集根目录 |
| `LELAB_OUTPUT_ROOT` | 所有节点共享的任务、日志与 checkpoint 根目录 |
| `LELAB_MODEL_TEMPLATES` | 管理员登记的模型模板 JSON |
| `LELAB_SSH_IDENTITY_FILE` | `robot-train` 用于只读 GPU 探测的 SSH 密钥 |

为 `robot-train` 配置到 4 台远程节点的无密码 SSH。密钥只需允许执行 `nvidia-smi`；试点期可先使用普通 SSH key，后续再通过 `authorized_keys` command restriction 收紧。

管理机上的私钥应由服务账号持有：

```bash
sudo install -o robot-train -g robotdata -m 0600 \
  /secure/source/lelab_ssh_key /etc/robot-platform/lelab_ssh_key
```

验证：

```bash
sudo -u robot-train ssh -i /etc/robot-platform/lelab_ssh_key gpu01 \
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits
```

## 3. 模型模板

默认模板文件：

```text
apps/lelab/config/model-templates.json.example
```

安装脚本会首次复制到：

```text
/etc/robot-platform/model-templates.json
```

模板固定：

- LeRobot policy 类型；
- Python 训练环境；
- Slurm partition；
- 最低空闲显存；
- CPU 和内存申请。

第一阶段模板 `id` 必须与 `policy_type` 相同，避免用户提交任意命令。调整模板后重启：

```bash
sudo systemctl restart lelab-platform
```

## 4. NAS 数据集布局

leLab 识别包含 `meta/info.json` 的 LeRobot 数据集，例如：

```text
/mnt/robot_platform/datasets/team/pick-cube/
├── meta/info.json
├── data/
└── videos/
```

API 返回 `team/pick-cube` 作为显示 ID，同时把绝对 `dataset_root` 传给训练任务，因此 Worker 不需要再从 Hugging Face Hub 下载相同数据。

## 5. Checkpoint 续训

每个任务使用固定共享目录：

```text
/mnt/robot_platform/jobs/<job-id>/
├── job.json
├── job.sbatch
├── log.jsonl
├── slurm.out
└── run/checkpoints/<step>/pretrained_model/
```

停止、失败或中断的 Slurm 任务只要存在完整 checkpoint，页面会显示 “Resume from checkpoint”。恢复时保留原输出目录，将 `resume=true` 交给 LeRobot，并重新选择一台空闲 GPU。

## 6. 启动与检查

```bash
sudo ./scripts/15-install-lelab-platform.sh --apply
systemctl status lelab-platform
curl -fsS http://192.168.100.202:8000/health
curl -fsS http://192.168.100.202:8000/cluster/status | jq
curl -fsS http://192.168.100.202:8000/cluster/templates | jq
```

小组成员从任意内网电脑打开：

```text
http://192.168.100.202:8000
```

反向代理、HTTPS 和成员登录不是第一阶段前置条件。
