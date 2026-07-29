# GPU 节点安装

GPU 节点运行 NVIDIA Driver、Docker/NVIDIA Runtime、Munge、slurmd 和统一 LeRobot 训练环境。管理机也按相同标准作为第 5 个 GPU Worker。

## 1. 逐机准备

按 IP 设置主机名：

```text
192.168.100.123  gpu01
192.168.100.206  gpu02
192.168.100.208  gpu03
192.168.100.209  gpu04
```

示例：

```bash
sudo hostnamectl set-hostname gpu01
cp config/site.env.example config/site.env
editor config/site.env
./scripts/00-audit-host.sh gpu
```

确认 `nvidia-smi` 正常并记录：

- 驱动版本；
- GPU 名称与显存；
- CPU、内存；
- NVMe 型号、容量和挂载点；
- 时间同步状态。

不要让安装脚本自动猜测或升级 NVIDIA 驱动。

## 2. 安装节点组件

```bash
sudo ./scripts/20-install-gpu-node.sh --apply
```

脚本将：

- 验证 NVIDIA Driver；
- 安装/检查 Docker 和 NVIDIA Container Toolkit；
- 安装 NFS、Chrony、Munge 和 Slurm；
- 创建 `robot-train` 服务账号；
- 创建 `/cache/datasets`、`/cache/exports` 和 `/work/runs`；
- 以读写方式挂载 QNAP，读取数据集并回写共享 checkpoint；
- 暂停 Munge/slurmd，等待集群统一密钥和配置。

## 3. 收集真实 Slurm 资源

5 台节点（包括 `mgmt01`）分别执行并把完整 NodeName 行交给管理机：

```bash
sudo slurmd -C
```

不能照抄示例 CPU/内存值。四台主机硬件不同也没有问题，但每台必须使用自己的 `slurmd -C` 输出。

Slurm 要求任务用户和主组在全部 Worker 上使用相同数字 ID。安装前确认 `TRAIN_UID` 和 `DATA_GID` 没有被现有账号占用；这项同步只服务于 Slurm，不代表要在 QNAP 上配置数字 UID/GID ACL。

## 4. 安装统一训练环境

每台 GPU 主机执行一次：

```bash
sudo ./scripts/25-install-training-environment.sh --apply
```

模型模板引用 `/opt/robot-platform/train-venv/bin/python`。环境升级通过重新运行脚本完成，不在每个训练任务开始时临时安装依赖。

## 5. 安装集群配置

管理机生成 `slurm.conf.generated` 后，将它和 Munge 密钥通过受控临时通道复制到节点，再执行：

```bash
sudo ./scripts/cluster/install-worker-config.sh \
  /secure/temp/munge.key \
  /secure/temp/slurm.conf.generated \
  --apply
```

完成后立即删除临时 Munge 密钥副本，并检查：

```bash
systemctl status munge slurmd
nvidia-smi
findmnt /mnt/robot_platform
```

## 6. 统一训练环境 GPU 测试

每台 Worker（包括管理机）安装同一版本的训练环境：

```bash
sudo ./scripts/25-install-training-environment.sh --apply
sudo -u robot-train /opt/robot-platform/train-venv/bin/python -c \
  'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

`LEROBOT_GIT_REF` 必须在五台机器保持一致。升级 LeRobot 或 PyTorch 时应整批升级并重新完成五节点 smoke test，不能让不同 Worker 长期运行不同依赖版本。

## 7. 节点约定

- NAS 数据和任务目录在主机上可读写；训练参数传入的 `dataset_root` 只用于读取；
- Checkpoint 写入 NAS `jobs/<job-id>/run`，因此中断后可换节点继续；
- `/cache` 是可删除缓存，不能保存唯一副本；
- 不开放 Docker TCP API；
- 正式训练通过 Slurm 提交，不以个人 SSH 会话作为正式 Run。
- 小组成员手动启动的 CUDA 程序由 leLab 的 SSH `nvidia-smi` 探测识别；CPU 日常使用不影响调度。
