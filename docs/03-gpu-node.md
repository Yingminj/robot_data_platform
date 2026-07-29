# GPU 节点安装

GPU 节点运行 NVIDIA Driver、Docker/NVIDIA Runtime、Munge、slurmd，并把训练数据预热到本地 NVMe。

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
- 以只读方式挂载 QNAP；
- 暂停 Munge/slurmd，等待集群统一密钥和配置。

## 3. 收集真实 Slurm 资源

每台节点执行并把完整 NodeName 行交给管理机：

```bash
sudo slurmd -C
```

不能照抄示例 CPU/内存值。四台主机硬件不同也没有问题，但每台必须使用自己的 `slurmd -C` 输出。

## 4. 安装集群配置

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

## 5. 容器 GPU 测试

使用团队批准的 CUDA 镜像进行测试：

```bash
docker run --rm --gpus all APPROVED_CUDA_IMAGE nvidia-smi
```

不要把 `latest` 作为正式训练镜像。ACT、SigLIP2 和 H5 converter 应记录不可变镜像 digest。

## 6. 节点限制

- NAS 数据源只读；
- Checkpoint 先写 `/work/runs/<job-id>`，再上传 MLflow；
- `/cache` 是可删除缓存，不能保存唯一副本；
- 不开放 Docker TCP API；
- 正式训练通过 Slurm 提交，不以个人 SSH 会话作为正式 Run。

