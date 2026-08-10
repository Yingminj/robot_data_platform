# Combined collector and GPU node

**English** | [简体中文](05-combined-node.zh-CN.md)

One machine can serve as both a collector node and a GPU node, but it needs the union of the two roles installed, and the accounts, directories and data flows must stay isolated.

## 1. Installation

```bash
cp config/site.env.example config/site.env
editor config/site.env
./scripts/00-audit-host.sh combined
sudo ./scripts/40-install-combined-node.sh --apply
```

Required services:

| Category | Services |
|---|---|
| GPU | NVIDIA driver, Docker, NVIDIA Toolkit, Munge, slurmd |
| Storage | NFS read/write mount, local dataset cache, `/work/runs` |
| Collection | ROS2, camera/robot drivers, H5 converter, validator, upload agent |
| Common | Chrony, SSH, monitoring |

PostgreSQL, the MLflow server and the Slurm controller do not need to be installed.

## 2. Permission isolation

- `robot-collector` only writes to `/var/spool/robot-data`;
- `robot-train` only writes to `/cache` and `/work/runs`;
- in phase one the NAS is mounted read/write on this host, with fine-grained permissions handled later;
- collected data is uploaded through the management node API;
- do not grant the collection processes write access to `raw` just because the machine is also a GPU node.

## 3. Resource contention

The following problems are the most likely on a combined node:

- training saturates the GPU and disrupts the cameras or vision preprocessing;
- dataset warm-up saturates disk bandwidth, causing dropped frames in H5 writes;
- training exhausts memory, causing ROS2 / the H5 converter to be OOM-killed;
- `/work` and the spool share a disk, so training output squeezes out space for data not yet uploaded.

Choose at least one operating policy:

1. drain the node in Slurm during collection;
2. give collection a fixed time window and run training only outside it;
3. reserve CPU/memory in Slurm and put the spool on a dedicated SSD;
4. if collection also needs the GPU, prevent the node from accepting GPU jobs during collection.

Draining before collection:

```bash
sudo scontrol update NodeName=gpu01 State=DRAIN Reason=collection
```

Resume once collection has finished and the upload is confirmed safe:

```bash
sudo scontrol update NodeName=gpu01 State=RESUME
```

These commands need Slurm administrative rights and should be run from the management node or through a controlled operations process.

## 4. Recommended disk layout

```text
dedicated collection disk  /var/spool/robot-data
training cache disk        /cache/datasets
training work disk         /work/runs
NAS                        /mnt/robot_platform (read/write)
```

If only one NVMe is available, set separate space watermarks and prioritize protecting collected data that is not yet `COMMITTED`.
