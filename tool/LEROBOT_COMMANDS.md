# LeRobot 常用命令速查

针对本平台的 LeRobot **0.6.0** / 数据集格式 **v3.0**。本文所有命令都在 `lerobot` conda 环境中
实测通过（`conda run -n lerobot ...`），输出行为以实测为准，不是照抄上游文档。

数据集的产生方式见 `tool/CONVERSION.md`；本文讲的是数据集**产生之后**怎么查看、拆分、合并、修改。

## 0. 通用约定

```bash
# 所有命令都要在 lerobot 环境里跑
conda activate lerobot
# 或者逐条：conda run -n lerobot <命令>
```

两个贯穿全文的参数：

| 参数 | 含义 |
|---|---|
| `--repo-id` / `--repo_id` | 数据集标识符，本地数据集一般是 `local/<名字>` |
| `--root` | 数据集在磁盘上的实际目录 |

**`--root` 必须指向一个已经存在的本地目录。** 如果路径不存在（或者干脆没传 `--root`），
LeRobot 会拿 `repo_id` 去 Hugging Face Hub 上找，然后报 `RepositoryNotFoundError: 401`。
看到 401 先检查路径拼写，不是权限问题。

注意两个脚本的参数风格不一致，抄命令时别串了：

- `lerobot-dataset-viz` 用**连字符**：`--repo-id`、`--episode-index`
- `lerobot-edit-dataset` 用**下划线**：`--repo_id`、`--new_repo_id`

本文示例统一用变量表示数据集根目录：

```bash
DS=/mnt/robot_platform/datasets
```

## 1. 查看数据集信息

### 环境自检

```bash
lerobot-info
```

打印 LeRobot 版本、PyTorch/CUDA/FFmpeg 版本、GPU 型号和所有可用的 `lerobot-*` 命令。
报 bug 或者对比两台机器环境时先跑这个。

### 数据集摘要

```bash
lerobot-edit-dataset \
  --operation.type=info \
  --repo_id=local/my_dataset \
  --root=$DS/my_dataset
```

输出 episode 数、帧数、fps、robot_type、任务列表。加 `--operation.show_features=true`
会额外打印完整的 feature schema（每个键的 dtype / shape / names，视频键还带
`video.height` / `video.width` / `video.codec` 等编码信息）。

**合并前必看这份 feature 输出** —— 它就是判断两个数据集能不能合并的依据（见 §3）。

也可以直接读元数据文件，适合写脚本：

```bash
python -c "import json; d=json.load(open('$DS/my_dataset/meta/info.json')); \
print(d['total_episodes'], d['total_frames'], d['fps'], d['robot_type'])"
```

注意 v3.0 的 `meta/info.json` **不包含 `repo_id` 字段**，别去读它。

## 2. 可视化

```bash
lerobot-dataset-viz \
  --repo-id local/my_dataset \
  --root $DS/my_dataset \
  --episode-index 0
```

一次只能看一个 episode（`--episode-index` 必填）。默认用 Rerun 后端，会在本机弹出查看器窗口。

### 在服务器上跑、在本地看

mgmt01 / gpu01 上没有图形界面，有三种办法：

**方式 A：导出 .rrd 文件，拷回本地看**（最省事，实测可用）

```bash
lerobot-dataset-viz \
  --repo-id local/my_dataset --root $DS/my_dataset \
  --episode-index 0 \
  --save 1 --output-dir /tmp/rrd
# 产出 /tmp/rrd/local_my_dataset_episode_0.rrd
```

`--save 1` 会**关闭**查看器的启动，只写文件。把文件拷到本地后 `rerun path/to/file.rrd`。

**方式 B：服务器起服务，本地连过去**

```bash
lerobot-dataset-viz \
  --repo-id local/my_dataset --root $DS/my_dataset \
  --episode-index 0 \
  --mode distant --web-port 9090 --grpc-port 9876
# 本地执行：rerun rerun+http://<服务器IP>:9876/proxy
```

**方式 C：Foxglove 后端**（时间轴可拖动，适合逐帧核对对齐）

```bash
lerobot-dataset-viz \
  --repo-id local/my_dataset --root $DS/my_dataset \
  --episode-index 0 \
  --display-mode foxglove --host 0.0.0.0 --web-port 8765
# 本地 Foxglove 客户端连 ws://<服务器IP>:8765
```

`--host 0.0.0.0` 会监听所有网卡。当前部署没有认证层，只在内网这么用。

### 其他有用的开关

| 参数 | 用途 |
|---|---|
| `--display-compressed-images` | 显示 JPEG 压缩图而不是解码后的原图，远程时省带宽 |
| `--no-autoplay` | Foxglove 模式下连上不自动播放，停在第 0 帧 |
| `--tolerance-s` | 放宽时间戳与 fps 的一致性校验（默认 1e-4），只在加载报时间戳错误时才动 |
| `--batch-size` / `--num-workers` | 加载速度，长 episode 可以调大 |

## 3. 合并数据集

用于「先录了 50 条，后来又录 50 条」这种增量场景。本仓库的转换脚本
（`rdp convert`、`rdp convert-hdf5`）**没有追加模式**：它们先写临时
目录再整体改名，`--output` 已存在时要么报错要么在 `--overwrite` 下整个删掉重建。所以正确做法是
**新批次单独转换成第二个数据集，再合并**。

### 合并的前置条件

`aggregate_datasets` 会逐项校验，任一不满足直接抛 `ValueError`：

| 必须一致 | 由哪个转换参数决定 |
|---|---|
| `fps` | `--fps` |
| `robot_type` | `--robot-type`（不传则来自 profile） |
| feature 键集合、dtype、shape | `--profile`（决定 `state_dim` 和相机集合）、`--include-velocity`、`--include-depth`、`--image-storage` |
| 视频的 `video.height` / `video.width` / `video.channels` / `video.fps` | `--image-height`、`--image-width`、`--fps` |

**允许不一致**：视频编码参数（`--video-codec`、`--crf`、`--gop`、`--preset` 等）。合并时逐字段比对，
不一致的键会被置成 `null` 并打 warning，不阻断合并。虽然允许，还是建议保持一致。

不要凭记忆去对这些参数，从第一批的 `meta/conversion_manifest.json` 里读回来：

```bash
python -c "import json; m=json.load(open('$DS/my_dataset/meta/conversion_manifest.json')); \
print(json.dumps({k:m[k] for k in ('profile','alignment','rgb_encoder','image_storage')}, indent=2, ensure_ascii=False))"
```

### 命令

```bash
lerobot-edit-dataset \
  --operation.type=merge \
  --operation.repo_ids='["local/my_dataset","local/my_dataset_b2"]' \
  --operation.roots='["'"$DS"'/my_dataset","'"$DS"'/my_dataset_b2"]' \
  --new_repo_id=local/my_dataset_v2 \
  --new_root=$DS/my_dataset_v2
```

或者用 Python API（列表参数不用跟 shell 引号搏斗，推荐）：

```python
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets

DS = Path("/mnt/robot_platform/datasets")
b1 = LeRobotDataset("local/my_dataset",    root=DS / "my_dataset")
b2 = LeRobotDataset("local/my_dataset_b2", root=DS / "my_dataset_b2")

merged = merge_datasets([b1, b2],
                        output_repo_id="local/my_dataset_v2",
                        output_dir=DS / "my_dataset_v2")
print(merged.meta.total_episodes, merged.meta.total_frames)
```

合并的语义（实测 4 episodes + 2 episodes → 6 episodes / 36 frames）：

- **源数据集不被修改**，结果写到新目录。所以磁盘要能同时放下三份，验证完再删源。
- episode 按列表顺序重新编号：第一个数据集保持 `0..N-1`，第二个接在后面。
- `index`、`task_index` 全部重映射；`tasks.parquet` 按任务文本去重，所以两批用不同 `--task`
  是没问题的，会变成两个独立的 task index。
- 统计量（`meta/stats.json`）在合并后的全集上重算。
- `--operation.concatenate_videos=false` / `--operation.concatenate_data=false` 可以保留
  每个源文件一个分片，不重新打包。默认打包，通常不用改。

### 合并不会带走 conversion_manifest.json

`meta/conversion_manifest.json` 是**本仓库转换脚本写的**，不是 LeRobot 的标准文件，合并时不会被
复制或拼接。它是每条 episode 唯一的审计记录（对齐模式、`audit.hold` 各项、`missing_topics`、
`unique_ratio` 等）。删掉源数据集之前先手工归档：

```bash
cp $DS/my_dataset/meta/conversion_manifest.json    $DS/my_dataset_v2/meta/conversion_manifest.b1.json
cp $DS/my_dataset_b2/meta/conversion_manifest.json $DS/my_dataset_v2/meta/conversion_manifest.b2.json
```

### 不合并也能一起训

`lerobot-train` 的 `--dataset.repo_id` 接受数据集列表，train.py 会把它们拼起来（只保留各数据集
**共有**的键）。如果只是想用两批数据训一次、不想多存一份合并结果，可以走这条路。要长期维护、
反复用的数据集还是合并成一个更清楚。

## 4. 拆分数据集

```bash
lerobot-edit-dataset \
  --operation.type=split \
  --operation.splits='{"train": 0.8, "val": 0.2}' \
  --repo_id=local/my_dataset --root=$DS/my_dataset \
  --new_root=$DS/my_dataset_splits
```

产出目录结构（`--new_root` 是**父目录**，每个 split 一个子目录）：

```text
my_dataset_splits/
├── train/   # repo_id = local/my_dataset_train
│   ├── data/  meta/  videos/
└── val/     # repo_id = local/my_dataset_val
    ├── data/  meta/  videos/
```

split 的 `repo_id` 由**源** `repo_id` 加后缀自动生成，`--new_repo_id` 在这个操作里被忽略
（会打 warning）。

### 按比例拆 vs 按 episode 编号拆

```bash
# 比例：值必须都是浮点数，总和 <= 1.0
--operation.splits='{"train": 0.8, "val": 0.2}'

# 显式编号：值是 episode 索引列表
--operation.splits='{"train": [0, 1, 3, 4, 5], "test": [2, 6]}'
```

**按比例拆是顺序切分，不打乱。** `{"train": 0.8, "val": 0.2}` 就是前 80% 给 train、
剩下的给 val（最后一个 split 会吃掉所有余数，所以不会丢 episode）。

这一点很重要：如果录制时是按任务、按时段、按操作员分批录的，顺序切分出来的验证集会系统性偏斜
——比如全部来自最后一天的录制。想要随机验证集，就自己抽索引：

```bash
python - <<'PY'
import json, random
n = json.load(open("/mnt/robot_platform/datasets/my_dataset/meta/info.json"))["total_episodes"]
idx = list(range(n)); random.Random(0).shuffle(idx)
k = int(n * 0.8)
print(json.dumps({"train": sorted(idx[:k]), "val": sorted(idx[k:])}))
PY
# 把输出贴到 --operation.splits='...'
```

其他约束：同一个 episode 不能出现在多个 split；空 split 会报错；比例模式下算出 0 条的 split
会被跳过并打 warning。

### 训练时不落盘的替代方案

只是要一个验证集的话，未必需要真的拆出目录：

```bash
lerobot-train ... --dataset.eval_split=0.1     # 每个 task 留出 10% episode 做离线评估
lerobot-train ... --dataset.episodes='[0,1,2]' # 只用指定 episode 训练
```

要固定、可复现、要分发给别人的划分，才用 §4 真正拆成目录。

## 5. 其他数据集编辑操作

都是 `lerobot-edit-dataset --operation.type=<op>`。

### 删除坏 episode

```bash
lerobot-edit-dataset \
  --operation.type=delete_episodes \
  --operation.episode_indices='[3, 17, 42]' \
  --repo_id=local/my_dataset --root=$DS/my_dataset \
  --new_repo_id=local/my_dataset_clean --new_root=$DS/my_dataset_clean
```

剩余 episode 会重新连续编号，视频分片重新编码，统计量重算。

### 改任务描述

```bash
# 全部改成同一句
--operation.type=modify_tasks --operation.new_task="pick up the red cube"

# 按 episode 改
--operation.type=modify_tasks --operation.episode_tasks='{"0": "task A", "5": "task B"}'
```

### 删特征列

```bash
--operation.type=remove_feature --operation.feature_names='["observation.velocity"]'
```

### 重算统计量

```bash
--operation.type=recompute_stats --operation.overwrite=true
```

默认 `--operation.skip_image_video=true`（跳过图像/视频统计，快很多）。手工改过 parquet 之后用。

### 重新编码视频 / 图像转视频

```bash
# 换编码参数重压（比如从 libx264 换到 AV1）
--operation.type=reencode_videos --operation.rgb_encoder.vcodec=libsvtav1 \
  --operation.rgb_encoder.crf=30 --operation.overwrite=true

# 用 --image-storage image 转出来的数据集，事后转成视频
--operation.type=convert_image_to_video --operation.num_workers=8
```

`reencode_videos` 是全量重压，100 条 episode 是小时级的，放 Slurm 里跑别开终端等。

## 6. 输出路径的坑（务必看）

**不传 `--new_root` 时，结果不会写在 `--root` 旁边，而是落到 HF 缓存目录
`$HF_LEROBOT_HOME/<repo_id>`（默认 `~/.cache/huggingface/lerobot/<repo_id>`）。**

实测：对 `--root=$DS/ds_del` 跑 `delete_episodes` 且不传 `--new_root`，源目录原封不动还是 4 条
episode，删好的 2 条 episode 版本静悄悄写进了 `~/.cache/huggingface/lerobot/local/ds_a/`。
命令返回 0，日志也正常，只是结果不在你以为的地方，而且会慢慢吃满系统盘（NAS 上的数据集动辄
几百 GB，缓存盘一般放不下）。

**每条编辑命令都显式写 `--new_root`。**

### 原地编辑会留备份

当 `--root` 和 `--new_root` 相同（且 `repo_id` / `new_repo_id` 相同）时是真正的原地修改，
LeRobot 会把原始数据集改名成 `<root>_old` 作为备份：

```bash
lerobot-edit-dataset --operation.type=delete_episodes --operation.episode_indices='[1,2]' \
  --repo_id=local/my_dataset --root=$DS/my_dataset \
  --new_repo_id=local/my_dataset --new_root=$DS/my_dataset
# 结果：$DS/my_dataset（已修改） + $DS/my_dataset_old（原始备份）
```

备份是**整份拷贝**，磁盘占用翻倍，确认无误后自己删。

### shell 引号

`lerobot-edit-dataset` 用 draccus 解析，列表和字典参数要传 JSON 字面量，整体用单引号包住：

```bash
--operation.episode_indices='[1, 2, 3]'
--operation.splits='{"train": 0.8, "val": 0.2}'
--operation.repo_ids='["local/a","local/b"]'
```

里面要插 shell 变量时得断开引号：`'["'"$DS"'/a"]'`。嫌麻烦就用 Python API（§3 有示例）。

## 7. 训练与推理

```bash
lerobot-train \
  --dataset.repo_id=local/my_dataset \
  --dataset.root=$DS/my_dataset \
  --policy.type=act \
  --output_dir=/mnt/robot_platform/jobs/<job-id> \
  --job_name=my_run \
  --batch_size=8 --steps=100000 --save_freq=10000
```

常用开关：`--resume=true`（续训）、`--seed`、`--num_workers`、`--policy.device`、
`--dataset.image_transforms.enable=true`（数据增强）、`--policy.push_to_hub=false`。
`--policy.type` 的完整取值用 `lerobot-train --help` 看，0.6.0 里包含 `act`、`diffusion`、
`smolvla`、`pi0`、`pi05`、`vqbet`、`tdmpc` 等。

平台上一般不手敲这条命令，而是走 LeLab 提交 Slurm 作业（`apps/lelab/lelab/train.py` 拼的就是
这条命令行）。手敲适合调参试跑。

推理与回放：

```bash
lerobot-rollout    # 跑策略（LeLab 依赖这个脚本，pyproject.toml 里把 lerobot 钉在 v0.6.0）
lerobot-eval       # 离线评估
lerobot-replay     # 回放某条 episode 的动作到真机
```

`lerobot-replay` 会驱动真实硬件，动手前确认工作空间没人。

## 8. 硬件相关（采集端）

只在接了机械臂的采集机上用，mgmt01 / gpu01 上跑不了：

```bash
lerobot-find-port        # 找机械臂串口
lerobot-find-cameras     # 枚举可用相机
lerobot-setup-motors     # 电机 ID 配置
lerobot-calibrate        # 关节标定
lerobot-find-joint-limits
lerobot-teleoperate      # 遥操作
lerobot-record           # 直接录成 LeRobot 数据集
```

本平台的数据是先录 rosbag、再用 `tool/` 下的脚本转换（见 `tool/CONVERSION.md`），
不走 `lerobot-record`。

## 9. 典型流程串起来

```bash
DS=/mnt/robot_platform/datasets

# 1) 新一批 rosbag 转成第二个数据集，参数与第一批完全一致
conda run -n lerobot tool/rdp convert \
  --input /path/to/new_bags --output $DS/my_dataset_b2 \
  --repo-id local/my_dataset_b2 --task "pick up the object" \
  --recipe mcap-dexhand

# 2) 核对两边 schema 一致
conda run -n lerobot lerobot-edit-dataset --operation.type=info --operation.show_features=true \
  --repo_id=local/my_dataset --root=$DS/my_dataset > /tmp/f1.txt
conda run -n lerobot lerobot-edit-dataset --operation.type=info --operation.show_features=true \
  --repo_id=local/my_dataset_b2 --root=$DS/my_dataset_b2 > /tmp/f2.txt
diff /tmp/f1.txt /tmp/f2.txt

# 3) 合并
conda run -n lerobot lerobot-edit-dataset --operation.type=merge \
  --operation.repo_ids='["local/my_dataset","local/my_dataset_b2"]' \
  --operation.roots='["'"$DS"'/my_dataset","'"$DS"'/my_dataset_b2"]' \
  --new_repo_id=local/my_dataset_v2 --new_root=$DS/my_dataset_v2

# 4) 归档两份转换审计记录
cp $DS/my_dataset/meta/conversion_manifest.json    $DS/my_dataset_v2/meta/conversion_manifest.b1.json
cp $DS/my_dataset_b2/meta/conversion_manifest.json $DS/my_dataset_v2/meta/conversion_manifest.b2.json

# 5) 抽查几条 episode
conda run -n lerobot lerobot-dataset-viz --repo-id local/my_dataset_v2 --root $DS/my_dataset_v2 \
  --episode-index 50 --save 1 --output-dir /tmp/rrd

# 6) 确认无误后删掉源数据集，然后训练
```
