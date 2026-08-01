# 严格对齐转换工具

这组工具面向 `marvin_msgs/msg/Jointcmd`，统一的数据语义是：

```text
observation.state = /joint_states 的 14 维实际关节状态
                  + 观测时刻已经生效的左右夹爪值

action            = 观测之后的第一对 /control/joint_cmd_A/B
                  + command 时刻已经生效的左右夹爪值
```

`Jointcmd.header.stamp` 和 `Image.header.stamp` 可用于采集时间对齐；
`std_msgs/Float32` 没有 Header，因此夹爪只能使用 rosbag record timestamp。缺少 action、
维度错误、双臂 command 时间差过大或跨越缺口时，默认拒绝 episode，绝不使用 qpos 代替 action。

## 对齐模式

- `capture`：默认用于 rosbag→HDF5。图像按 `header.stamp` 有界最近邻，关节状态在固定网格时刻插值，
  command 取 observation 之后第一条。这种模式强调物理采集时间。
- `lerobot-loop`：默认用于 rosbag→LeRobot v3。固定网格使用 bag record time，图像/state 取 tick
  之前最新可用值，command 取 tick 之后第一条。这种模式模拟 LeRobot 在线录制的“后台相机最新帧 →
  当前 state → teleop action”循环。

LeRobot 0.6 的官方 `record_loop()` 实际顺序是 `robot.get_observation()` →
`teleop.get_action()` → `robot.send_action()` → `dataset.add_frame()`；dataset 中保存的是处理后的
teleop action。这里的 `/control/joint_cmd_A/B` 按你确认的语义映射到该 action。

两种模式都生成严格固定的 `frame_index / fps` 控制行。所有误差上限均可通过 CLI 修改。

## 1. rosbag2 → 对齐 HDF5

单个 bag：

```bash
conda run -n aloha python tool/rosbag2_to_hdf5_aligned.py \
  --input /path/to/rosbag_episode \
  --output-dir /path/to/hdf5 \
  --fps 30 \
  --alignment-mode capture
```

批量目录：

```bash
conda run -n aloha python tool/rosbag2_to_hdf5_aligned.py \
  --input /path/to/rosbags \
  --output-dir /path/to/hdf5 \
  --recursive \
  --sort-by name \
  --output-name-template 'episode_{index:06d}.hdf5' \
  --on-error skip
```

新 HDF5 除 ACT 字段外还保存：

- `timestamps/grid_ns`
- 每路选中图像、joint state、joint command、observation/action gripper 的源时间戳
- `fps`、topic mapping、alignment mode 和完整 QA 指标

默认保留相机原分辨率。只有同时传 `--image-height`、`--image-width` 才做 letterbox resize。

## 2. rosbag2 → LeRobotDataset v3

```bash
conda run -n lerobot python tool/rosbag2_to_lerobotv3.py \
  --input /path/to/rosbags \
  --output /path/to/my_dataset \
  --repo-id local/my_dataset \
  --task "pick up the object" \
  --fps 30 \
  --alignment-mode lerobot-loop \
  --video-codec h264 \
  --crf 0
```

输入目录中的每个 rosbag 对应一个 LeRobot episode。脚本使用 LeRobot 0.6 官方
`LeRobotDataset.create()`、`add_frame()`、`save_episode()`、`finalize()`，不会自行写 Parquet/MP4。

## 3. HDF5 → LeRobotDataset v3

```bash
conda run -n lerobot python tool/hdf5_to_lerobotv3.py \
  --input /path/to/hdf5_folder \
  --output /path/to/my_dataset \
  --repo-id local/my_dataset \
  --task "pick up the object" \
  --fps 30 \
  --video-codec h264 \
  --crf 0
```

目录内每个 `.hdf5`/`.h5` 对应一个 episode。改进版 HDF5 的 `fps` 属性必须与 `--fps` 一致；
旧 HDF5 没有 FPS 时使用 CLI 值。只有明确传 `--allow-fps-override` 才允许覆盖已有 FPS。

脚本会拒绝整个 `action == qpos` 的旧错误数据。

## 视频配置

两个 v3 脚本均通过 LeRobot 官方 `RGBEncoderConfig` 配置视频：

```text
--video-codec h264       # 默认；也可选择 LeRobot 0.6 支持的其他 codec
--video-pixel-format yuv420p
--crf 0                  # 默认
--gop 2
--preset ...
--encoder-threads ...
```

当前环境实测 H.264 日志为 `rc=cqp qp=0 / Avg QP 0.00`，LeRobot metadata 也记录
`video.crf = 0.0`。注意 CRF 0 消除量化损失，但默认 `yuv420p` 仍包含色度下采样；如果要求 RGB
逐像素可逆，请使用 `--image-storage image` 保存 PNG。`--video-pixel-format yuv444p` 可以避免 4:2:0
色度下采样，但仍会经过 RGB↔YUV 变换。

## 常用门禁参数

```text
--image-tolerance-ms          capture 默认半帧；lerobot-loop 默认 1.5 帧
--state-tolerance-ms          默认 10 ms
--action-tolerance-ms         默认一帧周期（30 FPS 时 33.33 ms）
--action-pair-tolerance-ms    默认 5 ms
--gripper-tolerance-ms        默认 100 ms
--invalid-frame-policy fail   默认；可选 drop
--max-decode-errors 0         默认任何必需消息解析错误都失败
```

`drop` 会压缩无效控制行之间的真实时间，只适合明确接受该行为的清洗流程；生产数据建议保持 `fail`，
先修复录制或调整有依据的容差。

action 使用 observation 之后的第一条命令；如果某一侧命令超过一帧仍未出现，或 A/B 的时间差超过
5 ms，该控制行默认失败。因此不会在 joint_cmd 缺失时退化为 qpos，也不会静默拿更晚控制周期的命令。

相机和深度 topic 可重复配置：

```bash
--camera top=/my/top/image --camera wrist=/my/wrist/image
--include-depth --depth top=/my/top/depth --depth wrist=/my/wrist/depth
```

多任务目录可以使用 `--task-map tasks.json`：

```json
{
  "episode_0001.hdf5": "pick up the red block",
  "rosbag2_2026_01_01-12_00_00": "place the cup"
}
```

每个 v3 输出都会在 `meta/conversion_manifest.json` 保存源文件、task、编码器配置和对齐审计。
