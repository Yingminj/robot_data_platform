# 严格对齐转换工具

统一的数据语义是：

```text
observation.state = 手臂 joint_states 的实际关节状态
                  + 每个末端执行器在观测时刻的状态

action            = 观测之后的第一对 joint_cmd
                  + command 时刻已经生效的末端执行器指令
```

维度不再写死。状态与动作的宽度由 **robot profile** 推导：手臂关节数 + 各末端执行器自由度。
夹爪（1 自由度）与灵巧手（20 自由度）走同一条代码路径。

支持 rosbag2 的两种存储后端（sqlite3 `.db3` 与 MCAP `.mcap`），以及两种图像格式
（`sensor_msgs/Image` 与 `sensor_msgs/CompressedImage`）。图像格式由连接的消息类型自动识别，
无需手工配置。MCAP 自带 schema，因此不需要再手工注册 `marvin_msgs` 类型定义。

## Robot profile

`tool/robot_profile.py` 用声明式配置描述"机器人是什么"：话题名、末端执行器类型与自由度、相机映射。
内置两个 profile：

| profile | 图像 | 末端执行器 | state_dim |
|---|---|---|---|
| `marvin-gripper` | `sensor_msgs/Image` | `gripper` × 2（各 1 维，`std_msgs/Float32`） | 16 |
| `tj-dexhand`（默认） | `CompressedImage`（JPEG） | `dexhand` × 2（各 20 维，`JointState`） | 54 |

用 `--profile <名称或 JSON 路径>` 选择。自定义机器人写一个 JSON 即可，不需要改代码：

```json
{
  "name": "my-robot",
  "robot_type": "my-robot",
  "arm": {
    "joint_states_topic": "/tj/joint_states",
    "joint_names": ["Joint1_L", "...", "Joint7_R"],
    "command_topics": ["/tj/control/joint_cmd_A", "/tj/control/joint_cmd_B"],
    "command_dim": 7
  },
  "end_effectors": [
    {
      "name": "right_hand",
      "kind": "dexhand",
      "dim": 20,
      "command_topic": "/hand_right/joint_commands",
      "command_kind": "jointstate",
      "state_topic": "/hand_right/joint_states",
      "joint_names": ["right_finger1_joint1", "..."]
    }
  ],
  "cameras": {"top": "/head_camera/camera/color/image_raw/compressed"},
  "anchor_camera": "top"
}
```

末端执行器的 `kind` 取 `gripper`（dim 必须为 1）或 `dexhand`。`command_kind` / `state_kind`
取 `float32`（无 Header，只能用 record time）或 `jointstate`。

**没有 `state_topic` 时，观测会退化为回显指令**，此时该部分观测就是动作的副本，策略会学到恒等映射。
转换审计里的 `end_effector_state_source` 会把这种情况标成 `command_echo`，请优先在录制端补上状态话题。

## 对齐模式

- `capture`：图像按 `header.stamp` 有界最近邻，关节状态在网格时刻插值。强调物理采集时间，
  但会用到 tick 之后的数据（非因果），**仅建议用于诊断，不要用于训练集**。
- `lerobot-loop`（默认）：图像/state 取 tick 之前最新可用值，command 取 tick 之后第一条。
  模拟 LeRobot 在线录制的"后台相机最新帧 → 当前 state → teleop action"循环。

LeRobot 0.6 的官方 `record_loop()` 顺序是 `robot.get_observation()` → `teleop.get_action()` →
`robot.send_action()` → `dataset.add_frame()`。官方不做任何重采样：`timestamp` 列是
`frame_index / fps` 直接编造的，相机读取用 `read_latest()`（最多容忍 500 ms 陈旧帧）。
本工具在这个语义之上额外做了时间戳门禁。

## Episode 窗口：按遥操作活动切分

`joint_cmd` 通常只在使能状态下发布，一个 bag 里可能只有一小段是真正的遥操作。窗口规则：

- 起点：首条 `joint_cmd` **之前最近的一帧锚点相机图像**（`--grid-anchor anchor-camera`，默认），
  这样第 0 行就是一帧新鲜图像，同时把遥操作开始作为数据集起点
- 终点：最后一条 `joint_cmd`
- 窗口之外的数据全部丢弃
- 窗口内部的 `joint_cmd` 断档：保持最后一条已下发指令（zero-order hold）

`--grid-anchor` 三种取值：

| 取值 | 网格 | 锚点相机陈旧度 |
|---|---|---|
| `anchor-camera` | 从锚点相机帧起，固定 1/fps | 最大约一个相机周期 |
| `anchor-camera-ticks` | **直接以锚点相机帧时刻为 tick** | 恒为 0 |
| `first-command` | 从首条 joint_cmd 起，固定 1/fps | 最大约一个相机周期 |

当相机帧率与 `--fps` 接近时（例如 30 Hz 相机 + `--fps 30`），`anchor-camera` 的相位会锁死，
导致几乎每一行的图像都陈旧接近一整个周期。实测样例 bag：`anchor-camera` 的 top 相机年龄
p50 = 30.78 ms，而 `anchor-camera-ticks` 为 0.00 ms，帧数与命令延迟完全相同。
**相机是最慢的观测流时，建议用 `anchor-camera-ticks`。**

### hold 行的记账

⚠️ **hold 行表达的是"控制器保持上一条指令"，不是遥操作者的新意图。训练侧应按
`action_hold_mask` 过滤或降权这些行。**

- `timestamps/action_hold_mask`（bool 数组）
- `audit.hold`：`rows` / `fraction` / `max_run_s` / `segments` / `real_command_rows`
- `--max-hold-fraction` 与 `--max-hold-run-s` 可以直接拒绝 hold 过多的 episode
- 整集没有任何真实指令行时抛异常拒绝

## 1. rosbag2 → 对齐 HDF5

```bash
conda run -n lerobot python tool/rosbag2_to_hdf5_aligned.py \
  --input /path/to/rosbags \
  --output-dir /path/to/hdf5 \
  --profile tj-dexhand \
  --fps 30 \
  --alignment-mode lerobot-loop \
  --grid-anchor anchor-camera-ticks \
  --recursive --on-error skip
```

HDF5 除 ACT 字段外还保存 `timestamps/grid_ns`、每路源时间戳、`action_hold_mask`、
`state_names_json`、`state_dim`、`profile_json` 和完整审计。默认 `--compression gzip`。

## 2. rosbag2 → LeRobotDataset v3

```bash
conda run -n lerobot python tool/rosbag2_to_lerobotv3.py \
  --input /path/to/rosbags \
  --output /path/to/my_dataset \
  --repo-id local/my_dataset \
  --task "pick up the object" \
  --profile tj-dexhand \
  --fps 30 \
  --grid-anchor anchor-camera-ticks
```

**训练数据建议直接走这条路径，不要经过 HDF5。** 同一段数据实测：
gzip HDF5 273 MB，LeRobot v3（AV1 CRF 0 无损）7.6 MB，差 36 倍。
HDF5 适合做归档、ACT 兼容或调试。

## 3. HDF5 → LeRobotDataset v3

```bash
conda run -n lerobot python tool/hdf5_to_lerobotv3.py \
  --input /path/to/hdf5_folder \
  --output /path/to/my_dataset \
  --repo-id local/my_dataset \
  --task "pick up the object" \
  --fps 30
```

状态维度与名称从 HDF5 的 `state_dim` / `state_names_json` 属性读取，不再假定 16 维。
缺少 `schema_version` 属性（对齐来源未知）的文件默认拒绝，确认无误后可加
`--allow-unaligned-source`。只有明确传 `--allow-fps-override` 才允许覆盖已有 FPS。

整个 `action == qpos` 的数据不再被拒绝：脚本仅打印告警并照常转换，同时在
`meta/conversion_manifest.json` 中按 episode 记录 `action_equals_state`，
并在顶层汇总 `action_equals_state_episodes`。这类 episode 没有指令信号，
策略只会学到 `a_t = s_t`，请确认这是预期行为。需要恢复旧的严格行为时加 `--strict-action`。

转换前可用 `tool/check_hdf5_quality.py` 先体检 HDF5（`--layout` 可列出数据集结构）。

转换过程默认屏蔽 ffmpeg/libx264 与 tqdm 日志，改为显示进度块（episode 计数、已写帧数、
已用时间与 ETA、`action==state` 计数）。`--progress auto|bar|plain|none` 控制显示方式
（auto 在 TTY 下用进度条，重定向到文件时每 episode 一行）；`--verbose-encoder` 恢复原始编码器日志。

## 视频配置

两个 v3 脚本均通过 LeRobot 官方 `RGBEncoderConfig` 配置视频，默认与 LeRobot 一致：

```text
--video-codec libsvtav1   # 与 test_lerobot/REPORT.md 的基准一致
--video-pixel-format yuv420p
--crf 0                   # 0 = 无损
--gop 2                   # 与 LeRobot 默认一致，短 GOP 便于随机访问
```

⚠️ **CRF 数值不能跨编码器套用**：SVT-AV1 是 0–63，x264 是 0–51。
`test_lerobot/REPORT.md` 的 "CRF 20" 结论只在 `libsvtav1` 下成立。

CRF 0 消除量化损失，但 `yuv420p` 仍有色度下采样；要求 RGB 逐像素可逆请用
`--image-storage image`（PNG）或 `--video-pixel-format yuv444p`。

## 常用门禁参数

```text
--image-tolerance-ms          capture 默认半帧；lerobot-loop 默认 1.5 帧
--state-tolerance-ms          默认为实测 joint_states 周期的 1.5 倍
--action-tolerance-ms         默认一帧周期（30 FPS 时 33.33 ms）
--action-pair-tolerance-ms    默认 5 ms
--end-effector-tolerance-ms   默认 100 ms
--invalid-frame-policy fail   默认；可选 drop
--action-gap-policy hold-last-command   默认；可选 fail
--max-hold-fraction / --max-hold-run-s  hold 过多时拒绝 episode
--max-decode-errors 0         默认任何必需消息解析错误都失败
```

`--state-tolerance-ms` 默认自适应：以"tick 之前最新一帧"取值时，状态年龄天然分布在一个发布周期内，
固定阈值等于发布周期会因为普通抖动误杀。

`drop` 会压缩无效控制行之间的真实时间，只适合明确接受该行为的清洗流程；生产数据建议保持 `fail`。

命令流频率远高于 `--fps` 时（例如 500 Hz 命令 + 30 fps），可以把 `--action-tolerance-ms`
收紧到几毫秒，让它变成真正的门禁而不是形式。

## 审计字段

每个 episode 的审计写入 HDF5 的 `alignment_audit_json` 和 v3 的 `meta/conversion_manifest.json`：

| 字段 | 含义 |
|---|---|
| `window` | bag 时长、命令跨度、转换跨度、命令覆盖率、遥操作起点偏移 |
| `hold` | hold 行数/占比/最长连续时长/分段 |
| `unique_ratio` | 每路图像与命令被选中的去重率；明显低于 1 说明 `--fps` 高于该流真实频率 |
| `end_effector_state_source` | `measured` 或 `command_echo` |
| `image_formats` | 每路相机识别到的 `raw_image` / `compressed_image` |
| `tolerant_cdr_parses` | 需要容错 CDR 解析的消息数（见下） |
| `metrics_ms` | 命令延迟、A/B 偏差、末端执行器年龄、图像陈旧度的 mean/p50/p95/max |
| `source_period_ms` | 实测的 joint_states 发布周期 |

命令延迟统计只对真实指令行计算，hold 行不参与（否则会被断档宽度污染）。

## 容错 CDR 解析

部分发布者复用序列化缓冲区却没有截断长度，导致 `sensor_msgs/JointState` 消息尾部带上几个字节的
垃圾数据。ROS 2 自己的反序列化器会忽略尾部多余字节，但 `rosbags` 会直接断言失败。

本工具对 JointState 先用 `rosbags` 解析，失败时回退到 `tool/ros_messages.py` 里的容错 CDR
读取器，并在 `audit.tolerant_cdr_parses` 中记录每个话题的回退次数。

**这是录制端的 bug，回退只是让数据可读，请在录制端修复。** 该字段非零就说明问题仍然存在。

## 相机与话题覆盖

```bash
--camera top=/my/top/image --camera wrist=/my/wrist/image
--include-depth --depth top=/my/top/depth --depth wrist=/my/wrist/depth
--anchor-camera top
```

一旦指定 `--camera`，就整体替换 profile 中的相机映射。

多任务目录可以使用 `--task-map tasks.json`：

```json
{
  "episode_0001.hdf5": "pick up the red block",
  "rosbag2_2026_01_01-12_00_00": "place the cup"
}
```

每个 v3 输出都会在 `meta/conversion_manifest.json` 保存源文件、task、编码器配置和对齐审计。
