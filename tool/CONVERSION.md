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
无需手工配置。

MCAP 自带 schema；**rosbag2 sqlite3（format version 5）不带**，`.db3` 里只有类型名称，
没有消息定义，直接打开会报 `Bag contains no type definitions`。因此自定义类型由 profile 的
`message_definitions` 提供（`类型名 -> .msg 定义文本`），内置 profile 已经声明了
`marvin_msgs/msg/Jointcmd` 与 `marvin_msgs/msg/JointcmdArm`（两者布局相同，只是改了名）。
标准 ROS 2 消息来自内置 Humble typestore。该 typestore 只在 bag 自身没有定义时才被使用，
所以对 MCAP 不会产生任何覆盖。

## Robot profile

`tool/robot_profile.py` 用声明式配置描述"机器人是什么"：话题名、末端执行器类型与自由度、相机映射。
内置四个 profile：

| profile | 手臂话题 | 图像 | 相机 | 末端执行器 | state_dim |
|---|---|---|---|---|---|
| `marvin-gripper` | `/joint_states`, `/control/joint_cmd_A\|B` | `sensor_msgs/Image` | 3 | `gripper` × 2（各 1 维，`std_msgs/Float32`） | 16 |
| `marvin-dexhand` | 同上 | `sensor_msgs/Image` | 3（`/camera/camera`, `/wrist_*`） | `dexhand` × 2（各 20 维，`JointState`） | 54 |
| `marvin-dexhand-head` | 同上 | `sensor_msgs/Image` | 1（仅 `/camera/camera`） | `dexhand` × 2 | 54 |
| `tj-dexhand`（默认） | `/tj/joint_states`, `/tj/control/joint_cmd_A\|B` | `CompressedImage`（JPEG） | 3（`/head_camera`, `/wrist_*_camera`） | `dexhand` × 2 | 54 |

另有 `tool/profiles/marvin-gripper-quadtile.json`（非内置，按路径引用）：手臂/夹爪与 `marvin-gripper`
相同，但三路相机合成在单个 `/quad_tile/compressed` 话题里，靠 `camera_tiles` 拆分，见下文。

相机数量不同必须用不同 profile：LeRobot 数据集要求每个 episode 暴露相同的特征集，
所以"只有头部相机"的录制和"三相机"的录制是两个数据集，不能混在一起转换。

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

### 拼接图（mosaic）拆分：`camera_tiles`

新版录制把多路相机合成到**单个** `/quad_tile/compressed` 话题里，而不是每路相机一个话题。
此时多个相机共用同一个 topic，各自用 `camera_tiles` 声明自己占据拼接图的哪一块：

```json
"cameras": {
  "top":     "/quad_tile/compressed",
  "wrist_L": "/quad_tile/compressed",
  "wrist_R": "/quad_tile/compressed"
},
"camera_tiles": {
  "top":     {"left": 0.0, "top": 0.0,      "right": 1.0, "bottom": 0.666666, "width": 640, "height": 480},
  "wrist_L": {"left": 0.0, "top": 0.666666, "right": 0.5, "bottom": 1.0,      "width": 640, "height": 480},
  "wrist_R": {"left": 0.5, "top": 0.666666, "right": 1.0, "bottom": 1.0,      "width": 640, "height": 480}
}
```

边界是**比例**而不是像素，和部署端的拆分函数
（`Apex_Deploy_new/robot_node/vlahost/vlahost/lebot_client.py:split_hero3_image`）保持一致：
拼接图在送到消费者之前可能被整体缩放，比例写法能扛住这种缩放。`width`/`height` 是裁剪后
resize 的目标尺寸，缩放用 `INTER_LINEAR`，与部署端 `_resize_camera` 相同，**保证训练帧和推理
帧经过同一条滤波路径**。

`marvin-gripper-quadtile`（`tool/profiles/marvin-gripper-quadtile.json`）就是这样一个 profile，
对应 realsense 节点的 hero3 布局（`components/realsense/config/realsense.yaml`：
`top_count: 1`, `top_height: 960`, 输出 1280x1440）：

```text
[            head  1280x960            ]   <- 原生 640x480 的 2 倍放大
[ wrist_left 640x480 | wrist_right 640x480 ]   <- 原生尺寸
```

注意**头部相机在拼接图里是 2 倍放大的**，不是 640x480。按 640x480 直接切三块会切出头部画面的
四分之一角落，得到完全错误的图像。两路腕部相机才是原生尺寸，对它们而言 resize 是空操作。

拼接图每帧只解码一次，再按各相机的 tile 裁剪，所以三个相机不会带来三倍解码开销。
共用 topic 的相机如果没有全部声明 `camera_tiles`，profile 会直接报错——否则它们会是同一张图。

**能录per-camera话题就不要录拼接图**：realsense 节点其实已经在发
`/head_camera|wrist_left_camera|wrist_right_camera/camera/color/image_raw/compressed`
（`publish.per_camera_compressed: true`），只是在
`UI_node/.../config/recording_topics_gripper.yaml` 里被注释掉了。per-camera 话题是原生分辨率、
每路相机独立的采集时间戳；拼接图把三路强行压到一个 30 Hz 的合成时间戳上，腕部相机原本 60 fps
的独立性会**永久丢失**，而且 JPEG 是在放大之后才编码的，头部画质也更差。

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

- 起点：首条 `joint_cmd` **之前最近的一帧锚点相机图像**，
  这样第 0 行就是一帧新鲜图像，同时把遥操作开始作为数据集起点
- 终点：最后一条 `joint_cmd`
- 窗口之外的数据全部丢弃
- 窗口内部的 `joint_cmd` 断档：按 `--action-gap-policy` 处理（见下）

### 断档处理：`--action-gap-policy`

| 取值 | 断档时的 action | 适用场景 |
|---|---|---|
| `hold-last-command`（默认） | 保持最后一条已下发指令（zero-order hold） | 命令流连续、只有偶发抖动 |
| `joint-state-fill` | 用**该手臂自己实测的 `joint_states`** 填充 | 旧版 `.db3`：遥操作逻辑会让某条手臂的 `joint_cmd` 整段静默 |
| `fail` | 拒绝该 episode | 需要严格门禁的生产数据 |

`joint-state-fill` 是按手臂分别填充的：`arm.command_topics` 中第 *i* 个话题固定对应
`joint_names[i * command_dim : (i+1) * command_dim]` 这一段列（由 `ArmSpec` 保证），
所以某条手臂静默时只填它自己的列，另一条手臂的真实指令不受影响。

被填充的列在这些行上就是观测的恒等副本。因此该策略下的行判定也随之改变：
只要**任意一条**手臂有真实指令，该行就算有效遥操作意图（否则单臂 episode 会被整体屏蔽）。

- `timestamps/action_hold_mask`：该策略下标记的是"**没有任何**手臂下发指令"的行
- `audit.hold.joint_state_fill_rows`：每条手臂被 joint_states 填充的行数
- `audit.hold.any_arm_real_command_rows` / `real_command_rows`：任意手臂 / 全部手臂有真实指令的行数

⚠️ 某条手臂**全程**被填充时（例如实测样例 bag 的左臂 637/637 行），它的动作列对策略而言
就是恒等映射，只能表达"保持当前位姿"。确认该手臂确实是停放状态再用于训练——
实测样例中左臂在整个窗口内的关节变化仅 0.0056 rad，属于传感器噪声量级。

### 整段缺失的话题：`--missing-topic-policy`

`--action-gap-policy` 处理的是窗口**内部**的断档。但同一台机器不同批次的录制，
经常会有 profile 声明、录制里却**一条消息都没有**的话题：某次任务只用右手，
`/control/gripperValueL` 就整段为空；某代录制根本没有 `/gripper/feedback_*`。
为每种组合单独写一个 profile 会让 `state_dim` 在批次之间漂移——而 LeRobot 数据集
要求所有 episode 的特征集完全一致，schema 一漂移就只能拆成多个数据集。

因此话题按"缺了能不能补"分三类：

| 类别 | 话题 | 缺失时 |
|---|---|---|
| 必需 | `joint_states`、所有相机 | 一律拒绝——没有诚实的替代品 |
| 可重建 | `joint_cmd_*`、末端执行器指令 | `--missing-topic-policy fill` 时由实测状态重建 |
| 可选 | 末端执行器 `state_topic`（实测反馈） | 自动降级为 command echo，不影响转换 |

`--missing-topic-policy` 取值：

| 取值 | 行为 |
|---|---|
| `fail`（默认） | 任何可重建话题为空/缺失都拒绝该 episode |
| `fill` | 由实测状态重建，必须同时用 `--action-gap-policy joint-state-fill` |

`fill` 的重建来源与 `joint-state-fill` 完全同源，只是把"某一段"扩展成"整集"：

- 手臂 `joint_cmd_X` 整段为空 → 用该手臂自己的 `joint_states` 列填充全部行，
  计入 `audit.hold.joint_state_fill_rows`
- 末端执行器指令话题整段为空 → 用它自己的实测 `state_topic` 填充，
  记为 `audit.end_effector_action_source = "state_fill"`
- 末端执行器指令与实测**都**没有 → 仍然拒绝，没有可用来源

`audit.missing_topics` 逐条记录缺了什么、被什么替代：

```json
"missing_topics": {
  "/control/gripperValueR": {"status": "empty",  "filled_from": "/gripper/feedback_R (measured position)"},
  "/gripper/feedback_L":    {"status": "absent", "filled_from": "/control/gripperValueL (command echo)"}
}
```

⚠️ 与 `joint-state-fill` 相同的告诫：**重建出来的动作列是观测的恒等副本**，
只能表达"保持当前位姿"。用于训练前请确认该自由度在整集内确实是停放状态
（`check_rosbag_quality.py` 会给出该话题的实际活动区间）。

`--grid-anchor` 三种取值：

| 取值 | 网格 | 锚点相机陈旧度 |
|---|---|---|
| `anchor-camera-ticks`（默认） | **直接以锚点相机帧时刻为 tick** | 恒为 0 |
| `anchor-camera` | 从锚点相机帧起，固定 1/fps | 最大约一个相机周期 |
| `first-command` | 从首条 joint_cmd 起，固定 1/fps | 最大约一个相机周期 |

当相机帧率与 `--fps` 接近时（例如 30 Hz 相机 + `--fps 30`），`anchor-camera` 的相位会锁死，
导致几乎每一行的图像都陈旧接近一整个周期。实测样例 bag：`anchor-camera` 的 top 相机年龄
p50 = 30.78 ms，而 `anchor-camera-ticks` 为 0.00 ms，帧数与命令延迟完全相同。
因此 `anchor-camera-ticks` 是两个转换脚本的默认值。注意此时行数由锚点相机的实际帧率决定，
`--fps` 只用于写入元数据与容差换算。

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

旧版 `.db3`（`marvin-gripper` 一代）需要加 `--action-gap-policy joint-state-fill`，
因为这批录制里两条手臂是**交替**遥操作的，某条手臂会整段没有 `joint_cmd`。
如果这批数据里还有整段为空的指令话题（单手任务、缺 `gripperValue*`），
再加上 `--missing-topic-policy fill`，一个 profile 就能覆盖整个语料，
不必按批次拆 profile：

```bash
conda run -n lerobot python tool/rosbag2_to_lerobotv3.py \
  --input /path/to/gift_bags \
  --output /path/to/my_dataset \
  --profile marvin-gripper \
  --action-gap-policy joint-state-fill \
  --missing-topic-policy fill \
  --end-effector-tolerance-ms 300 \
  --fps 30 --on-error skip
```

`--end-effector-tolerance-ms` 需要放宽是因为夹爪指令值不变时发布端会停发：
实测 `/control/gripperValueL` 名义 20 Hz，夹爪停在闭合位时出现过 250 ms 的间隔。

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
维度不写死：state 宽度、手臂维度、各末端执行器的切片都从文件的 `state_dim` /
`state_names_json` / `profile_json` 属性读取，因此 16 维夹爪与 54 维灵巧手都能正确检查；
缺少这些属性的旧文件退化为按 `action` 实际宽度检查。

## 0. 转换前：体检 rosbag

```bash
conda run -n lerobot python tool/check_rosbag_quality.py /path/to/rosbag --profile marvin-dexhand
```

检查器与转换脚本共用同一份 profile 和同一个 reader，因此两者对"这个录制应该包含什么"
的判断永远一致，`.db3` 与 `.mcap` 都能读。只读时间戳、不解码图像，多 GiB 的 bag 几秒完成。
标称频率由实测中位周期得出，不写死，所以 30 Hz 相机与 500 Hz 关节流用同一套阈值。

命令话题（`joint_cmd`、末端执行器指令）只在使能时发布，其断档是操作行为而不是丢帧，
因此不参与丢帧评分，改为在"遥操作命令活动区间"一节里按区间列出。这一节还会直接指出
**两臂交替遥操作**的情况：如果某个手臂的静默区间覆盖了整个重叠窗口，
转换必然产出零行，检查器会提前报 FAIL 而不是让转换抛出难以理解的错误。

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
--action-gap-policy hold-last-command   默认；可选 joint-state-fill / fail
--missing-topic-policy fail   默认；可选 fill（需配合 joint-state-fill）
--max-tick-rate-deviation 0.1 anchor-camera-ticks 下 tick 频率与 --fps 的最大相对偏差
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
| `missing_topics` | profile 声明但录制未提供的话题，及其重建来源 |
| `end_effector_state_source` | `measured` 或 `command_echo` |
| `end_effector_action_source` | `command` 或 `state_fill`（指令话题整段为空，由实测状态重建） |
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
