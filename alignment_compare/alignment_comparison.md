# 三种数据转换工具的对齐机制对比

## 工具概览

| 工具 | 对齐策略 | 断档处理 | 用途 |
|------|---------|---------|------|
| **rosbag2_to_hdf5_aligned.py** | 严格固定频率网格 + 多模态对齐 | 明确拒绝或删除无效帧 | rosbag2 → HDF5（带完整审计） |
| **rosbag2_to_act_hdf5.py** | 最近邻全局对齐 | 静默使用最近值，无门禁 | rosbag2 → ACT HDF5 |
| **rrd_to_lerobot.py** | Rerun时间轴对齐 + schema拼接 | shift+copy策略 | Rerun RRD → LeRobot |

---

## 对齐机制详细对比

### 1. rosbag2_to_hdf5_aligned.py（严格对齐）

```
时间轴模式：capture mode（物理采集时间）
================================================================================

Grid (30 FPS):    t0        t1        t2        t3        t4        t5
                  |         |         |         |         |         |
                  ├─────────┼─────────┼─────────┼─────────┼─────────┤
                  0ms      33ms      66ms      100ms     133ms     166ms

Image (header):   *      *    *           *  *              *     *
                  ├──────┤               缺失
                  ≤16ms  ✓reject          ✗

State (插值):     ○────○────○────○────○────○    (线性插值到网格点)
                       ≤10ms容差

Action (first):            A    A    缺失    A         A    A
                           └─↑  └─↑   ✗     └─↑      └─↑  └─↑
                  observation后第一条，≤33ms

Gripper:          G─────────────────────G──────────────────G
                  └────≤100ms────┘             (慢速topic，使用最近值)

断档示例：
Grid:            t0    t1    [断档]    t2    t3
Action:           A     A     (缺失)    A     A
                  ✓     ✓      ✗拒绝    ✓     ✓
处理：invalid_frame_policy = "fail" → 整个episode失败
      invalid_frame_policy = "drop" → 删除t1帧，压缩时间轴
```

**关键特性**：
- **固定网格**：`grid_ns = start + i * (1e9 / fps)`，严格 30Hz
- **Image**: 按 `header.stamp` 有界最近邻，默认容差 ≤16ms（半帧）
- **State**: 线性插值到网格点，容差 ≤10ms
- **Action**: 取 observation **之后第一条** command，容差 ≤33ms（一帧），双臂skew ≤5ms
- **Gripper**: 取 action 时刻最近值，容差 ≤100ms
- **断档拒绝**：任何帧缺失 action/image/state 时，默认 **整个 episode 失败**
- **审计完整**：输出时间戳、offset、skew、dropped frames 统计

---

### 2. rosbag2_to_act_hdf5.py（宽松对齐）

```
时间轴模式：全局最近邻（bag record timestamp）
================================================================================

Camera timestamps:  10ms   45ms   78ms   110ms  145ms  180ms
                     *      *      *       *      *      *
                     
Grid (inferred):    [---0---][---1---][---2---][---3---][---4---]
                    |       |       |       |       |       |
                    0      40      80     120     160     200ms
                          ↓       ↓       ↓       ↓       ↓
                    最近邻 argmin(|camera_t - grid_t|)
                    
Selected:            45ms    78ms   110ms   145ms  180ms
                     cam1    cam2    cam3    cam4    cam5

Joint State:         15ms        85ms            170ms
                      *           *                *
Grid对齐:            15ms───────85ms──────────────170ms
                      ↑           ↑               ↑
                    [复制]    [复制]          [复制]
输出:                 0=15ms  1=15ms  2=85ms  3=85ms  4=170ms

Action (joint_cmd):   20ms      90ms            175ms
                       A          A               A
对齐策略:              0=20ms  1=20ms  2=90ms  3=90ms  4=175ms
                      (与state独立，各自最近邻)

断档示例：
Grid:            0      1      2      3      4
Camera:          ✓      ✓     缺失    ✓      ✓
Joint State:     ✓     复制   复制    ✓      ✓
Action:          ✓     复制   复制    ✓      ✓

处理：找不到数据时，复制上一帧；没有上一帧时用零填充
```

**关键特性**：
- **推断网格**：从相机时间戳推断主时间轴，无固定FPS
- **全局最近邻**：`argmin(|timestamps - target|)`，**无容差上限**
- **复制填充**：找不到数据时，使用上一个有效值；开头缺失时用零
- **断档静默**：缺失数据不报错，直接复制最近帧
- **无审计**：不记录时间戳偏移、跳变、对齐质量
- **action = qpos 可能**：由于独立对齐，可能出现 action 完全等于 qpos

---

### 3. rrd_to_lerobot.py（Rerun对齐）

```
时间轴模式：Rerun aligned table（已对齐的时序表）
================================================================================

Rerun RRD (aligned table):
时间轴已固定为采集时的主频率（video2rrd时确定）

Column schema:
  /joint_states/position_L    [7-dim]
  /joint_states/position_R    [7-dim]
  /control/joint_cmd_A        [7-dim]
  /control/joint_cmd_B        [7-dim]
  /info/eef_left              [7-dim: xyz + quat]
  /info/gripper_feedback_L    [1-dim]

对齐已完成，只做向量拼接：

Frame:       0      1      2      3      4
State_L:     s0     s1     s2     s3     s4
State_R:     s0     s1     s2     s3     s4
Cmd_A:       a0     a1     a2     a3     a4
Cmd_B:       b0     b1     b2     b3     b4
EEF_L:       e0     e1     e2     e3     e4
Gripper_L:   g0     g1     g2     g3     g4

LeRobot输出拼接：
observation.state = [State_L, State_R, EEF_L, EEF_R, Gripper_L, Gripper_R]
action            = [Cmd_A, Cmd_B, EEF_L(shift), EEF_R(shift), Gripper_L(shift), Gripper_R(shift)]

Shift策略（部分action topic需要t+1）：
原始:    e0    e1    e2    e3    e4
Shift:   e1    e2    e3    e4   [e4复制]
         └──向前移动一帧──┘      └末尾复制

断档示例：
Frame:       0      1      2(缺失)   3      4
处理：Rerun aligned table已处理断档（可能插值/跳过）
     rrd_to_lerobot只读取表，不再处理断档
```

**关键特性**：
- **预对齐**：依赖 Rerun 的 `video2rrd` 已完成时间对齐
- **Schema拼接**：从多列向量按配置顺序水平拼接
- **Shift策略**：`eef_*` 和 `gripper_feedback_*` 向前移动一帧（t+1），末尾复制
- **无容差检查**：假设 Rerun table 已正确对齐
- **断档上游处理**：断档在 `video2rrd` 阶段处理，此工具不感知

---

## 断档处理策略对比

### 场景1：单帧数据完全缺失

```
Grid:    t0     t1     t2(缺)  t3     t4
Camera:   ✓      ✓      ✗      ✓      ✓
State:    ✓      ✓      ✗      ✓      ✓
Action:   ✓      ✓      ✗      ✓      ✓
```

| 工具 | 处理方式 | 结果 |
|------|---------|------|
| **rosbag2_to_hdf5_aligned** | `fail`: 抛异常，episode失败<br>`drop`: 删除t2，输出4帧 | episode拒绝 或 4帧(压缩时间) |
| **rosbag2_to_act_hdf5** | 复制t1的所有数据到t2 | 5帧，t2=t1副本 |
| **rrd_to_lerobot** | 上游处理（video2rrd） | 取决于上游策略 |

---

### 场景2：Action晚到（超过容差）

```
Grid:    t0     t1     t2     t3     t4
Obs:      ✓      ✓      ✓      ✓      ✓
Action:   ✓      ✓    (晚80ms) ✓      ✓
```

| 工具 | 容差 | 处理 |
|------|-----|------|
| **rosbag2_to_hdf5_aligned** | ≤33ms | t2 action超时 → 拒绝t2帧 |
| **rosbag2_to_act_hdf5** | 无限制 | 使用晚到的action，不报错 |
| **rrd_to_lerobot** | N/A | 表中action已对齐 |

---

### 场景3：双臂command时间差大

```
Grid:       t0      t1      t2
Cmd_A:      100ms   133ms   166ms
Cmd_B:      100ms   145ms   166ms  (t1时相差12ms)
```

| 工具 | 容差 | 处理 |
|------|-----|------|
| **rosbag2_to_hdf5_aligned** | skew ≤5ms | t1拒绝（12ms超限） |
| **rosbag2_to_act_hdf5** | 无检查 | 接受12ms skew |
| **rrd_to_lerobot** | N/A | 假设已同步 |

---

### 场景4：Image容差超限

```
Grid:    t0=0ms     t1=33ms    t2=66ms
Image:   5ms        28ms       90ms(超限)
```

| 工具 | 容差 | 处理 |
|------|-----|------|
| **rosbag2_to_hdf5_aligned** | ≤16ms (capture)<br>≤50ms (lerobot-loop) | t2拒绝（偏移24ms） |
| **rosbag2_to_act_hdf5** | 无限制 | 接受90ms图像 |
| **rrd_to_lerobot** | N/A | 使用表中图像 |

---

## 数据质量保证对比

| 维度 | rosbag2_to_hdf5_aligned | rosbag2_to_act_hdf5 | rrd_to_lerobot |
|------|------------------------|--------------------|--------------------|
| **FPS保证** | 严格固定网格 | 推断，不保证 | 继承上游 |
| **Action语义** | 观测后第一条 | 最近邻 | 配置化拼接 |
| **拒绝无效数据** | ✅ 默认拒绝 | ❌ 静默填充 | ⚠️ 上游责任 |
| **时间戳审计** | ✅ 完整记录 | ❌ 无 | ⚠️ 有限 |
| **action=qpos检测** | ✅ 明确拒绝 | ❌ 可能发生 | ⚠️ 取决于schema |
| **容差可配置** | ✅ 所有参数 | ❌ 固定逻辑 | ❌ N/A |
| **错误可追溯** | ✅ 完整metrics | ❌ 无 | ⚠️ Rerun日志 |

---

## 使用场景建议

### rosbag2_to_hdf5_aligned.py
**适用**：
- 需要高质量训练数据，严格控制对齐误差
- 需要审计源时间戳和对齐质量
- 后续会转换为LeRobot或其他格式
- 录制质量不确定，需要明确拒绝坏数据

**不适用**：
- 接受"尽力而为"的数据质量
- 无法重新录制，必须用所有数据

### rosbag2_to_act_hdf5.py
**适用**：
- 快速原型，容忍对齐误差
- 数据已知质量较好，topic连续
- ACT训练已验证该对齐策略有效

**不适用**：
- 需要严格时间同步（如高速操作）
- 需要审计数据质量
- Topic有频繁丢帧或延迟

### rrd_to_lerobot.py
**适用**：
- 使用Rerun/Kscale生态录制数据
- 需要灵活的schema配置（手、夹爪、EEF）
- 时间对齐已在video2rrd完成

**不适用**：
- 直接从rosbag转换（需先转rrd）
- 需要自定义对齐策略
- 不信任上游对齐质量

---

## 关键差异总结

| 特性 | Aligned | ACT | RRD |
|------|---------|-----|-----|
| 断档处理 | 拒绝/删除 | 复制 | 上游 |
| 对齐保证 | 容差门禁 | 尽力而为 | 信任上游 |
| 时间轴 | 固定网格 | 推断 | 继承 |
| Action语义 | obs后first | 最近邻 | 可配置shift |
| 错误报告 | 详细审计 | 静默 | 依赖日志 |
| 生产就绪 | ✅ | ⚠️ | ⚠️ |
