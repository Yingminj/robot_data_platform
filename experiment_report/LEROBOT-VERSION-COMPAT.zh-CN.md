# LeRobot 分支兼容性：训练端（v0.6.0）与推理端（v0.5.2）

[English](LEROBOT-VERSION-COMPAT.md) | **简体中文**

审计日期：2026-08-12

## 问题

训练与开发使用 `robot_data_platform/lerobot`（`pyproject.toml` 版本号 `0.6.0`），
推理使用 `lerobot_vlahost`（`pyproject.toml` 版本号 `0.5.2`）。这个版本差会不会带来问题？

## 一段话结论

**对目前实际训练并部署的两种策略——ACT 和 Vita——不会。** 两个仓库都是上游 LeRobot 的 fork，
彼此只差几个上游提交；版本号（`0.5.2` / `0.6.0`）夸大了实际差距。checkpoint 加载所涉及的全部代码
（`configs/policies.py`、整个 `processor/` 包、`datasets/` v3.0 格式）在两边**逐字节相同**，
平台训练出的 checkpoint 在两个 fork 下推理输出**逐比特一致**。真正的风险不在版本号，而在于：
(1) 有四个策略只存在于训练端 fork；(2) `train_config.json` 确实无法跨端解析；
(3) `lelab-venv` 里还有第三份 LeRobot，版本比前两份都旧。

---

## 1. 实际比较的对象

| | 训练 / 开发 | 推理 |
|---|---|---|
| 路径 | `/home/kewei/YING/robot_data_platform/lerobot` | `/home/kewei/YING/lerobot_vlahost` |
| remote | `github.com/Yingminj/lerobot_dev`（另有 `upstream` = huggingface/lerobot） | `github.com/tangyubbb/lerobot_vlahost` |
| 分支 / 版本 | `dev` @ `27fbaee7`（`v0.6.0-2-g27fbaee7`） | `dev-yw` @ `79cc122` |
| 声明版本 | `0.6.0` | `0.5.2` |
| 工作区 | 有改动：未跟踪的 `policies/act_delta/` | 未跟踪的 `record_chunk.txt` |

两个仓库的 **git 历史互不相关**（vlahost 只有 30 个提交，且没有上游 tag），因此下文的比较是对
`src/lerobot` 做文件系统 diff，而不是 `git diff`。

注意：两边的 `src/lerobot/__version__.py` 都是从**已安装包的元数据**读取 `__version__`，而不是
从源码树读取。因此 `lerobot.__version__` 报告的是当前 venv 安装时的版本，**不能**用来区分两个
fork。要区分请改看是否存在 `policies/vita` / `policies/act_delta` 目录。

## 2. 分歧是双向的，而且大部分并非上游版本漂移

两个仓库互相都不是对方的超集，各自 fork 之后都加了本地改动：

**只在推理端（`lerobot_vlahost`）**
- `Marvin_sdk_pro/`、`robots/marvain_m6{,_http,_hybrid}/` —— 真正的机器人驱动
- `rollout/inference/chunk.py` 及 `ChunkInferenceConfig`、`send_action_chunk`、
  引擎基类上的 `produces_chunks`、`strategies/core.py` 中的 `go_home()` 回退逻辑
- `scripts/lerobot_replay_chunk.py`
- ACT 的 `loss_time_decay` / `loss_front_weight`（时间加权 L1，仅训练侧生效）
- `policies/vita/modeling_vita.py` 中的一处修复：当 rollout 引擎直接调用
  `predict_action_chunk` 时，为 `observation.state` 补上 observation-step 轴

**只在训练端（`robot_data_platform/lerobot`）**
- 策略 `act_delta`、`evo1`、`fastwam`、`lingbot_va`
- 上游 v0.6.0 的**深度相机特性集**：`datasets/depth_utils.py`、
  `DepthEncoderConfig`/`RGBEncoderConfig`、`depth_output_unit`、`is_depth_map` 特征标记
- HF Jobs 支持（`jobs/`、`JobConfig`）
- Foxglove 可视化（`utils/foxglove_visualization.py`、`display_mode`）
- `decode_video_frames_pyav` 中的 pyav seek 修复（改为基于 stream 的 seek，
  替代原来的 `int(first_ts * av.time_base)`）

"0.5.2 → 0.6.0" 的差异绝大部分是深度相机改造加 HF Jobs。**这些都不触及 RGB 推理链路。**

## 3. checkpoint 加载相关代码完全相同

以下文件在两个仓库中逐字节一致，这正是 checkpoint 能干净迁移的原因：

- `configs/policies.py` —— `PreTrainedConfig.from_pretrained`
- 整个 `processor/` 包 —— 每个 pipeline step、其注册名、以及序列化格式
- `policies/act/processor_act.py`
- 两边的 `datasets` 都是 `CODEBASE_VERSION = "v3.0"`

来自
`/mnt/robot_platform/jobs/act_tidy_up_stationery_le_batch_4_.../checkpoints/004000/pretrained_model`
的真实 ACT checkpoint 只用到通用 step：

```
preprocessor : rename_observations → to_batch → device → normalizer
postprocessor: unnormalizer → device
```

其中每一个在两个 fork 中都能解析。

### 已验证：推理输出逐比特一致

同一 checkpoint、同一固定随机种子的合成观测、CPU、经过完整前后处理 pipeline 的 `select_action`，
分别在两个 fork 下运行：

| checkpoint | 训练端 fork（v0.6.0） | 推理端 fork（v0.5.2） |
|---|---|---|
| ACT `act_tidy_up_stationery_le_batch_4` step 4000 | `sha256 8878961073bde336…`，sum `-3.1512776017` | **完全一致** |
| Vita `vita_tidy_up_stationery_le_batch_3` step 1000 | `sha256 e6eea1d23af2934f…`，sum `-7.5564867780` | **完全一致** |

数据集读取同样一致。`/mnt/robot_platform/datasets/express` 的 episode 0、frame 5，在两个 fork 下
分别用默认 torchcodec 后端和 pyav 后端读取：`action`、`observation.state`、`observation.velocity`
以及三路相机张量全部相同。

### 为什么 ACT 的 config 差异无害

推理端的 `ACTConfig` 多出两个字段（`loss_time_decay`、`loss_front_weight`），训练端没有。
方向很关键：平台写出的 `config.json` **不含**这两个字段，推理端的 dataclass 会退回默认值
（`0.0` / `1.0`），恰好等于关闭该特性。这两个字段只在 `compute_loss` 中使用，推理时不参与计算。

反方向会出问题——draccus 拒绝未知字段（见第 5 节）——因此
**在 `lerobot_vlahost` 里训练出的 checkpoint 无法被平台 fork 加载。** 训练请始终留在平台侧。

## 4. 分策略可移植性（平台训练的 checkpoint → vlahost 推理）

| 策略 | 代码差异 | 仅平台侧存在的 config 字段 | 可移植？ |
|---|---|---|---|
| **act** | 2 个文件不同（loss 代码 + 空白字符） | ——（平台侧是严格子集） | ✅ 已验证逐比特一致 |
| **vita** | 1 个文件不同（vlahost 多一处 rollout 修复） | —— | ✅ 已验证逐比特一致 |
| diffusion、eo1、gaussian_actor、multi_task_dit、pi0、pi05、pi0_fast、rtc、smolvla、tdmpc、vqbet、wall_x、xvla | 完全相同 | —— | ✅ |
| vla_jepa | 2 个文件不同（modeling、qwen_interface） | —— | ⚠️ config 可加载；权重/前向未验证 |
| **molmoact2** | config 与 processor 均不同 | `joint_offsets`、`joint_signs` | ❌ config.json 会被拒绝 |
| **groot** | 10 个文件不同——实现本身就不一样（`groot_n1_7.py` vs `groot_n1.py` + `eagle2_hg_model/`），processor 差异 2584 行 | `use_relative_actions`、`relative_exclude_joints`、`action_decode_transform`、`num_inference_timesteps`、`rtc_ramp_rate`、`model_path`、`model_params_fp32`、`use_flash_attention`、`tune_vlln`、`tune_top_llm_layers` | ❌ 不可移植 |
| **act_delta**、**evo1**、**fastwam**、**lingbot_va** | vlahost 中不存在 | —— | ❌ `DecodingError: Couldn't find a choice class` |

需要特别留意 `act_delta` 实验：它目前只是平台 submodule 工作区里一个**未跟踪**的目录——没有提交、
没有安装进 `train-venv`、`lerobot_vlahost` 中也完全没有。任何 `act_delta` checkpoint 现阶段都无法部署。

## 5. 已确认的不兼容：`train_config.json` 无法跨端解析

```
$ PYTHONPATH=.../robot_data_platform/lerobot/src python -c "TrainPipelineConfig.from_pretrained(ckpt)"
OK: parsed train_config.json -> act

$ PYTHONPATH=.../lerobot_vlahost/src python -c "TrainPipelineConfig.from_pretrained(ckpt)"
FAIL: DecodingError `dataset`: The fields `depth_output_unit` are not valid for DatasetConfig
```

平台会写入 `dataset.depth_output_unit` 以及顶层 `job` 段，而旧版 `DatasetConfig` /
`TrainPipelineConfig` dataclass 并未声明这些字段，draccus 对普通 dataclass 的未知字段会直接报错。
（`config.json` 不受影响——那里平台写出的是严格子集。）

**影响范围有限**，因为部署链路上没有任何代码读取该文件：vlahost 的 `lerobot_rollout.py`、
`rollout/`、`lerobot_record.py` 都不会构造 `TrainPipelineConfig`。只有当你试图在 vlahost 代码树上
**恢复或重新发起训练**，或让 vlahost 的工具去读 `train_config.json` 时才会踩到。

同一区域还有一处重命名：`DatasetRecordConfig.camera_encoder` → `rgb_encoder`（并新增
`depth_encoder`）。因此 `--dataset.camera_encoder.*` 只在 vlahost 有效，
`--dataset.rgb_encoder.*` 只在平台有效。

## 6. 更严重的问题在隔壁：mgmt01 上有三份 LeRobot

审计过程中发现平台主机上存在三份不同版本的 LeRobot 代码树：

| 位置 | 版本 | 有 `vita` | 有 `act_delta` |
|---|---|---|---|
| `lerobot/` submodule 工作区 | `27fbaee7` + 未提交的 `act_delta` | 有 | 有（未跟踪） |
| `/opt/robot-platform/train-venv` | `27fbaee7`（见 `SUBMODULE_REVISION` 文件） | 有 | **无** |
| `/opt/robot-platform/lelab-venv` | fork 的**加入 vita 之前**的提交 | **无** | 无 |

已复现的后果：

```
$ /opt/robot-platform/lelab-venv/bin/python load_ckpt.py <vita checkpoint>
draccus.utils.DecodingError: Couldn't find a choice class for 'vita' in PreTrainedConfig
```

也就是说，**LeLab 无法加载自己集群训练出来的 Vita checkpoint**。ACT 在那里可以正常加载。
`apps/lelab/pyproject.toml` 至今仍声明
`lerobot[...] @ git+https://github.com/huggingface/lerobot.git@v0.6.0`——指向上游而非本仓 fork，
这正是 `lelab-venv` 落后的原因。另外注意 `train-venv` 的 `dist-info/direct_url.json` 声称装的是
上游 v0.6.0，实际内容却是 fork；只有 `SUBMODULE_REVISION` 文件记录了真实版本。

这比训练/推理的版本差更具现实风险，建议单独修复。

## 7. 建议

1. **ACT 与 Vita 的部署维持现状。** 已验证版本差对两者无害。
2. **不要在 `lerobot_vlahost` 里训练。** 它的 `ACTConfig` 是超集，产出的 `config.json`
   会被平台 fork 拒绝。
3. **部署任何新策略类型之前**，先在推理主机上跑一次低成本的可移植性检查：
   ```bash
   python -c "
   from lerobot.configs.policies import PreTrainedConfig
   from lerobot.policies.factory import get_policy_class
   cfg = PreTrainedConfig.from_pretrained('<ckpt>'); cfg.device='cpu'
   get_policy_class(cfg.type).from_pretrained('<ckpt>', config=cfg)
   print('loadable')"
   ```
   一秒内即可同时捕获两类失败（未知策略类型、未知 config 字段）。
4. **在 `act_delta` 进入部署之前**，先把它提交进 submodule，并移植到 `lerobot_vlahost`。
   目前它既无法部署也无法重装。
5. **把 `apps/lelab/pyproject.toml` 指向 fork**（至少指向 submodule 的具体版本），
   让 `lelab-venv` 不再落后于 `train-venv`。在此之前，不要指望 LeLab 的 rollout 链路能处理 Vita。
6. **考虑合并两个 fork 各自独有的改动。** 双方各持有对方缺失的修复——Vita 的
   `predict_action_chunk` 补轴修复（vlahost）与 pyav seek 修复（平台）——放任分歧重复累积，
   正是把一个无害的版本差拖成真问题的方式。

## 附录：复现方式

```bash
# 结构性 diff
diff -rq --exclude=__pycache__ \
  /home/kewei/YING/lerobot_vlahost/src/lerobot \
  /home/kewei/YING/robot_data_platform/lerobot/src/lerobot

# 分策略可移植性扫描
for p in $(ls robot_data_platform/lerobot/src/lerobot/policies/ | grep -v '\.py$\|__pycache__'); do
  [ -d "lerobot_vlahost/src/lerobot/policies/$p" ] || { echo "MISSING $p"; continue; }
  diff -rq --exclude=__pycache__ --exclude='*.md' \
    lerobot_vlahost/src/lerobot/policies/$p \
    robot_data_platform/lerobot/src/lerobot/policies/$p >/dev/null && echo "SAME $p" || echo "DIFF $p"
done

# 逐比特一致性验证：通过 PYTHONPATH 分别在两个代码树下运行同一脚本，
# 比较后处理输出 action 张量的 sha256。
```
