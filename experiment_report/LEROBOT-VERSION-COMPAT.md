# LeRobot fork compatibility: training (v0.6.0) vs. inference (v0.5.2)

**English** | [简体中文](LEROBOT-VERSION-COMPAT.zh-CN.md)

Audit date: 2026-08-12

## Question

Training and development run on `robot_data_platform/lerobot` (`pyproject.toml` version `0.6.0`).
Inference runs on `lerobot_vlahost` (`pyproject.toml` version `0.5.2`). Does the version gap
cause problems?

## Answer in one paragraph

**For the policies you actually train and deploy today — ACT and Vita — no.** Both repos are
forks of upstream LeRobot that sit only a few upstream commits apart; the version strings
(`0.5.2` / `0.6.0`) overstate the distance. The checkpoint-loading surface (`configs/policies.py`,
the whole `processor/` package, `datasets/` v3.0 format) is byte-identical between the two trees,
and a checkpoint trained on the platform produces **bit-identical actions** under either fork.
The real risks are not the version number: they are (1) four policies that exist only in the
training fork, (2) `train_config.json`, which genuinely does *not* cross-parse, and (3) a third
LeRobot copy inside `lelab-venv` that is older than both.

---

## 1. What was actually compared

| | training / dev | inference |
|---|---|---|
| path | `/home/kewei/YING/robot_data_platform/lerobot` | `/home/kewei/YING/lerobot_vlahost` |
| remote | `github.com/Yingminj/lerobot_dev` (+ `upstream` = huggingface/lerobot) | `github.com/tangyubbb/lerobot_vlahost` |
| branch / rev | `dev` @ `27fbaee7` (`v0.6.0-2-g27fbaee7`) | `dev-yw` @ `79cc122` |
| declared version | `0.6.0` | `0.5.2` |
| working tree | modified: untracked `policies/act_delta/` | untracked `record_chunk.txt` |

The two repos have **unrelated git histories** (vlahost carries 30 commits, no upstream tags), so
the comparison below is a filesystem diff of `src/lerobot`, not a `git diff`.

Note that `src/lerobot/__version__.py` in *both* repos resolves `__version__` from installed
package metadata, not from the source tree. `lerobot.__version__` therefore reports whatever the
active venv was installed as and is **not** a reliable way to tell the two forks apart. Use the
presence of `policies/vita` / `policies/act_delta` instead.

## 2. Divergence is bidirectional, and mostly not upstream version drift

Neither repo is a superset of the other. Both forked upstream and then added local work:

**Only in the inference fork (`lerobot_vlahost`)**
- `Marvin_sdk_pro/`, `robots/marvain_m6{,_http,_hybrid}/` — the actual robot drivers
- `rollout/inference/chunk.py` + `ChunkInferenceConfig`, `send_action_chunk`,
  `produces_chunks` on the engine base class, `go_home()` fallback in `strategies/core.py`
- `scripts/lerobot_replay_chunk.py`
- ACT `loss_time_decay` / `loss_front_weight` (time-weighted L1, training-side only)
- a fix in `policies/vita/modeling_vita.py` that adds the observation-step axis to
  `observation.state` when rollout engines call `predict_action_chunk` directly

**Only in the training fork (`robot_data_platform/lerobot`)**
- policies `act_delta`, `evo1`, `fastwam`, `lingbot_va`
- the upstream v0.6.0 **depth-camera feature set**: `datasets/depth_utils.py`,
  `DepthEncoderConfig`/`RGBEncoderConfig`, `depth_output_unit`, `is_depth_map` feature markers
- HF Jobs support (`jobs/`, `JobConfig`)
- Foxglove visualisation (`utils/foxglove_visualization.py`, `display_mode`)
- a pyav seek fix in `decode_video_frames_pyav` (stream-relative seek instead of
  `int(first_ts * av.time_base)`)

Most of the "0.5.2 → 0.6.0" delta is the depth-camera work plus HF Jobs. **None of it touches the
RGB inference path.**

## 3. The checkpoint-loading surface is identical

These files are byte-for-byte the same in both repos, which is why checkpoints port cleanly:

- `configs/policies.py` — `PreTrainedConfig.from_pretrained`
- the entire `processor/` package — every pipeline step, its registry name, and its
  serialisation format
- `policies/act/processor_act.py`
- `datasets` `CODEBASE_VERSION = "v3.0"` on both sides

A real ACT checkpoint from
`/mnt/robot_platform/jobs/act_tidy_up_stationery_le_batch_4_.../checkpoints/004000/pretrained_model`
uses only generic steps:

```
preprocessor : rename_observations → to_batch → device → normalizer
postprocessor: unnormalizer → device
```

Every one of those resolves in both forks.

### Verified: bit-identical inference

Same checkpoint, same seeded synthetic observation, CPU, `select_action` through the full
pre/post-processor pipeline, under each fork in turn:

| checkpoint | training fork (v0.6.0) | inference fork (v0.5.2) |
|---|---|---|
| ACT `act_tidy_up_stationery_le_batch_4` step 4000 | `sha256 8878961073bde336…`, sum `-3.1512776017` | **identical** |
| Vita `vita_tidy_up_stationery_le_batch_3` step 1000 | `sha256 e6eea1d23af2934f…`, sum `-7.5564867780` | **identical** |

Dataset reads match too. `/mnt/robot_platform/datasets/express`, episode 0, frame 5, read under
both forks with the default (torchcodec) and with the pyav backend: identical tensors for
`action`, `observation.state`, `observation.velocity` and all three camera streams.

### Why the ACT config difference is harmless

`ACTConfig` in the inference fork has two extra fields (`loss_time_decay`, `loss_front_weight`)
that the training fork does not. Direction matters: the platform writes a `config.json` **without**
them, and the inference fork's dataclass simply falls back to its defaults (`0.0` / `1.0`), which
disable the feature. Both fields are used only in `compute_loss`, never at inference.

The reverse direction would break — draccus rejects unknown fields (see §5) — so **a checkpoint
trained inside `lerobot_vlahost` cannot be loaded by the platform fork.** Keep training on the
platform side.

## 4. Per-policy portability (platform-trained checkpoint → vlahost inference)

| policy | code | config fields only in platform | portable? |
|---|---|---|---|
| **act** | 2 files differ (loss code + whitespace) | — (platform is a strict subset) | ✅ verified bit-identical |
| **vita** | 1 file differs (vlahost has an extra rollout fix) | — | ✅ verified bit-identical |
| diffusion, eo1, gaussian_actor, multi_task_dit, pi0, pi05, pi0_fast, rtc, smolvla, tdmpc, vqbet, wall_x, xvla | identical | — | ✅ |
| vla_jepa | 2 files differ (modeling, qwen_interface) | — | ⚠️ config loads; weights/forward not verified |
| **molmoact2** | config + processor differ | `joint_offsets`, `joint_signs` | ❌ config.json rejected |
| **groot** | 10 files differ — different implementations (`groot_n1_7.py` vs `groot_n1.py` + `eagle2_hg_model/`), 2584-line processor diff | `use_relative_actions`, `relative_exclude_joints`, `action_decode_transform`, `num_inference_timesteps`, `rtc_ramp_rate`, `model_path`, `model_params_fp32`, `use_flash_attention`, `tune_vlln`, `tune_top_llm_layers` | ❌ not portable |
| **act_delta**, **evo1**, **fastwam**, **lingbot_va** | absent from vlahost | — | ❌ `DecodingError: Couldn't find a choice class` |

Your `act_delta` experiment is the one to watch. It exists only as an *untracked* directory in the
platform submodule working tree — it is not committed, not installed in `train-venv`, and not
present in `lerobot_vlahost` at all. Any `act_delta` checkpoint is currently undeployable.

## 5. Confirmed incompatibility: `train_config.json` does not cross-parse

```
$ PYTHONPATH=.../robot_data_platform/lerobot/src python -c "TrainPipelineConfig.from_pretrained(ckpt)"
OK: parsed train_config.json -> act

$ PYTHONPATH=.../lerobot_vlahost/src python -c "TrainPipelineConfig.from_pretrained(ckpt)"
FAIL: DecodingError `dataset`: The fields `depth_output_unit` are not valid for DatasetConfig
```

The platform writes `dataset.depth_output_unit` and a top-level `job` block that the older
`DatasetConfig` / `TrainPipelineConfig` dataclasses do not declare, and draccus rejects unknown
fields on plain dataclasses. (`config.json` is unaffected — there the platform writes a strict
subset.)

**Impact is limited**, because nothing on the deployment path reads this file:
`lerobot_rollout.py`, `rollout/`, and `lerobot_record.py` in vlahost never construct a
`TrainPipelineConfig`. It only bites if you try to *resume or re-launch training* from a
platform checkpoint using the vlahost tree, or point any vlahost tooling at `train_config.json`.

Related rename in the same area: `DatasetRecordConfig.camera_encoder` → `rgb_encoder` (+ new
`depth_encoder`). A record command with `--dataset.camera_encoder.*` works only on vlahost;
`--dataset.rgb_encoder.*` works only on the platform.

## 6. The bigger problem is next door: three LeRobot copies on mgmt01

While auditing, three different LeRobot trees turned up on the platform host, at three different
revisions:

| location | revision | has `vita` | has `act_delta` |
|---|---|---|---|
| `lerobot/` submodule working tree | `27fbaee7` + uncommitted `act_delta` | yes | yes (untracked) |
| `/opt/robot-platform/train-venv` | `27fbaee7` (`SUBMODULE_REVISION` file) | yes | **no** |
| `/opt/robot-platform/lelab-venv` | fork, **pre-vita** commit | **no** | no |

Consequence, reproduced:

```
$ /opt/robot-platform/lelab-venv/bin/python load_ckpt.py <vita checkpoint>
draccus.utils.DecodingError: Couldn't find a choice class for 'vita' in PreTrainedConfig
```

So **LeLab cannot load the Vita checkpoints its own cluster produced.** ACT loads fine there.
`apps/lelab/pyproject.toml` still declares
`lerobot[...] @ git+https://github.com/huggingface/lerobot.git@v0.6.0` — upstream, not the fork —
which is why `lelab-venv` drifts. Note also that `train-venv`'s `dist-info/direct_url.json`
claims upstream v0.6.0 while its contents are in fact the fork; only the `SUBMODULE_REVISION`
file records the truth.

This is a bigger practical risk than the training/inference version gap and is worth fixing
independently.

## 7. Recommendations

1. **Keep ACT and Vita deployments as they are.** The gap is verified harmless for both.
2. **Do not train inside `lerobot_vlahost`.** Its `ACTConfig` is a superset; the resulting
   `config.json` will be rejected by the platform fork.
3. **Before deploying any new policy type**, run the cheap portability check on the inference
   host:
   ```bash
   python -c "
   from lerobot.configs.policies import PreTrainedConfig
   from lerobot.policies.factory import get_policy_class
   cfg = PreTrainedConfig.from_pretrained('<ckpt>'); cfg.device='cpu'
   get_policy_class(cfg.type).from_pretrained('<ckpt>', config=cfg)
   print('loadable')"
   ```
   This catches both failure modes (unknown policy type, unknown config field) in one second.
4. **Before `act_delta` reaches deployment**, commit it to the submodule and port it into
   `lerobot_vlahost`. Right now it is undeployable and unreinstallable.
5. **Repoint `apps/lelab/pyproject.toml` at the fork** (or at least at the submodule revision) so
   `lelab-venv` stops lagging `train-venv`. Until then, do not expect LeLab's rollout path to
   handle Vita.
6. **Consider merging the two forks' non-overlapping work.** Each holds a fix the other lacks —
   the Vita `predict_action_chunk` axis fix (vlahost) and the pyav seek fix (platform) — and
   duplicated divergence is what turns a harmless version gap into a real one over time.

## Appendix: how to reproduce

```bash
# Structural diff
diff -rq --exclude=__pycache__ \
  /home/kewei/YING/lerobot_vlahost/src/lerobot \
  /home/kewei/YING/robot_data_platform/lerobot/src/lerobot

# Per-policy portability sweep
for p in $(ls robot_data_platform/lerobot/src/lerobot/policies/ | grep -v '\.py$\|__pycache__'); do
  [ -d "lerobot_vlahost/src/lerobot/policies/$p" ] || { echo "MISSING $p"; continue; }
  diff -rq --exclude=__pycache__ --exclude='*.md' \
    lerobot_vlahost/src/lerobot/policies/$p \
    robot_data_platform/lerobot/src/lerobot/policies/$p >/dev/null && echo "SAME $p" || echo "DIFF $p"
done

# Bit-identical inference check: run the same script under each tree via PYTHONPATH,
# compare the sha256 of the post-processed action tensor.
```
