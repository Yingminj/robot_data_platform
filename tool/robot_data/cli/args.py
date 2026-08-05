"""Argument groups shared by the entry points, and recipe resolution.

Every recipe-controlled flag defaults to ``None`` rather than to its real
default.  That is what makes "the user typed ``--crf 28``" distinguishable from
"the user said nothing about crf", which in turn is what lets a recipe supply a
value without an argparse default silently overriding it.  The real defaults
live once, in the dataclasses these functions build.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from robot_data.align.config import (
    ACTION_GAP_POLICIES,
    ALIGNMENT_MODES,
    GRID_ANCHORS,
    INVALID_FRAME_POLICIES,
    MISSING_TOPIC_POLICIES,
    AlignmentConfig,
)
from robot_data.profiles.schema import (
    RobotProfile,
    apply_topic_overrides,
    builtin_profile_names,
    load_profile,
    parse_name_topic,
)
from robot_data.progress import PROGRESS_MODES
from robot_data.recipes import Recipe, load_recipe, recipe_names
from robot_data.writers.hdf5 import HDF5_COMPRESSIONS
from robot_data.writers.lerobot_v3 import RGBVideoConfig, parse_preset


def add_recipe_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("recipe")
    group.add_argument(
        "--recipe",
        default=None,
        help=f"转换配方，内置 {recipe_names()} 或 JSON 文件路径；"
        "提供 profile、相机话题、对齐与视频默认值。命令行显式参数优先于配方。",
    )


def add_profile_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("profile")
    group.add_argument(
        "--profile",
        default=None,
        help=f"机器人配置：内置 {builtin_profile_names()} 或 JSON 文件路径。"
        "未指定时取配方的 profile；两者都没有则取 tj-dexhand。",
    )
    group.add_argument(
        "--camera",
        action="append",
        metavar="NAME=TOPIC",
        help="可重复；一旦指定即替换 profile/配方中的相机映射",
    )
    group.add_argument("--depth", action="append", metavar="NAME=TOPIC")
    group.add_argument("--anchor-camera", default=None, help="用作网格锚点的相机名")
    group.add_argument(
        "--include-depth", action=argparse.BooleanOptionalAction, default=None
    )


def add_alignment_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("alignment")
    group.add_argument("--fps", type=int, default=None, help="默认 30，或配方指定值")
    group.add_argument(
        "--alignment-mode",
        dest="mode",
        choices=ALIGNMENT_MODES,
        default=None,
        help="lerobot-loop（默认）=按 bag 到达时间取 tick 之前最新帧，因果；"
        "capture=按 header.stamp 物理采集时间，会用到 tick 之后的数据，仅用于诊断",
    )
    group.add_argument("--image-tolerance-ms", type=float, default=None)
    group.add_argument(
        "--state-tolerance-ms",
        type=float,
        default=None,
        help="绝对上限；未指定时按实测 joint_states 周期的倍数",
    )
    group.add_argument(
        "--state-tolerance-periods",
        type=float,
        default=None,
        help="未指定 --state-tolerance-ms 时使用的周期倍数，默认 1.5（db3 配方为 4.5）",
    )
    group.add_argument(
        "--action-tolerance-ms", type=float, default=None, help="默认一帧周期；30 FPS 时 33.33 ms"
    )
    group.add_argument("--action-pair-tolerance-ms", type=float, default=None)
    group.add_argument(
        "--end-effector-tolerance-ms",
        type=float,
        default=None,
        help="夹爪/灵巧手状态与命令的最大时间偏差",
    )
    group.add_argument("--invalid-frame-policy", choices=INVALID_FRAME_POLICIES, default=None)
    group.add_argument(
        "--action-gap-policy",
        choices=ACTION_GAP_POLICIES,
        default=None,
        help="hold-last-command=断档时保持最后一条 joint_cmd；"
        "joint-state-fill=断档时按该手臂实测 joint_states 填充（旧版 .db3 遥操作会整段静默）；"
        "fail=拒绝该 episode",
    )
    group.add_argument(
        "--missing-topic-policy",
        choices=MISSING_TOPIC_POLICIES,
        default=None,
        help="profile 声明但整段录制里没有的话题如何处理："
        "fail=拒绝该 episode；"
        "fill=用实测状态重建——手臂 joint_cmd 用该臂 joint_states，末端执行器指令用其实测反馈"
        "（需配合 --action-gap-policy joint-state-fill；重建列与 observation 完全相同）",
    )
    group.add_argument(
        "--grid-anchor",
        choices=GRID_ANCHORS,
        default=None,
        help="anchor-camera-ticks（默认）=直接以锚点相机帧时刻为 tick，图像陈旧度恒为 0；"
        "anchor-camera=从首条 joint_cmd 之前最近的锚点相机帧起按 1/fps 取 tick；"
        "first-command=从首条 joint_cmd 起按 1/fps 取 tick",
    )
    group.add_argument(
        "--max-hold-fraction", type=float, default=None, help="保持动作的行数占比上限"
    )
    group.add_argument(
        "--max-hold-run-s", type=float, default=None, help="单段保持动作的最长时长（秒）"
    )
    group.add_argument(
        "--max-tick-rate-deviation",
        type=float,
        default=None,
        help="anchor-camera-ticks 下实测 tick 频率与 --fps 的最大相对偏差，默认 0.1",
    )
    group.add_argument("--max-decode-errors", type=int, default=None)
    group.add_argument("--image-height", type=int, default=None, help="0 表示保留原高度")
    group.add_argument("--image-width", type=int, default=None, help="0 表示保留原宽度")


def add_video_args(parser: argparse.ArgumentParser) -> None:
    """The knobs tuned most often; every one is also settable from a recipe."""
    group = parser.add_argument_group(
        "video encoding", "最常调整的参数；配方可提供默认值，此处显式指定则优先"
    )
    group.add_argument(
        "--video-codec",
        dest="codec",
        default=None,
        help="默认 h264；CRF 取值范围随编码器不同（AV1 0-63，x264 0-51）",
    )
    group.add_argument(
        "--crf", type=float, default=None, help="默认 20，0 表示无损；见 test_lerobot/REPORT.md"
    )
    group.add_argument("--gop", type=int, default=None, help="默认 2")
    group.add_argument("--video-pixel-format", dest="pixel_format", default=None)
    group.add_argument("--preset", default=None)
    group.add_argument("--fast-decode", type=int, default=None)
    group.add_argument("--encoder-threads", type=int, default=None)
    group.add_argument("--depth-crf", type=float, default=0)


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("dataset output")
    group.add_argument("--repo-id", default=None, help="默认 local/<output目录名>")
    group.add_argument("--task", default="Unspecified task")
    group.add_argument("--task-map", type=Path, default=None, help="JSON: 名称或路径 -> task")
    group.add_argument("--robot-type", default=None, help="覆盖 profile 中的 robot_type")
    group.add_argument("--image-storage", choices=("video", "image"), default="video")
    group.add_argument(
        "--include-velocity", action=argparse.BooleanOptionalAction, default=True
    )
    group.add_argument("--image-writer-processes", type=int, default=0)
    group.add_argument("--image-writer-threads", type=int, default=8)


def add_hdf5_output_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("hdf5 output")
    group.add_argument("--compression", choices=HDF5_COMPRESSIONS, default="gzip")
    group.add_argument("--compression-level", type=int, default=4)


def add_batch_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("batch")
    group.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--sort-by", choices=("name", "mtime"), default="name")
    group.add_argument("--limit", type=int, default=0)
    group.add_argument("--on-error", choices=("fail", "skip"), default="fail")
    group.add_argument("--overwrite", action="store_true")
    group.add_argument(
        "--progress",
        choices=PROGRESS_MODES,
        default="auto",
        help="auto=stderr 是终端时用进度条，否则每 10 秒一行",
    )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_recipe(args: argparse.Namespace) -> Recipe | None:
    return load_recipe(args.recipe) if getattr(args, "recipe", None) else None


def _layer(
    field: str, args: argparse.Namespace, recipe_block: dict[str, Any]
) -> dict[str, Any]:
    """One field's value, taking the CLI over the recipe over the default."""
    value = getattr(args, field, None)
    if value is not None:
        return {field: value}
    if field in recipe_block:
        return {field: recipe_block[field]}
    return {}


def resolve_profile(args: argparse.Namespace, recipe: Recipe | None) -> RobotProfile:
    """Pick the profile, then apply any camera overrides given on the CLI.

    A ``--profile`` on the command line replaces the recipe's profile entirely,
    including the camera overrides the recipe carried: those describe the
    recipe's own profile and would be nonsense against a different one.
    """
    if args.profile is not None:
        profile = load_profile(args.profile)
    elif recipe is not None:
        profile = recipe.profile
    else:
        from robot_data.profiles.schema import DEFAULT_PROFILE

        profile = load_profile(DEFAULT_PROFILE)
    return apply_topic_overrides(
        profile,
        cameras=parse_name_topic(args.camera),
        depths=parse_name_topic(args.depth),
        anchor_camera=args.anchor_camera,
    )


def resolve_alignment(args: argparse.Namespace, recipe: Recipe | None) -> AlignmentConfig:
    block = dict(recipe.alignment) if recipe else {}
    fields = [
        "fps",
        "mode",
        "image_tolerance_ms",
        "state_tolerance_ms",
        "state_tolerance_periods",
        "action_tolerance_ms",
        "action_pair_tolerance_ms",
        "end_effector_tolerance_ms",
        "image_height",
        "image_width",
        "invalid_frame_policy",
        "include_depth",
        "max_decode_errors",
        "action_gap_policy",
        "missing_topic_policy",
        "grid_anchor",
        "max_hold_fraction",
        "max_hold_run_s",
        "max_tick_rate_deviation",
    ]
    values: dict[str, Any] = {}
    for field in fields:
        values.update(_layer(field, args, block))
    return AlignmentConfig(**values)


def resolve_video(args: argparse.Namespace, recipe: Recipe | None) -> RGBVideoConfig:
    block = dict(recipe.video) if recipe else {}
    values: dict[str, Any] = {}
    for field in ("codec", "pixel_format", "crf", "gop", "fast_decode", "encoder_threads"):
        values.update(_layer(field, args, block))
    preset = getattr(args, "preset", None)
    if preset is not None:
        values["preset"] = parse_preset(preset)
    elif "preset" in block:
        values["preset"] = parse_preset(block["preset"])
    return RGBVideoConfig(**values)


def resolve_include_depth(args: argparse.Namespace, recipe: Recipe | None) -> bool:
    if getattr(args, "include_depth", None) is not None:
        return bool(args.include_depth)
    if recipe is not None and "include_depth" in recipe.alignment:
        return bool(recipe.alignment["include_depth"])
    return False


def describe_selection(
    profile: RobotProfile, recipe: Recipe | None, cfg: AlignmentConfig
) -> str:
    """The one-line banner every converter prints before it starts."""
    parts = [f"profile={profile.name}"]
    if recipe is not None:
        parts.insert(0, f"recipe={recipe.name}")
    parts += [
        f"state_dim={profile.state_dim}",
        f"cameras={sorted(profile.cameras)}",
        f"anchor={profile.resolved_anchor_camera}",
        f"fps={cfg.fps}",
        f"gap={cfg.action_gap_policy}",
        f"missing={cfg.missing_topic_policy}",
    ]
    return " ".join(parts)
