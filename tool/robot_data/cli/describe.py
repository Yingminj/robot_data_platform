"""``rdp recipes`` / ``rdp profiles`` -- what is available and what it implies.

Printed rather than only documented, so the answer to "which recipe matches this
batch?" comes from the files that will actually be used.
"""

from __future__ import annotations

import argparse
import json

from robot_data.profiles.schema import builtin_profile_names, load_profile
from robot_data.recipes import describe_recipes, load_recipe

RECIPES_HELP = "列出内置转换配方及其 profile、相机话题与关键对齐参数"
PROFILES_HELP = "列出内置 robot profile 及其话题与状态维度"


def add_recipe_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("name", nargs="?", help="只显示这一个配方的完整内容")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")


def add_profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("name", nargs="?", help="只显示这一个 profile 的完整内容")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")


def run_recipes(args: argparse.Namespace) -> int:
    if args.name:
        recipe = load_recipe(args.name)
        if args.json:
            print(json.dumps(recipe.to_dict(), ensure_ascii=False, indent=2))
            return 0
        print(f"配方 {recipe.name}（{recipe.storage}）")
        print(f"  {recipe.description}")
        print(f"  profile: {recipe.profile.name}  state_dim={recipe.profile.state_dim}")
        print(f"  手臂: state={recipe.profile.arm.joint_states_topic}")
        for topic in recipe.profile.arm.command_topics:
            print(f"        cmd={topic}")
        for effector in recipe.profile.end_effectors:
            state = effector.state_topic or "无（观测为命令回显）"
            print(
                f"  末端 {effector.name}（{effector.kind}，{effector.dim} 维）: "
                f"cmd={effector.command_topic}  state={state}"
            )
        for camera in sorted(recipe.profile.cameras):
            mark = " [锚点]" if camera == recipe.profile.resolved_anchor_camera else ""
            print(f"  相机 {camera}{mark}: {recipe.profile.cameras[camera]}")
        print("  对齐:")
        for key, value in recipe.alignment.items():
            print(f"    {key} = {value}")
        print("  视频（最常调整）:")
        for key, value in recipe.video.items():
            print(f"    {key} = {value}")
        for note in recipe.notes:
            print(f"  注意：{note}")
        return 0

    rows = describe_recipes()
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    print(f"{'配方':<28}{'存储':<9}{'profile':<26}{'维度':>5}  说明")
    for row in rows:
        print(
            f"{row['name']:<28}{row['storage']:<9}{row['profile']:<26}"
            f"{row['state_dim']:>5}  {row['description']}"
        )
    print("\n相机话题：")
    for row in rows:
        topics = sorted(set(row["cameras"].values()))
        joined = ", ".join(topics)
        print(f"  {row['name']:<28}{joined}")
    print("\n关键对齐差异（其余为默认值）：")
    for row in rows:
        interesting = {
            key: value
            for key, value in row["alignment"].items()
            if key
            in {
                "action_gap_policy",
                "missing_topic_policy",
                "invalid_frame_policy",
                "max_tick_rate_deviation",
                "image_tolerance_ms",
                "state_tolerance_periods",
                "max_hold_fraction",
            }
        }
        print(f"  {row['name']:<28}{interesting}")
    print("\n视频默认值（--crf / --video-codec 等可覆盖）：")
    for row in rows:
        print(f"  {row['name']:<28}{row['video']}")
    print("\n用法：rdp convert --recipe <配方> --input <目录> --output <数据集目录>")
    print("      rdp recipes <配方>   查看单个配方的完整内容")
    return 0


def run_profiles(args: argparse.Namespace) -> int:
    names = [args.name] if args.name else builtin_profile_names()
    if args.json:
        print(
            json.dumps(
                [load_profile(name).to_dict() for name in names], ensure_ascii=False, indent=2
            )
        )
        return 0
    for name in names:
        profile = load_profile(name)
        print(f"{profile.name}  robot_type={profile.robot_type}  state_dim={profile.state_dim}")
        print(f"  手臂 state: {profile.arm.joint_states_topic}")
        print(f"  手臂 cmd  : {', '.join(profile.arm.command_topics)}")
        for effector in profile.end_effectors:
            state = effector.state_topic or "无（观测为命令回显）"
            print(
                f"  末端 {effector.name}（{effector.kind}，{effector.dim} 维）"
                f" cmd={effector.command_topic} state={state}"
            )
        for camera in sorted(profile.cameras):
            mark = " [锚点]" if camera == profile.resolved_anchor_camera else ""
            tile = profile.camera_tiles.get(camera)
            crop = ""
            if tile is not None:
                crop = (
                    f"  裁切 x[{tile.left:.3f},{tile.right:.3f}]"
                    f" y[{tile.top:.3f},{tile.bottom:.3f}] -> {tile.width}x{tile.height}"
                )
            print(f"  相机 {camera}{mark}: {profile.cameras[camera]}{crop}")
        for depth in sorted(profile.depths):
            print(f"  深度 {depth}: {profile.depths[depth]}")
        print()
    return 0
