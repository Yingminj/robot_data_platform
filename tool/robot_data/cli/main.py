"""The ``rdp`` command: one entry point for every conversion and check."""

from __future__ import annotations

import argparse
import sys

from robot_data import __version__
from robot_data.cli import (
    check_hdf5,
    check_rosbag,
    convert_hdf5,
    convert_rosbag,
    describe,
    export_hdf5,
    upload,
)

EPILOG = """\
常用流程：
  rdp recipes                                          # 有哪些配方
  rdp check-rosbag <bag> --recipe mcap-gripper-quadtile # 先确认话题与 profile 对得上
  rdp convert --recipe mcap-gripper-quadtile \\
      --input <录制目录> --output <数据集目录>          # 再转换
  rdp convert --recipe db3-gripper --crf 28 ...        # 显式参数覆盖配方

配方提供 profile、相机话题、对齐与视频默认值；命令行显式参数始终优先。
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rdp",
        description="机器人录制数据转换工具（rosbag2 / HDF5 -> LeRobot v3）",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"robot_data {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<命令>")

    for name, module in (
        ("convert", convert_rosbag),
        ("convert-hdf5", convert_hdf5),
        ("export-hdf5", export_hdf5),
        ("check-rosbag", check_rosbag),
        ("check-hdf5", check_hdf5),
        ("upload", upload),
    ):
        sub = subparsers.add_parser(
            name,
            help=module.HELP,
            description=module.__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        module.add_arguments(sub)
        sub.set_defaults(_run=module.run)

    recipes = subparsers.add_parser(
        "recipes", help=describe.RECIPES_HELP, description=describe.RECIPES_HELP
    )
    describe.add_recipe_arguments(recipes)
    recipes.set_defaults(_run=describe.run_recipes)

    profiles = subparsers.add_parser(
        "profiles", help=describe.PROFILES_HELP, description=describe.PROFILES_HELP
    )
    describe.add_profile_arguments(profiles)
    profiles.set_defaults(_run=describe.run_profiles)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "_run", None):
        parser.print_help()
        return 2
    return int(args._run(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
