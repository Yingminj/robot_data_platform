"""``rdp check-rosbag`` -- inspect a .db3 or .mcap recording before converting.

The report leads with the topic inventory, because the mismatch it catches --
the profile pointing at a camera topic the recording does not publish -- is the
one that wastes the most time when it is discovered halfway through a batch
conversion instead of here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_data.cli.args import (
    add_profile_args,
    add_recipe_args,
    resolve_include_depth,
    resolve_profile,
    resolve_recipe,
)
from robot_data.discovery import discover_rosbags, sort_and_limit
from robot_data.errors import ProfileError, RecipeError
from robot_data.qc.inventory import camera_geometry, print_camera_geometry, print_inventory
from robot_data.qc.rosbag import QualityThresholds, check_bag, print_report

HELP = "按 profile/配方检查 rosbag（sqlite3 与 MCAP）的话题、时间戳、频率与丢帧"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("bag", type=Path, help="rosbag2 目录，或包含多个 bag 的父目录")
    add_recipe_args(parser)
    add_profile_args(parser)
    parser.add_argument(
        "--action-gap-policy",
        choices=("fail", "hold-last-command", "joint-state-fill"),
        default=None,
        help="与转换保持一致，用于判断命令断档是否可转换；默认取配方设置",
    )
    parser.add_argument(
        "--recursive", action=argparse.BooleanOptionalAction, default=True,
        help="父目录模式下递归查找 bag"
    )
    parser.add_argument("--limit", type=int, default=0, help="最多检查前 N 个 bag，0 表示不限")
    parser.add_argument("--sort-by", choices=("name", "mtime"), default="name")
    parser.add_argument("--json", type=Path, dest="json_path", help="同时写出机器可读 JSON 报告")
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="只列出话题清单与 profile 匹配结论，跳过频率/丢帧统计（最快）",
    )
    parser.add_argument(
        "--camera-geometry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="每个相机话题解码一帧，报告实际分辨率与拼接裁切（默认开启）",
    )
    parser.add_argument(
        "--gap-factor", type=float, default=1.5, help="间隔超过实测周期的此倍数视为掉帧"
    )
    parser.add_argument("--warn-drop-ratio", type=float, default=0.01)
    parser.add_argument("--fail-drop-ratio", type=float, default=0.05)
    parser.add_argument(
        "--sync-warn-ms", type=float, default=20.0, help="相机时间戳最近邻偏差 P95 的警告阈值"
    )
    parser.add_argument("--warn-latency-ms", type=float, default=100.0)
    parser.add_argument("--fail-latency-ms", type=float, default=500.0)
    parser.add_argument("--command-gap-warn-s", type=float, default=1.0)
    parser.add_argument(
        "--no-quick-check", action="store_true", help="跳过 sqlite3 的 PRAGMA quick_check"
    )


def run(args: argparse.Namespace) -> int:
    try:
        recipe = resolve_recipe(args)
        profile = resolve_profile(args, recipe)
        include_depth = resolve_include_depth(args, recipe)
        limits = QualityThresholds(
            gap_factor=args.gap_factor,
            warn_drop_ratio=args.warn_drop_ratio,
            fail_drop_ratio=args.fail_drop_ratio,
            sync_warn_ms=args.sync_warn_ms,
            warn_latency_ms=args.warn_latency_ms,
            fail_latency_ms=args.fail_latency_ms,
            command_gap_warn_s=args.command_gap_warn_s,
        )
    except (ProfileError, RecipeError, ValueError) as exc:
        print(f"检查失败：{exc}", file=sys.stderr)
        return 2

    gap_policy = args.action_gap_policy or (
        (recipe.alignment.get("action_gap_policy") if recipe else None) or "hold-last-command"
    )

    try:
        bags = sort_and_limit(
            discover_rosbags(args.bag, args.recursive), args.sort_by, args.limit
        )
    except Exception as exc:
        print(f"检查失败：{exc}", file=sys.stderr)
        return 2
    if not bags:
        print(f"未在 {args.bag} 下找到任何 rosbag", file=sys.stderr)
        return 2

    reports = []
    worst = "PASS"
    for index, bag_dir in enumerate(bags, 1):
        if len(bags) > 1:
            print(f"\n════ [{index}/{len(bags)}] {bag_dir}")
        try:
            report, inventory = check_bag(
                bag_dir,
                profile,
                include_depth=include_depth,
                limits=limits,
                action_gap_policy=gap_policy,
                run_quick_check=not args.no_quick_check,
                recipe_name=recipe.name if recipe else None,
            )
        except Exception as exc:
            print(f"检查失败：{exc}", file=sys.stderr)
            return 2

        print_inventory(inventory, recipe.name if recipe else None)
        if args.camera_geometry:
            try:
                from robot_data.align.bag_io import scan_timestamps

                geometry = camera_geometry(
                    scan_timestamps(bag_dir, profile, header_topics=set(), only_topics=None),
                    profile,
                )
                print_camera_geometry(geometry, profile)
                report["camera_geometry"] = geometry
            except Exception as exc:
                print(f"\n相机分辨率检查跳过：{exc}")
        if not args.inventory_only:
            print()
            print_report(report)
        elif not inventory.matches:
            report["overall_status"] = "FAIL"

        reports.append(report)
        status = report["overall_status"] if not args.inventory_only else (
            "PASS" if inventory.matches else "FAIL"
        )
        worst = "FAIL" if "FAIL" in {worst, status} else "WARN" if "WARN" in {worst, status} else "PASS"

    if len(bags) > 1:
        print(f"\n════ 共检查 {len(bags)} 个 bag，总体：{worst}")

    if args.json_path:
        json_path = args.json_path.expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = reports[0] if len(reports) == 1 else {"overall_status": worst, "bags": reports}
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"JSON 报告：{json_path}")

    return 0 if worst == "PASS" else 1 if worst == "WARN" else 2
