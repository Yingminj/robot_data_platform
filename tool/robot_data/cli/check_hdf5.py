"""``rdp check-hdf5`` -- validate aligned HDF5 episodes before LeRobot conversion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robot_data.qc.hdf5 import (
    HDF5CheckOptions,
    check_file,
    collect_files,
    print_file_report,
    write_json,
)

HELP = "检查 ACT HDF5 的数据集布局、来源话题与数据质量"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="HDF5 文件或包含 HDF5 的目录")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pattern", default="*.hdf5", help="目录模式下的文件通配符")
    parser.add_argument("--limit", type=int, default=0, help="最多检查前 N 个文件，0 表示不限制")
    parser.add_argument(
        "--include-velocity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="要求存在 observations/qvel（与转换脚本默认一致）",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="用于把关节跳变换算成每秒角速度；文件带 fps 属性时以文件为准",
    )
    parser.add_argument(
        "--sample-frames", type=int, default=8, help="每个相机抽样检查的帧数，0 表示跳过图像检查"
    )
    parser.add_argument("--identical-warn-ratio", type=float, default=0.9)
    parser.add_argument("--max-step-rad", type=float, default=0.35)
    parser.add_argument("--static-warn-ratio", type=float, default=0.98)
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--json", type=Path, dest="json_path", default=None)
    parser.add_argument("--quiet", action="store_true", help="只打印每个文件一行摘要")
    parser.add_argument(
        "--layout",
        action="store_true",
        help="打印文件内所有数据集的形状、dtype、chunk、压缩与占用空间",
    )
    parser.add_argument(
        "--layout-only", action="store_true", help="只打印布局，跳过全部质量检查（最快）"
    )


def run(args: argparse.Namespace) -> int:
    try:
        import h5py  # noqa: F401
    except ImportError:
        print("缺少依赖：pip install h5py", file=sys.stderr)
        return 2

    options = HDF5CheckOptions(
        input=args.input,
        recursive=args.recursive,
        pattern=args.pattern,
        limit=args.limit,
        include_velocity=args.include_velocity,
        fps=args.fps,
        sample_frames=args.sample_frames,
        identical_warn_ratio=args.identical_warn_ratio,
        max_step_rad=args.max_step_rad,
        static_warn_ratio=args.static_warn_ratio,
        min_frames=args.min_frames,
        max_lag=args.max_lag,
        json_path=args.json_path,
        quiet=args.quiet,
        layout=args.layout,
        layout_only=args.layout_only,
    )

    try:
        files = collect_files(
            options.input.expanduser().resolve(),
            options.pattern,
            options.recursive,
            options.limit,
        )
    except Exception as exc:
        print(f"检查失败：{exc}", file=sys.stderr)
        return 2
    if not files:
        print(f"未找到匹配 {options.pattern} 的 HDF5 文件", file=sys.stderr)
        return 2

    reports = []
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] ", end="")
        report = check_file(path, options)
        reports.append(report)
        print_file_report(report, options.quiet)

    counts = {
        status: sum(1 for report in reports if report["status"] == status)
        for status in ("PASS", "WARN", "FAIL")
    }
    overall = "FAIL" if counts["FAIL"] else "WARN" if counts["WARN"] else "PASS"

    if options.layout_only:
        print(f"\n共列出 {len(reports)} 个文件的布局（未执行质量检查）。")
        write_json(options, reports, "INFO", counts)
        return 0

    print()
    print(
        f"总计 {len(reports)} 个文件：PASS {counts['PASS']} | "
        f"WARN {counts['WARN']} | FAIL {counts['FAIL']}"
    )
    if counts["FAIL"]:
        print("FAIL 文件：")
        for report in reports:
            if report["status"] == "FAIL":
                print(f"  {Path(report['file']).name}: {'; '.join(report['issues'])}")
    if overall == "PASS":
        print("结论：全部文件满足 'rdp convert-hdf5' 的输入要求。")
    elif overall == "WARN":
        print("结论：可以转换，但存在数据质量警告，请逐条确认 WARN 行。")
    else:
        print("结论：存在会被转换脚本拒绝的文件，需要在生成 HDF5 的上游修复。")

    write_json(options, reports, overall, counts)
    return 0 if overall == "PASS" else 1 if overall == "WARN" else 2
