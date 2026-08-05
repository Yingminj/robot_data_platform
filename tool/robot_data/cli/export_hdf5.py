"""``rdp export-hdf5`` -- rosbag2 episodes to timestamp-audited ACT HDF5."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from robot_data.align import align_rosbag
from robot_data.cli.args import (
    add_alignment_args,
    add_batch_args,
    add_hdf5_output_args,
    add_profile_args,
    add_recipe_args,
    describe_selection,
    resolve_alignment,
    resolve_include_depth,
    resolve_profile,
    resolve_recipe,
)
from robot_data.discovery import discover_rosbags, sort_and_limit
from robot_data.errors import ConversionError
from robot_data.progress import set_progress_mode
from robot_data.writers.hdf5 import write_aligned_hdf5

HELP = "严格按控制频率对齐 rosbag2，写出带源时间戳的 ACT HDF5"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True, help="单个 rosbag 目录或其父目录")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-name-template",
        default="{name}.hdf5",
        help="支持 {name} 和从 0 开始的 {index}，例如 episode_{index:06d}.hdf5",
    )
    add_recipe_args(parser)
    add_profile_args(parser)
    add_alignment_args(parser)
    add_hdf5_output_args(parser)
    add_batch_args(parser)


def run(args: argparse.Namespace) -> int:
    set_progress_mode(args.progress)
    try:
        recipe = resolve_recipe(args)
        profile = resolve_profile(args, recipe)
        args.include_depth = resolve_include_depth(args, recipe)
        cfg = resolve_alignment(args, recipe)
        print(describe_selection(profile, recipe, cfg), flush=True)

        bags = sort_and_limit(
            discover_rosbags(args.input, args.recursive), args.sort_by, args.limit
        )
        if not bags:
            raise ConversionError(f"No rosbag2 episodes found under {args.input}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, object]] = []
        started = time.monotonic()
        for index, bag in enumerate(bags, 1):
            filename = args.output_name_template.format(name=bag.name, index=index - 1)
            if Path(filename).name != filename or not filename.endswith((".hdf5", ".h5")):
                raise ValueError(f"Invalid output filename from template: {filename!r}")
            output = args.output_dir / filename
            staging = output.with_name(f".{output.name}.incomplete")
            print(f"[{index}/{len(bags)}] {bag} -> {output}", flush=True)
            try:
                if output.exists():
                    if not args.overwrite:
                        raise FileExistsError(f"Output exists: {output}")
                    output.unlink()
                if staging.exists():
                    staging.unlink()
                episode = align_rosbag(bag, profile, cfg)
                write_aligned_hdf5(
                    staging,
                    episode,
                    profile,
                    compression=args.compression,
                    compression_level=args.compression_level,
                )
                staging.rename(output)
                results.append(
                    {
                        "source": str(bag),
                        "output": str(output),
                        "status": "converted",
                        **episode.audit,
                    }
                )
                print(f"  converted: {episode.frame_count} frames", flush=True)
            except Exception as exc:
                if staging.exists():
                    staging.unlink()
                results.append({"source": str(bag), "status": "failed", "error": str(exc)})
                print(f"  failed: {exc}", file=sys.stderr, flush=True)
                if args.on_error == "fail":
                    raise
        summary = {
            "converter": "rdp export-hdf5",
            "recipe": recipe.to_dict() if recipe else None,
            "elapsed_s": time.monotonic() - started,
            "episodes": results,
        }
        (args.output_dir / "conversion_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return 0 if any(item["status"] == "converted" for item in results) else 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
