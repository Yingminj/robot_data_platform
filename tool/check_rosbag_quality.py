#!/usr/bin/env python3
"""Offline quality checks for ROS 2 bags, driven by a robot profile.

The topic list, expected end effectors and camera set come from the same
``tool/robot_profile.py`` description the converters use, so this checker and
``rosbag2_to_lerobotv3.py`` always agree about what a recording should contain.
Both rosbag2 storage backends (sqlite3 ``.db3`` and MCAP ``.mcap``) are read
through the shared reader, which also supplies message definitions that a
``.db3`` does not carry.

Only timestamps are read: image payloads are never decoded, so a multi-GiB bag
is checked in seconds.  Nominal publishing rates are measured from the data
(median inter-message period) rather than hardcoded, so the same thresholds
apply to a 30 Hz camera and a 500 Hz joint-state stream.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from conversion_common import open_bag_reader
from robot_profile import (
    BUILTIN_PROFILES,
    DEFAULT_PROFILE,
    FLOAT32,
    ProfileError,
    RobotProfile,
    apply_topic_overrides,
    load_profile,
    parse_name_topic,
)
from ros_messages import header_stamp_ns_from_cdr


@dataclass
class TopicSpec:
    """What a topic is used for, derived from the profile."""

    role: str          # camera / depth / arm_state / arm_command / ee_state / ee_command
    has_header: bool
    required: bool = True
    # Sensor streams publish at a steady rate, so a gap really is a dropped
    # message.  Command streams only publish while teleoperation is enabled, so
    # their gaps are operator behaviour and are reported by teleop_report
    # instead of being scored as drops.
    continuous: bool = True


def build_specs(profile: RobotProfile, include_depth: bool) -> Dict[str, TopicSpec]:
    """Map every topic the profile needs to its role and header expectation."""
    specs: Dict[str, TopicSpec] = {}
    for topic in profile.cameras.values():
        specs[topic] = TopicSpec("camera", has_header=True)
    if include_depth:
        for topic in profile.depths.values():
            specs[topic] = TopicSpec("depth", has_header=True, required=False)
    specs[profile.arm.joint_states_topic] = TopicSpec("arm_state", has_header=True)
    for topic in profile.arm.command_topics:
        specs[topic] = TopicSpec("arm_command", has_header=True, continuous=False)
    for effector in profile.end_effectors:
        # std_msgs/Float32 grippers have no Header; record time is their only clock.
        header = effector.command_kind != FLOAT32
        specs[effector.command_topic] = TopicSpec("ee_command", has_header=header, continuous=False)
        if effector.state_topic:
            header = effector.state_kind != FLOAT32
            specs[effector.state_topic] = TopicSpec("ee_state", has_header=header)
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 robot profile 检查 ROS 2 bag（sqlite3/MCAP）的话题完整性、时间戳、频率与丢帧。"
    )
    parser.add_argument("bag", type=Path, help="包含 metadata.yaml 的 rosbag2 目录")
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help=f"机器人配置：内置 {sorted(BUILTIN_PROFILES)} 或 JSON 文件路径",
    )
    parser.add_argument("--camera", action="append", metavar="NAME=TOPIC",
                        help="可重复；一旦指定即替换 profile 中的相机映射")
    parser.add_argument("--depth", action="append", metavar="NAME=TOPIC")
    parser.add_argument("--anchor-camera", default=None)
    parser.add_argument("--include-depth", action=argparse.BooleanOptionalAction, default=False,
                        help="同时检查深度话题（默认关闭，与转换脚本一致）")
    parser.add_argument("--json", type=Path, dest="json_path", help="同时写出机器可读 JSON 报告")
    parser.add_argument("--gap-factor", type=float, default=1.5,
                        help="间隔超过实测周期的此倍数视为掉帧，默认 1.5")
    parser.add_argument("--warn-drop-ratio", type=float, default=0.01,
                        help="估算丢帧率达到此值时警告，默认 0.01")
    parser.add_argument("--fail-drop-ratio", type=float, default=0.05,
                        help="估算丢帧率达到此值时失败，默认 0.05")
    parser.add_argument("--sync-warn-ms", type=float, default=20.0,
                        help="相机时间戳最近邻偏差 P95 的警告阈值，默认 20 ms")
    parser.add_argument("--warn-latency-ms", type=float, default=100.0,
                        help="Header 到 rosbag 接收时间的 P95 延迟警告阈值，默认 100 ms")
    parser.add_argument("--fail-latency-ms", type=float, default=500.0,
                        help="Header 到 rosbag 接收时间的 P95 延迟失败阈值，默认 500 ms")
    parser.add_argument("--command-gap-warn-s", type=float, default=1.0,
                        help="遥操作命令流断档超过该秒数时报告，默认 1 s")
    parser.add_argument("--action-gap-policy",
                        choices=("fail", "hold-last-command", "joint-state-fill"),
                        default="hold-last-command",
                        help="与转换脚本保持一致，用于判断命令断档是否可转换；默认 hold-last-command")
    parser.add_argument("--no-quick-check", action="store_true",
                        help="跳过 sqlite3 的 PRAGMA quick_check（MCAP 无此步骤）")
    args = parser.parse_args()
    if args.gap_factor <= 1.0:
        parser.error("--gap-factor 必须大于 1")
    if not 0 <= args.warn_drop_ratio <= args.fail_drop_ratio <= 1:
        parser.error("丢帧率阈值必须满足 0 <= warn <= fail <= 1")
    if not 0 <= args.warn_latency_ms <= args.fail_latency_ms:
        parser.error("延迟阈值必须满足 0 <= warn <= fail")
    return args


def percentile(values: np.ndarray, q: float) -> Optional[float]:
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_bag(
    bag_dir: Path, profile: RobotProfile, specs: Dict[str, TopicSpec]
) -> Tuple[Dict[str, str], Dict[str, List[Tuple[int, Optional[int]]]], int, int]:
    """Collect (receive_ns, header_ns) per topic without decoding payloads."""
    samples: Dict[str, List[Tuple[int, Optional[int]]]] = {topic: [] for topic in specs}
    types: Dict[str, str] = {}
    reader = open_bag_reader(bag_dir, profile)
    reader.open()
    try:
        connections = [c for c in reader.connections if c.topic in specs]
        for connection in connections:
            types[connection.topic] = connection.msgtype
        for connection, record_ns, rawdata in reader.messages(connections=connections):
            header_ns = None
            if specs[connection.topic].has_header:
                try:
                    header_ns = header_stamp_ns_from_cdr(rawdata)
                except Exception:
                    header_ns = None
            samples[connection.topic].append((int(record_ns), header_ns))
        start = getattr(reader, "start_time", 0) or 0
        end = getattr(reader, "end_time", 0) or 0
    finally:
        reader.close()
    for values in samples.values():
        values.sort(key=lambda item: item[0])
    if not start or not end:
        flat = [item[0] for values in samples.values() for item in values]
        start, end = (min(flat), max(flat)) if flat else (0, 0)
    return types, samples, int(start), int(end)


def sqlite_integrity(bag_dir: Path, run_quick_check: bool) -> Dict[str, str]:
    """PRAGMA quick_check on every .db3; MCAP bags return an empty mapping."""
    result: Dict[str, str] = {}
    for db_path in sorted(bag_dir.glob("*.db3")):
        if not run_quick_check:
            result[db_path.name] = "skipped"
            continue
        try:
            connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                rows = connection.execute("PRAGMA quick_check").fetchall()
                result[db_path.name] = "ok" if rows == [("ok",)] else "; ".join(str(x[0]) for x in rows)
            finally:
                connection.close()
        except Exception as exc:
            result[db_path.name] = f"error: {exc}"
    return result


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------


def assess_topic(
    topic: str,
    spec: TopicSpec,
    actual_type: Optional[str],
    samples: List[Tuple[int, Optional[int]]],
    bag_start_ns: int,
    bag_duration_ns: int,
    args: argparse.Namespace,
) -> dict:
    result: Dict[str, Any] = {
        "topic": topic,
        "role": spec.role,
        "status": "PASS",
        "actual_type": actual_type,
        "count": len(samples),
        "issues": [],
    }
    if actual_type is None or not samples:
        result["status"] = "FAIL" if spec.required else "SKIP"
        result["issues"].append(
            "missing_or_empty" if spec.required else "optional_topic_not_recorded"
        )
        return result

    receive = np.asarray([item[0] for item in samples], dtype=np.int64)
    receive_deltas = np.diff(receive).astype(np.float64) / 1e9
    receive_duration = float((receive[-1] - receive[0]) / 1e9) if len(receive) > 1 else 0.0

    timeline = receive
    timing_source = "receive"
    valid = np.array([], dtype=bool)
    header_deltas_ns = np.array([], dtype=np.int64)
    latency_ms = np.array([], dtype=np.float64)
    if spec.has_header:
        header = np.asarray([item[1] if item[1] is not None else 0 for item in samples], dtype=np.int64)
        valid = header > 0
        valid_header = header[valid]
        if valid_header.size:
            # Large image writes make recorder receive times bursty; source
            # header stamps represent actual frame continuity more accurately.
            timeline = valid_header
            timing_source = "header"
            header_deltas_ns = np.diff(valid_header)
            latency_ms = (receive[valid] - valid_header).astype(np.float64) / 1e6

    deltas = np.diff(timeline).astype(np.float64) / 1e9
    active_duration = float((timeline[-1] - timeline[0]) / 1e9) if len(timeline) > 1 else 0.0
    bag_duration = bag_duration_ns / 1e9
    # The nominal period is measured, not assumed, so one set of thresholds
    # works for a 30 Hz camera and a 500 Hz joint-state stream alike.
    median_period = float(np.median(deltas)) if deltas.size else 0.0

    result.update({
        "first_receive_ns": int(receive[0]),
        "last_receive_ns": int(receive[-1]),
        "start_offset_s": float((receive[0] - bag_start_ns) / 1e9),
        "end_offset_s": float((bag_start_ns + bag_duration_ns - receive[-1]) / 1e9),
        "active_duration_s": active_duration,
        "coverage_ratio": receive_duration / bag_duration if bag_duration > 0 else None,
        "actual_hz": (len(timeline) - 1) / active_duration if active_duration > 0 else None,
        "measured_period_ms": median_period * 1e3 if median_period else None,
        "timing_source": timing_source,
        "median_dt_ms": percentile(deltas * 1e3, 50),
        "p95_dt_ms": percentile(deltas * 1e3, 95),
        "max_gap_ms": percentile(deltas * 1e3, 100),
        "receive_max_gap_ms": percentile(receive_deltas * 1e3, 100),
        "duplicate_receive_timestamps": int(np.count_nonzero(receive_deltas == 0)),
    })

    if spec.continuous and median_period > 0 and deltas.size:
        gap_mask = deltas > args.gap_factor * median_period
        estimated_drops = int(np.maximum(np.rint(deltas / median_period).astype(np.int64) - 1, 0).sum())
        drop_ratio = estimated_drops / (len(timeline) + estimated_drops)
        result.update({
            "gap_count": int(np.count_nonzero(gap_mask)),
            "estimated_dropped_frames": estimated_drops,
            "estimated_drop_ratio": float(drop_ratio),
        })
        if drop_ratio >= args.fail_drop_ratio:
            result["status"] = "FAIL"
            result["issues"].append("high_estimated_drop_ratio")
        elif drop_ratio >= args.warn_drop_ratio and result["status"] == "PASS":
            result["status"] = "WARN"
            result["issues"].append("estimated_drops")

    if result["duplicate_receive_timestamps"]:
        if result["status"] == "PASS":
            result["status"] = "WARN"
        result["issues"].append("duplicate_receive_timestamps")

    if spec.has_header:
        result.update({
            "invalid_or_zero_header_stamps": int(np.count_nonzero(~valid)),
            "header_regressions": int(np.count_nonzero(header_deltas_ns < 0)),
            "duplicate_header_stamps": int(np.count_nonzero(header_deltas_ns == 0)),
            "header_to_receive_median_ms": percentile(latency_ms, 50),
            "header_to_receive_abs_p95_ms": percentile(np.abs(latency_ms), 95),
            "header_to_receive_abs_max_ms": percentile(np.abs(latency_ms), 100),
        })
        if result["invalid_or_zero_header_stamps"] or result["header_regressions"]:
            result["status"] = "FAIL"
            result["issues"].append("invalid_header_timestamp")
        elif result["duplicate_header_stamps"]:
            if result["status"] == "PASS":
                result["status"] = "WARN"
            result["issues"].append("duplicate_header_timestamps")

        latency_p95 = result["header_to_receive_abs_p95_ms"]
        if latency_p95 is not None and latency_p95 >= args.fail_latency_ms:
            result["status"] = "FAIL"
            result["issues"].append("high_recording_latency")
        elif latency_p95 is not None and latency_p95 >= args.warn_latency_ms:
            if result["status"] == "PASS":
                result["status"] = "WARN"
            result["issues"].append("recording_latency")

    return result


def nearest_offsets_ms(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    if reference.size == 0 or target.size == 0:
        return np.array([], dtype=np.float64)
    right = np.searchsorted(target, reference, side="left")
    right = np.clip(right, 0, target.size - 1)
    left = np.clip(right - 1, 0, target.size - 1)
    return np.minimum(np.abs(reference - target[left]), np.abs(target[right] - reference)).astype(
        np.float64
    ) / 1e6


def synchronization_report(
    profile: RobotProfile,
    samples: Dict[str, List[Tuple[int, Optional[int]]]],
    include_depth: bool,
    warn_ms: float,
) -> List[dict]:
    """Compare every camera against the anchor camera, plus RGB/depth pairs."""
    anchor = profile.resolved_anchor_camera
    pairs: List[Tuple[str, str, str, str]] = []
    for name, topic in profile.cameras.items():
        if name != anchor:
            pairs.append((f"{anchor}:rgb", f"{name}:rgb", profile.cameras[anchor], topic))
    if include_depth:
        for name, topic in profile.depths.items():
            if name in profile.cameras:
                pairs.append((f"{name}:rgb", f"{name}:depth", profile.cameras[name], topic))

    def stamps(topic: str) -> np.ndarray:
        return np.asarray(
            [item[1] for item in samples.get(topic, []) if item[1] is not None and item[1] > 0],
            dtype=np.int64,
        )

    reports = []
    for left_label, right_label, left_topic, right_topic in pairs:
        left, right = stamps(left_topic), stamps(right_topic)
        if left.size == 0 or right.size == 0:
            reports.append({"reference": left_label, "target": right_label, "status": "SKIP",
                            "median_ms": None, "p95_ms": None, "max_ms": None})
            continue
        offsets = nearest_offsets_ms(left, right)
        p95 = percentile(offsets, 95)
        reports.append({
            "reference": left_label,
            "target": right_label,
            "status": "WARN" if p95 is not None and p95 > warn_ms else "PASS",
            "median_ms": percentile(offsets, 50),
            "p95_ms": p95,
            "max_ms": percentile(offsets, 100),
        })
    return reports


def teleop_report(
    profile: RobotProfile,
    samples: Dict[str, List[Tuple[int, Optional[int]]]],
    bag_start_ns: int,
    gap_warn_s: float,
    action_gap_policy: str,
) -> dict:
    """When each arm was actually commanded, and whether the arms overlap.

    The converter's episode window is the intersection of the arm command
    spans.  Under ``hold-last-command`` every row needs a genuine command on
    *every* arm, so arms teleoperated one after the other yield no convertible
    rows -- far easier to see here than in a conversion failure message.  Under
    ``joint-state-fill`` a silent arm is filled from its own joint_states, so
    the same recording is convertible and the finding drops to a warning.
    """
    fillable = action_gap_policy == "joint-state-fill"
    spans: Dict[str, Any] = {}
    active: List[Tuple[int, int]] = []
    for topic in profile.arm.command_topics:
        times = np.asarray([item[0] for item in samples.get(topic, [])], dtype=np.int64)
        if times.size < 2:
            spans[topic] = {"count": int(times.size), "bursts": []}
            continue
        deltas = np.diff(times)
        threshold = int(gap_warn_s * 1e9)
        breaks = np.flatnonzero(deltas > threshold)
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [times.size - 1]))
        bursts = [
            {"start_s": float((times[s] - bag_start_ns) / 1e9),
             "end_s": float((times[e] - bag_start_ns) / 1e9),
             "duration_s": float((times[e] - times[s]) / 1e9)}
            for s, e in zip(starts, ends)
        ]
        spans[topic] = {
            "count": int(times.size),
            "first_s": float((times[0] - bag_start_ns) / 1e9),
            "last_s": float((times[-1] - bag_start_ns) / 1e9),
            "max_gap_s": float(deltas.max() / 1e9),
            "bursts": bursts,
        }
        active.append((int(times[0]), int(times[-1])))

    result: Dict[str, Any] = {"per_topic": spans, "status": "PASS", "issues": []}
    if len(active) == len(profile.arm.command_topics) and active:
        overlap_start = max(item[0] for item in active)
        overlap_end = min(item[1] for item in active)
        result["overlap_s"] = float((overlap_end - overlap_start) / 1e9)
        if overlap_end <= overlap_start:
            result["status"] = "FAIL"
            result["issues"].append("arm_command_spans_do_not_overlap")
        else:
            # A hole wider than the overlap means the whole convertible window
            # sits inside one arm's silence: no tick can have both arms real.
            for topic, span in spans.items():
                if span.get("max_gap_s", 0.0) >= result["overlap_s"]:
                    if fillable:
                        if result["status"] == "PASS":
                            result["status"] = "WARN"
                        result["issues"].append(
                            f"{topic}_silent_across_entire_overlap"
                            "（joint-state-fill 将用该臂实测 joint_states 填充整段）"
                        )
                    else:
                        result["status"] = "FAIL"
                        result["issues"].append(
                            f"{topic}_silent_across_entire_overlap"
                            "（改用 --action-gap-policy joint-state-fill 可转换）"
                        )
                elif span.get("max_gap_s", 0.0) > gap_warn_s and result["status"] == "PASS":
                    result["status"] = "WARN"
                    result["issues"].append(f"{topic}_command_gap")
    else:
        result["status"] = "FAIL"
        result["issues"].append("arm_command_topic_missing")
    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_report(report: dict) -> None:
    print(f"ROSbag 质量检查：{report['overall_status']}")
    print(f"Bag: {report['bag']}")
    print(f"Profile: {report['profile']}  存储后端: {report['storage']}")
    print(f"时长 {report['duration_s']:.3f} s | 相关消息 {report['selected_message_count']} "
          f"| 体积 {report['size_gib']:.2f} GiB")
    print()
    print("状态  角色          数量      实测Hz    最大间隔  估算丢帧      Header异常  延迟P95   话题")
    for item in report["topics"]:
        drop_count = item.get("estimated_dropped_frames")
        drop_ratio = item.get("estimated_drop_ratio")
        drops = "-" if drop_count is None else f"{drop_count}({100 * drop_ratio:.2f}%)"
        header_errors = (item.get("invalid_or_zero_header_stamps", 0)
                         + item.get("header_regressions", 0))
        latency = item.get("header_to_receive_abs_p95_ms")
        latency_display = "-" if latency is None else f"{fmt(latency, 1)} ms"
        print(f"{item['status']:<5} {item['role']:<13}{item['count']:>7}  "
              f"{fmt(item.get('actual_hz'), 1):>8}  {fmt(item.get('max_gap_ms'), 1):>9}ms  "
              f"{drops:>12}  {header_errors:>8}  {latency_display:>10}  {item['topic']}")

    missing = [item["topic"] for item in report["topics"] if "missing_or_empty" in item["issues"]]
    if missing:
        print(f"\n缺失或为空的必需话题（profile {report['profile']} 需要）：")
        for topic in missing:
            print(f"  {topic}")

    teleop = report["teleop"]
    print("\n遥操作命令活动区间：")
    for topic, span in teleop["per_topic"].items():
        if not span.get("bursts"):
            print(f"  {topic}: 无数据")
            continue
        segments = " ".join(f"[{b['start_s']:.1f}s→{b['end_s']:.1f}s]" for b in span["bursts"][:6])
        more = f" 等 {len(span['bursts'])} 段" if len(span["bursts"]) > 6 else ""
        print(f"  {topic}: {span['count']} 条，最大断档 {span['max_gap_s']:.1f}s")
        print(f"      {segments}{more}")
    if "overlap_s" in teleop:
        print(f"  两臂命令重叠区间：{teleop['overlap_s']:.1f}s")
    for issue in teleop["issues"]:
        print(f"  {teleop['status']}: {issue}")

    if report["synchronization"]:
        print("\n相机时间同步（Header 时间戳最近邻偏差）：")
        for item in report["synchronization"]:
            print(f"{item['status']:<5} {item['reference']:>14} -> {item['target']:<14} "
                  f"median={fmt(item['median_ms'])} ms p95={fmt(item['p95_ms'])} ms "
                  f"max={fmt(item['max_ms'])} ms")

    bad = {name: status for name, status in report["database_integrity"].items()
           if status not in {"ok", "skipped"}}
    if bad:
        print("\nSQLite 完整性错误：")
        for name, status in bad.items():
            print(f"  {name}: {status}")

    if report["overall_status"] == "PASS":
        print("\n结论：关键话题、时间戳、帧间隔与遥操作覆盖均通过当前阈值。")
    elif report["overall_status"] == "WARN":
        print("\n结论：bag 可读取，但存在时间间隔、同步或命令断档警告，请查看 WARN 行。")
    else:
        print("\n结论：存在缺失话题、时间戳错误、高丢帧、命令不重叠或文件损坏。")


def main() -> int:
    args = parse_args()
    bag_dir = args.bag.expanduser().resolve()
    try:
        profile = apply_topic_overrides(
            load_profile(args.profile),
            cameras=parse_name_topic(args.camera),
            depths=parse_name_topic(args.depth),
            anchor_camera=args.anchor_camera,
        )
    except ProfileError as exc:
        print(f"检查失败：{exc}", file=sys.stderr)
        return 2

    specs = build_specs(profile, args.include_depth)
    try:
        types, samples, start_ns, end_ns = read_bag(bag_dir, profile, specs)
    except Exception as exc:
        print(f"检查失败：{exc}", file=sys.stderr)
        return 2

    integrity = sqlite_integrity(bag_dir, run_quick_check=not args.no_quick_check)
    storage = "sqlite3" if any(bag_dir.glob("*.db3")) else "mcap" if any(bag_dir.glob("*.mcap")) else "unknown"
    duration_ns = max(end_ns - start_ns, 0)

    topic_reports = [
        assess_topic(topic, spec, types.get(topic), samples[topic], start_ns, duration_ns, args)
        for topic, spec in specs.items()
    ]
    sync_reports = synchronization_report(profile, samples, args.include_depth, args.sync_warn_ms)
    teleop = teleop_report(profile, samples, start_ns, args.command_gap_warn_s, args.action_gap_policy)

    statuses = [item["status"] for item in topic_reports]
    statuses.extend(item["status"] for item in sync_reports)
    statuses.append(teleop["status"])
    statuses.extend("FAIL" if status not in {"ok", "skipped"} else "PASS" for status in integrity.values())
    overall = "FAIL" if "FAIL" in statuses else "WARN" if "WARN" in statuses else "PASS"

    size_gib = sum(path.stat().st_size for path in bag_dir.iterdir() if path.is_file()) / (1024**3)
    report = {
        "schema_version": 2,
        "generated_at": datetime.now().astimezone().isoformat(),
        "bag": str(bag_dir),
        "profile": profile.name,
        "storage": storage,
        "overall_status": overall,
        "duration_s": duration_ns / 1e9,
        "selected_message_count": sum(len(values) for values in samples.values()),
        "size_gib": size_gib,
        "database_integrity": integrity,
        "thresholds": {
            "gap_factor": args.gap_factor,
            "warn_drop_ratio": args.warn_drop_ratio,
            "fail_drop_ratio": args.fail_drop_ratio,
            "sync_warn_ms": args.sync_warn_ms,
            "warn_latency_ms": args.warn_latency_ms,
            "fail_latency_ms": args.fail_latency_ms,
            "command_gap_warn_s": args.command_gap_warn_s,
        },
        "topics": topic_reports,
        "synchronization": sync_reports,
        "teleop": teleop,
    }
    print_report(report)

    if args.json_path:
        json_path = args.json_path.expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"JSON 报告：{json_path}")

    return 0 if overall == "PASS" else 1 if overall == "WARN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
