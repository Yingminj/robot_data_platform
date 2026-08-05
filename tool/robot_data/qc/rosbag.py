"""Offline quality checks for ROS 2 bags, driven by a robot profile.

The topic list, expected end effectors and camera set come from the same profile
description the converters use, so this checker and the conversion path always
agree about what a recording should contain.  Both rosbag2 storage backends
(sqlite3 ``.db3`` and MCAP ``.mcap``) are read through the shared reader, which
also supplies message definitions that a ``.db3`` does not carry.

Only timestamps are read: image payloads are never decoded, so a multi-GiB bag
is checked in seconds.  Nominal publishing rates are measured from the data
(median inter-message period) rather than hardcoded, so the same thresholds
apply to a 30 Hz camera and a 500 Hz joint-state stream.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from robot_data.align.bag_io import BagScan, TopicScan, scan_timestamps
from robot_data.discovery import bag_storage_kind
from robot_data.profiles.schema import FLOAT32, FLOAT32MULTIARRAY, RobotProfile
from robot_data.qc.inventory import Inventory, build_inventory


@dataclass(frozen=True)
class QualityThresholds:
    gap_factor: float = 1.5
    warn_drop_ratio: float = 0.01
    fail_drop_ratio: float = 0.05
    sync_warn_ms: float = 20.0
    warn_latency_ms: float = 100.0
    fail_latency_ms: float = 500.0
    command_gap_warn_s: float = 1.0

    def __post_init__(self) -> None:
        if self.gap_factor <= 1.0:
            raise ValueError("gap_factor must be greater than 1")
        if not 0 <= self.warn_drop_ratio <= self.fail_drop_ratio <= 1:
            raise ValueError("drop ratios must satisfy 0 <= warn <= fail <= 1")
        if not 0 <= self.warn_latency_ms <= self.fail_latency_ms:
            raise ValueError("latency thresholds must satisfy 0 <= warn <= fail")


@dataclass
class TopicSpec:
    """What a topic is used for, derived from the profile."""

    role: str  # camera / depth / arm_state / arm_command / ee_state / ee_command
    has_header: bool
    required: bool = True
    # Sensor streams publish at a steady rate, so a gap really is a dropped
    # message.  Command streams only publish while teleoperation is enabled, so
    # their gaps are operator behaviour and are reported by teleop_report
    # instead of being scored as drops.
    continuous: bool = True


def build_specs(profile: RobotProfile, include_depth: bool) -> dict[str, TopicSpec]:
    """Map every topic the profile needs to its role and header expectation."""
    specs: dict[str, TopicSpec] = {}
    for topic in profile.cameras.values():
        specs[topic] = TopicSpec("camera", has_header=True)
    if include_depth:
        for topic in profile.depths.values():
            specs[topic] = TopicSpec("depth", has_header=True, required=False)
    specs[profile.arm.joint_states_topic] = TopicSpec("arm_state", has_header=True)
    for topic in profile.arm.command_topics:
        specs[topic] = TopicSpec("arm_command", has_header=True, continuous=False)
    for effector in profile.end_effectors:
        # std_msgs/Float32 and Float32MultiArray have no Header; record time is
        # their only clock.
        headerless = {FLOAT32, FLOAT32MULTIARRAY}
        specs[effector.command_topic] = TopicSpec(
            "ee_command", has_header=effector.command_kind not in headerless, continuous=False
        )
        if effector.state_topic:
            # A measured end-effector feedback topic only enriches the
            # observation; recordings that lack it fall back to the command
            # echo, so its absence is reported but does not fail the bag.
            specs[effector.state_topic] = TopicSpec(
                "ee_state",
                has_header=effector.state_kind not in headerless,
                required=False,
            )
    return specs


def percentile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def sqlite_integrity(bag_dir: Path, run_quick_check: bool) -> dict[str, str]:
    """PRAGMA quick_check on every .db3; MCAP bags return an empty mapping."""
    result: dict[str, str] = {}
    for db_path in sorted(bag_dir.glob("*.db3")):
        if not run_quick_check:
            result[db_path.name] = "skipped"
            continue
        try:
            connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                rows = connection.execute("PRAGMA quick_check").fetchall()
                result[db_path.name] = (
                    "ok" if rows == [("ok",)] else "; ".join(str(x[0]) for x in rows)
                )
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
    scan: TopicScan | None,
    bag_start_ns: int,
    bag_duration_ns: int,
    limits: QualityThresholds,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "topic": topic,
        "role": spec.role,
        "status": "PASS",
        "actual_type": scan.msgtype if scan else None,
        "count": scan.count if scan else 0,
        "issues": [],
    }
    if scan is None or not scan.count:
        result["status"] = "FAIL" if spec.required else "SKIP"
        result["issues"].append(
            "missing_or_empty" if spec.required else "optional_topic_not_recorded"
        )
        return result

    receive = np.asarray(scan.receive_ns, dtype=np.int64)
    receive_deltas = np.diff(receive).astype(np.float64) / 1e9
    receive_duration = float((receive[-1] - receive[0]) / 1e9) if len(receive) > 1 else 0.0

    timeline = receive
    timing_source = "receive"
    valid = np.array([], dtype=bool)
    header_deltas_ns = np.array([], dtype=np.int64)
    latency_ms = np.array([], dtype=np.float64)
    if spec.has_header:
        header = np.asarray(
            [item if item is not None else 0 for item in scan.header_ns], dtype=np.int64
        )
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

    result.update(
        {
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
        }
    )

    if spec.continuous and median_period > 0 and deltas.size:
        gap_mask = deltas > limits.gap_factor * median_period
        estimated_drops = int(
            np.maximum(np.rint(deltas / median_period).astype(np.int64) - 1, 0).sum()
        )
        drop_ratio = estimated_drops / (len(timeline) + estimated_drops)
        result.update(
            {
                "gap_count": int(np.count_nonzero(gap_mask)),
                "estimated_dropped_frames": estimated_drops,
                "estimated_drop_ratio": float(drop_ratio),
            }
        )
        if drop_ratio >= limits.fail_drop_ratio:
            # Optional streams only enrich the output, and are read with a
            # tolerance far wider than their period, so their jitter is worth
            # reporting but must not condemn an otherwise sound recording.
            result["status"] = "FAIL" if spec.required else "WARN"
            result["issues"].append("high_estimated_drop_ratio")
        elif drop_ratio >= limits.warn_drop_ratio and result["status"] == "PASS":
            result["status"] = "WARN"
            result["issues"].append("estimated_drops")

    if result["duplicate_receive_timestamps"]:
        if result["status"] == "PASS":
            result["status"] = "WARN"
        result["issues"].append("duplicate_receive_timestamps")

    if spec.has_header:
        result.update(
            {
                "invalid_or_zero_header_stamps": int(np.count_nonzero(~valid)),
                "header_regressions": int(np.count_nonzero(header_deltas_ns < 0)),
                "duplicate_header_stamps": int(np.count_nonzero(header_deltas_ns == 0)),
                "header_to_receive_median_ms": percentile(latency_ms, 50),
                "header_to_receive_abs_p95_ms": percentile(np.abs(latency_ms), 95),
                "header_to_receive_abs_max_ms": percentile(np.abs(latency_ms), 100),
            }
        )
        if result["invalid_or_zero_header_stamps"] or result["header_regressions"]:
            result["status"] = "FAIL"
            result["issues"].append("invalid_header_timestamp")
        elif result["duplicate_header_stamps"]:
            if result["status"] == "PASS":
                result["status"] = "WARN"
            result["issues"].append("duplicate_header_timestamps")

        latency_p95 = result["header_to_receive_abs_p95_ms"]
        if latency_p95 is not None and latency_p95 >= limits.fail_latency_ms:
            result["status"] = "FAIL"
            result["issues"].append("high_recording_latency")
        elif latency_p95 is not None and latency_p95 >= limits.warn_latency_ms:
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
    return np.minimum(
        np.abs(reference - target[left]), np.abs(target[right] - reference)
    ).astype(np.float64) / 1e6


def synchronization_report(
    profile: RobotProfile,
    scan: BagScan,
    include_depth: bool,
    warn_ms: float,
) -> list[dict[str, Any]]:
    """Compare every camera against the anchor camera, plus RGB/depth pairs."""
    anchor = profile.resolved_anchor_camera
    pairs: list[tuple[str, str, str, str]] = []
    for name, topic in profile.cameras.items():
        if name != anchor and topic != profile.cameras[anchor]:
            # Cameras cut from one mosaic share a topic and therefore a stamp;
            # comparing them would report a meaningless perfect zero.
            pairs.append((f"{anchor}:rgb", f"{name}:rgb", profile.cameras[anchor], topic))
    if include_depth:
        for name, topic in profile.depths.items():
            if name in profile.cameras:
                pairs.append((f"{name}:rgb", f"{name}:depth", profile.cameras[name], topic))

    def stamps(topic: str) -> np.ndarray:
        entry = scan.topics.get(topic)
        if entry is None:
            return np.asarray([], dtype=np.int64)
        return np.asarray(
            [item for item in entry.header_ns if item is not None and item > 0], dtype=np.int64
        )

    reports = []
    for left_label, right_label, left_topic, right_topic in pairs:
        left, right = stamps(left_topic), stamps(right_topic)
        if left.size == 0 or right.size == 0:
            reports.append(
                {
                    "reference": left_label,
                    "target": right_label,
                    "status": "SKIP",
                    "median_ms": None,
                    "p95_ms": None,
                    "max_ms": None,
                }
            )
            continue
        offsets = nearest_offsets_ms(left, right)
        p95 = percentile(offsets, 95)
        reports.append(
            {
                "reference": left_label,
                "target": right_label,
                "status": "WARN" if p95 is not None and p95 > warn_ms else "PASS",
                "median_ms": percentile(offsets, 50),
                "p95_ms": p95,
                "max_ms": percentile(offsets, 100),
            }
        )
    return reports


def teleop_report(
    profile: RobotProfile,
    scan: BagScan,
    bag_start_ns: int,
    gap_warn_s: float,
    action_gap_policy: str,
) -> dict[str, Any]:
    """When each arm was actually commanded, and whether the arms overlap.

    The converter's episode window is the intersection of the arm command
    spans.  Under ``hold-last-command`` every row needs a genuine command on
    *every* arm, so arms teleoperated one after the other yield no convertible
    rows -- far easier to see here than in a conversion failure message.  Under
    ``joint-state-fill`` a silent arm is filled from its own joint_states, so
    the same recording is convertible and the finding drops to a warning.
    """
    fillable = action_gap_policy == "joint-state-fill"
    spans: dict[str, Any] = {}
    active: list[tuple[int, int]] = []
    for topic in profile.arm.command_topics:
        entry = scan.topics.get(topic)
        times = np.asarray(entry.receive_ns if entry else [], dtype=np.int64)
        if times.size < 2:
            spans[topic] = {"count": int(times.size), "bursts": []}
            continue
        deltas = np.diff(times)
        threshold = int(gap_warn_s * 1e9)
        breaks = np.flatnonzero(deltas > threshold)
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [times.size - 1]))
        bursts = [
            {
                "start_s": float((times[s] - bag_start_ns) / 1e9),
                "end_s": float((times[e] - bag_start_ns) / 1e9),
                "duration_s": float((times[e] - times[s]) / 1e9),
            }
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

    result: dict[str, Any] = {"per_topic": spans, "status": "PASS", "issues": []}
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
# Orchestration
# ---------------------------------------------------------------------------


def check_bag(
    bag_dir: Path,
    profile: RobotProfile,
    include_depth: bool = False,
    limits: QualityThresholds | None = None,
    action_gap_policy: str = "hold-last-command",
    run_quick_check: bool = True,
    recipe_name: str | None = None,
) -> tuple[dict[str, Any], Inventory]:
    """Read one bag once and produce both the quality report and the inventory."""
    limits = limits or QualityThresholds()
    specs = build_specs(profile, include_depth)
    header_topics = {topic for topic, spec in specs.items() if spec.has_header}
    # Every topic is scanned, not just the profile's: the inventory has to be
    # able to name the stream the profile *should* have pointed at.
    scan = scan_timestamps(bag_dir, profile, header_topics=header_topics, only_topics=None)

    inventory = build_inventory(scan, profile, include_depth=include_depth)
    integrity = sqlite_integrity(bag_dir, run_quick_check=run_quick_check)
    storage = bag_storage_kind(bag_dir)

    topic_reports = [
        assess_topic(topic, spec, scan.topics.get(topic), scan.start_ns, scan.duration_ns, limits)
        for topic, spec in specs.items()
    ]
    sync_reports = synchronization_report(profile, scan, include_depth, limits.sync_warn_ms)
    teleop = teleop_report(
        profile, scan, scan.start_ns, limits.command_gap_warn_s, action_gap_policy
    )

    statuses = [item["status"] for item in topic_reports]
    statuses.extend(item["status"] for item in sync_reports)
    statuses.append(teleop["status"])
    statuses.extend(
        "FAIL" if status not in {"ok", "skipped"} else "PASS" for status in integrity.values()
    )
    overall = "FAIL" if "FAIL" in statuses else "WARN" if "WARN" in statuses else "PASS"

    size_gib = sum(path.stat().st_size for path in bag_dir.iterdir() if path.is_file()) / (1024**3)
    report = {
        "schema_version": 3,
        "generated_at": datetime.now().astimezone().isoformat(),
        "bag": str(bag_dir),
        "profile": profile.name,
        "recipe": recipe_name,
        "storage": storage,
        "overall_status": overall,
        "duration_s": scan.duration_ns / 1e9,
        "selected_message_count": sum(
            entry.count for topic, entry in scan.topics.items() if topic in specs
        ),
        "total_message_count": sum(entry.count for entry in scan.topics.values()),
        "size_gib": size_gib,
        "database_integrity": integrity,
        "thresholds": {
            "gap_factor": limits.gap_factor,
            "warn_drop_ratio": limits.warn_drop_ratio,
            "fail_drop_ratio": limits.fail_drop_ratio,
            "sync_warn_ms": limits.sync_warn_ms,
            "warn_latency_ms": limits.warn_latency_ms,
            "fail_latency_ms": limits.fail_latency_ms,
            "command_gap_warn_s": limits.command_gap_warn_s,
        },
        "inventory": inventory.to_dict(),
        "topics": topic_reports,
        "synchronization": sync_reports,
        "teleop": teleop,
    }
    return report, inventory


def print_report(report: dict[str, Any]) -> None:
    print(f"ROSbag 质量检查：{report['overall_status']}")
    print(f"Bag: {report['bag']}")
    label = f"Profile: {report['profile']}"
    if report.get("recipe"):
        label += f"  Recipe: {report['recipe']}"
    print(f"{label}  存储后端: {report['storage']}")
    print(
        f"时长 {report['duration_s']:.3f} s | 相关消息 {report['selected_message_count']} "
        f"/ 全部 {report['total_message_count']} | 体积 {report['size_gib']:.2f} GiB"
    )
    print()
    print(
        "状态  角色          数量      实测Hz    最大间隔  估算丢帧      Header异常  延迟P95   话题"
    )
    for item in report["topics"]:
        drop_count = item.get("estimated_dropped_frames")
        drop_ratio = item.get("estimated_drop_ratio")
        drops = "-" if drop_count is None else f"{drop_count}({100 * drop_ratio:.2f}%)"
        header_errors = item.get("invalid_or_zero_header_stamps", 0) + item.get(
            "header_regressions", 0
        )
        latency = item.get("header_to_receive_abs_p95_ms")
        latency_display = "-" if latency is None else f"{fmt(latency, 1)} ms"
        print(
            f"{item['status']:<5} {item['role']:<13}{item['count']:>7}  "
            f"{fmt(item.get('actual_hz'), 1):>8}  {fmt(item.get('max_gap_ms'), 1):>9}ms  "
            f"{drops:>12}  {header_errors:>8}  {latency_display:>10}  {item['topic']}"
        )

    missing = [item["topic"] for item in report["topics"] if "missing_or_empty" in item["issues"]]
    if missing:
        print(f"\n缺失或为空的必需话题（profile {report['profile']} 需要）：")
        for topic in missing:
            print(f"  {topic}")
        print(
            "  转换时可用 --missing-topic-policy fill 由实测状态重建这些列"
            "（重建列等于 observation，训练前请确认该自由度确实静止）"
        )

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
            print(
                f"{item['status']:<5} {item['reference']:>14} -> {item['target']:<14} "
                f"median={fmt(item['median_ms'])} ms p95={fmt(item['p95_ms'])} ms "
                f"max={fmt(item['max_ms'])} ms"
            )

    bad = {
        name: status
        for name, status in report["database_integrity"].items()
        if status not in {"ok", "skipped"}
    }
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
