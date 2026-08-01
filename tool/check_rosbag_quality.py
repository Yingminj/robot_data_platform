#!/usr/bin/env python3
"""Fast offline quality checks for this project's ROS 2 sqlite3 bags.

The checker reads timestamps and only the first bytes of each serialized message;
it does not decode image payloads and does not run ``ros2 bag play``.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import struct
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml


@dataclass(frozen=True)
class TopicSpec:
    msg_type: str
    expected_hz: Optional[float] = None
    has_header: bool = False
    required: bool = True


TOPICS: Dict[str, TopicSpec] = {
    "/camera_head/camera_head/color/image_raw": TopicSpec(
        "sensor_msgs/msg/Image", 30.0, True
    ),
    "/camera_head/camera_head/color/camera_info": TopicSpec(
        "sensor_msgs/msg/CameraInfo", 30.0, True
    ),
    "/camera_head/camera_head/aligned_depth_to_color/image_raw": TopicSpec(
        "sensor_msgs/msg/Image", 30.0, True, False
    ),
    "/camera_left_wrist/camera_left_wrist/color/image_rect_raw": TopicSpec(
        "sensor_msgs/msg/Image", 30.0, True
    ),
    "/camera_left_wrist/camera_left_wrist/color/camera_info": TopicSpec(
        "sensor_msgs/msg/CameraInfo", 30.0, True
    ),
    "/camera_left_wrist/camera_left_wrist/aligned_depth_to_color/image_raw": TopicSpec(
        "sensor_msgs/msg/Image", 30.0, True, False
    ),
    "/camera_right_wrist/camera_right_wrist/color/image_rect_raw": TopicSpec(
        "sensor_msgs/msg/Image", 30.0, True
    ),
    "/camera_right_wrist/camera_right_wrist/color/camera_info": TopicSpec(
        "sensor_msgs/msg/CameraInfo", 30.0, True
    ),
    "/camera_right_wrist/camera_right_wrist/aligned_depth_to_color/image_raw": TopicSpec(
        "sensor_msgs/msg/Image", 30.0, True, False
    ),
    "/joint_states": TopicSpec("sensor_msgs/msg/JointState", 500.0, True),
    "/control/joint_cmd_A": TopicSpec("marvin_msgs/msg/Jointcmd", None, True),
    "/control/joint_cmd_B": TopicSpec("marvin_msgs/msg/Jointcmd", None, True),
    "/control/gripperValueL": TopicSpec("std_msgs/msg/Float32"),
    "/control/gripperValueR": TopicSpec("std_msgs/msg/Float32"),
}

SYNC_PAIRS = (
    (
        "/camera_head/camera_head/color/image_raw",
        "/camera_left_wrist/camera_left_wrist/color/image_rect_raw",
    ),
    (
        "/camera_head/camera_head/color/image_raw",
        "/camera_right_wrist/camera_right_wrist/color/image_rect_raw",
    ),
    (
        "/camera_head/camera_head/color/image_raw",
        "/camera_head/camera_head/aligned_depth_to_color/image_raw",
    ),
    (
        "/camera_left_wrist/camera_left_wrist/color/image_rect_raw",
        "/camera_left_wrist/camera_left_wrist/aligned_depth_to_color/image_raw",
    ),
    (
        "/camera_right_wrist/camera_right_wrist/color/image_rect_raw",
        "/camera_right_wrist/camera_right_wrist/aligned_depth_to_color/image_raw",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查 ROS 2 bag 的话题完整性、时间戳、频率、间隔和估算丢帧。"
    )
    parser.add_argument("bag", type=Path, help="包含 metadata.yaml 的 rosbag2 目录")
    parser.add_argument(
        "--json",
        type=Path,
        dest="json_path",
        help="同时写出机器可读 JSON 报告",
    )
    parser.add_argument(
        "--gap-factor",
        type=float,
        default=1.5,
        help="间隔超过期望周期的此倍数视为掉帧，默认 1.5",
    )
    parser.add_argument(
        "--warn-drop-ratio",
        type=float,
        default=0.01,
        help="估算丢帧率达到此值时警告，默认 0.01",
    )
    parser.add_argument(
        "--fail-drop-ratio",
        type=float,
        default=0.05,
        help="估算丢帧率达到此值时失败，默认 0.05",
    )
    parser.add_argument(
        "--sync-warn-ms",
        type=float,
        default=20.0,
        help="相机时间戳最近邻偏差 P95 的警告阈值，默认 20 ms",
    )
    parser.add_argument(
        "--warn-latency-ms",
        type=float,
        default=100.0,
        help="Header 到 rosbag 接收时间的 P95 延迟警告阈值，默认 100 ms",
    )
    parser.add_argument(
        "--fail-latency-ms",
        type=float,
        default=500.0,
        help="Header 到 rosbag 接收时间的 P95 延迟失败阈值，默认 500 ms",
    )
    parser.add_argument(
        "--no-quick-check",
        action="store_true",
        help="跳过 SQLite PRAGMA quick_check",
    )
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


def decode_header_stamp(prefix: Optional[bytes]) -> Optional[int]:
    """Read the first std_msgs/Header stamp from a CDR serialized message."""
    if prefix is None or len(prefix) < 12:
        return None
    representation = int.from_bytes(prefix[0:2], byteorder="big")
    if representation in {0x0001, 0x0003, 0x0007, 0x0009}:
        endian = "<"
    elif representation in {0x0000, 0x0002, 0x0006, 0x0008}:
        endian = ">"
    else:
        return None
    sec, nanosec = struct.unpack_from(f"{endian}iI", prefix, 4)
    if not 0 <= nanosec < 1_000_000_000:
        return None
    return sec * 1_000_000_000 + nanosec


def read_metadata(bag_dir: Path) -> Tuple[dict, List[Path]]:
    metadata_path = bag_dir / "metadata.yaml"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"找不到 {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as stream:
        root = yaml.safe_load(stream) or {}
    info = root.get("rosbag2_bagfile_information", root)
    if info.get("storage_identifier") != "sqlite3":
        raise ValueError(
            "当前快速检查器只支持 sqlite3/.db3；"
            f"检测到 {info.get('storage_identifier')!r}"
        )
    relative_paths = info.get("relative_file_paths") or []
    db_paths = [bag_dir / item for item in relative_paths if item.endswith(".db3")]
    if not db_paths:
        db_paths = sorted(bag_dir.glob("*.db3"))
    if not db_paths:
        raise FileNotFoundError(f"{bag_dir} 内没有 .db3 文件")
    missing = [str(path) for path in db_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("metadata 引用的文件不存在：" + ", ".join(missing))
    return info, db_paths


def query_database(
    db_path: Path,
    topic_names: Sequence[str],
    run_quick_check: bool,
) -> Tuple[Dict[str, str], Dict[str, List[Tuple[int, Optional[int]]]], str]:
    uri = f"file:{db_path}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        integrity = "skipped"
        if run_quick_check:
            rows = connection.execute("PRAGMA quick_check").fetchall()
            integrity = "ok" if rows == [("ok",)] else "; ".join(str(x[0]) for x in rows)

        placeholders = ",".join("?" for _ in topic_names)
        topic_rows = connection.execute(
            f"SELECT name, type FROM topics WHERE name IN ({placeholders})",
            tuple(topic_names),
        ).fetchall()
        types = {str(name): str(msg_type) for name, msg_type in topic_rows}

        samples: Dict[str, List[Tuple[int, Optional[int]]]] = {
            topic: [] for topic in topic_names
        }
        message_rows = connection.execute(
            f"""
            SELECT topics.name, messages.timestamp, substr(messages.data, 1, 12)
            FROM messages
            JOIN topics ON topics.id = messages.topic_id
            WHERE topics.name IN ({placeholders})
            ORDER BY messages.timestamp
            """,
            tuple(topic_names),
        )
        for topic, receive_ns, prefix in message_rows:
            header_ns = decode_header_stamp(prefix) if TOPICS[str(topic)].has_header else None
            samples[str(topic)].append((int(receive_ns), header_ns))
        return types, samples, integrity
    finally:
        connection.close()


def merge_samples(
    db_paths: Iterable[Path], run_quick_check: bool
) -> Tuple[Dict[str, str], Dict[str, List[Tuple[int, Optional[int]]]], Dict[str, str]]:
    all_types: Dict[str, str] = {}
    all_samples = {topic: [] for topic in TOPICS}
    integrity: Dict[str, str] = {}
    for db_path in db_paths:
        types, samples, check = query_database(db_path, list(TOPICS), run_quick_check)
        integrity[db_path.name] = check
        for topic, msg_type in types.items():
            previous = all_types.get(topic)
            if previous is not None and previous != msg_type:
                all_types[topic] = f"{previous} | {msg_type}"
            else:
                all_types[topic] = msg_type
        for topic, values in samples.items():
            all_samples[topic].extend(values)
    for values in all_samples.values():
        values.sort(key=lambda item: item[0])
    return all_types, all_samples, integrity


def assess_topic(
    topic: str,
    spec: TopicSpec,
    actual_type: Optional[str],
    samples: List[Tuple[int, Optional[int]]],
    bag_start_ns: int,
    bag_duration_ns: int,
    args: argparse.Namespace,
) -> dict:
    result = {
        "topic": topic,
        "status": "PASS",
        "expected_type": spec.msg_type,
        "actual_type": actual_type,
        "expected_hz": spec.expected_hz,
        "count": len(samples),
        "issues": [],
    }
    if actual_type is None or not samples:
        result["status"] = "FAIL" if spec.required else "SKIP"
        result["issues"].append(
            "missing_or_empty" if spec.required else "optional_topic_not_recorded"
        )
        return result
    if actual_type != spec.msg_type:
        result["status"] = "FAIL"
        result["issues"].append("type_mismatch")

    receive = np.asarray([item[0] for item in samples], dtype=np.int64)
    receive_deltas = np.diff(receive).astype(np.float64) / 1e9
    receive_duration = (
        float((receive[-1] - receive[0]) / 1e9) if len(receive) > 1 else 0.0
    )
    timeline = receive
    timing_source = "receive"
    valid = np.array([], dtype=bool)
    header_deltas_ns = np.array([], dtype=np.int64)
    latency_ms = np.array([], dtype=np.float64)
    if spec.has_header:
        header = np.asarray(
            [item[1] if item[1] is not None else 0 for item in samples], dtype=np.int64
        )
        valid = header > 0
        valid_header = header[valid]
        if valid_header.size:
            # Large image writes can make recorder receive times bursty. Source
            # header stamps represent actual frame continuity more accurately.
            timeline = valid_header
            timing_source = "header"
            header_deltas_ns = np.diff(valid_header)
            latency_ms = (receive[valid] - valid_header).astype(np.float64) / 1e6

    deltas = np.diff(timeline).astype(np.float64) / 1e9
    active_duration = (
        float((timeline[-1] - timeline[0]) / 1e9) if len(timeline) > 1 else 0.0
    )
    bag_duration = bag_duration_ns / 1e9
    result.update(
        {
            "first_receive_ns": int(receive[0]),
            "last_receive_ns": int(receive[-1]),
            "start_offset_s": float((receive[0] - bag_start_ns) / 1e9),
            "end_offset_s": float(
                (bag_start_ns + bag_duration_ns - receive[-1]) / 1e9
            ),
            "active_duration_s": active_duration,
            "coverage_ratio": receive_duration / bag_duration if bag_duration > 0 else None,
            "actual_hz": (len(timeline) - 1) / active_duration if active_duration > 0 else None,
            "timing_source": timing_source,
            "median_dt_ms": percentile(deltas * 1e3, 50),
            "p95_dt_ms": percentile(deltas * 1e3, 95),
            "max_gap_ms": percentile(deltas * 1e3, 100),
            "receive_max_gap_ms": percentile(receive_deltas * 1e3, 100),
            "duplicate_receive_timestamps": int(
                np.count_nonzero(receive_deltas == 0)
            ),
        }
    )

    if spec.expected_hz is not None and deltas.size:
        period = 1.0 / spec.expected_hz
        gap_mask = deltas > args.gap_factor * period
        estimated_drops = int(
            np.maximum(np.rint(deltas / period).astype(np.int64) - 1, 0).sum()
        )
        drop_ratio = estimated_drops / (len(timeline) + estimated_drops)
        result.update(
            {
                "gap_count": int(np.count_nonzero(gap_mask)),
                "estimated_dropped_frames": estimated_drops,
                "estimated_drop_ratio": float(drop_ratio),
            }
        )
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
    right_delta = np.abs(target[right] - reference)
    left_delta = np.abs(reference - target[left])
    return np.minimum(left_delta, right_delta).astype(np.float64) / 1e6


def synchronization_report(
    samples: Dict[str, List[Tuple[int, Optional[int]]]], warn_ms: float
) -> List[dict]:
    reports = []
    for reference_topic, target_topic in SYNC_PAIRS:
        reference_samples = samples[reference_topic]
        target_samples = samples[target_topic]
        if not reference_samples or not target_samples:
            reports.append(
                {
                    "reference": reference_topic,
                    "target": target_topic,
                    "status": "SKIP",
                    "median_ms": None,
                    "p95_ms": None,
                    "max_ms": None,
                }
            )
            continue
        reference_header = np.asarray(
            [item[1] for item in reference_samples if item[1] is not None and item[1] > 0],
            dtype=np.int64,
        )
        target_header = np.asarray(
            [item[1] for item in target_samples if item[1] is not None and item[1] > 0],
            dtype=np.int64,
        )
        offsets = nearest_offsets_ms(reference_header, target_header)
        p95_ms = percentile(offsets, 95)
        reports.append(
            {
                "reference": reference_topic,
                "target": target_topic,
                "status": "WARN" if p95_ms is not None and p95_ms > warn_ms else "PASS",
                "median_ms": percentile(offsets, 50),
                "p95_ms": p95_ms,
                "max_ms": percentile(offsets, 100),
            }
        )
    return reports


def print_report(report: dict) -> None:
    print(f"ROSbag 质量检查：{report['overall_status']}")
    print(f"Bag: {report['bag']}")
    print(
        f"时长 {report['duration_s']:.3f} s | 消息 {report['selected_message_count']} "
        f"| DB3 {report['size_gib']:.2f} GiB"
    )
    print()
    print("状态  数量       Hz/期望       最大间隔  估算丢帧     Header异常  延迟P95   话题")
    for item in report["topics"]:
        actual_hz = fmt(item.get("actual_hz"), 1)
        expected_hz = fmt(item.get("expected_hz"), 0)
        drop_count = item.get("estimated_dropped_frames")
        drop_ratio = item.get("estimated_drop_ratio")
        drops = "-" if drop_count is None else f"{drop_count}({100 * drop_ratio:.2f}%)"
        header_errors = (
            item.get("invalid_or_zero_header_stamps", 0)
            + item.get("header_regressions", 0)
        )
        latency_value = item.get("header_to_receive_abs_p95_ms")
        latency_display = (
            "-" if latency_value is None else f"{fmt(latency_value, 1)} ms"
        )
        print(
            f"{item['status']:<5} {item['count']:>8}  "
            f"{actual_hz:>6}/{expected_hz:<5}  {fmt(item.get('max_gap_ms'), 1):>8}ms  "
            f"{drops:>12}  {header_errors:>8}  {latency_display:>10}  {item['topic']}"
        )
        if "type_mismatch" in item["issues"]:
            print(
                f"      类型错误：{item['actual_type']}，期望 {item['expected_type']}"
            )

    print()
    print("相机时间同步（Header 时间戳最近邻偏差）：")
    for item in report["synchronization"]:
        def label(topic: str) -> str:
            camera = topic.strip("/").split("/")[0].removeprefix("camera_")
            stream = "depth" if "aligned_depth" in topic else "rgb"
            return f"{camera}:{stream}"

        left = label(item["reference"])
        right = label(item["target"])
        print(
            f"{item['status']:<5} {left:>11} -> {right:<11} "
            f"median={fmt(item['median_ms'])} ms "
            f"p95={fmt(item['p95_ms'])} ms max={fmt(item['max_ms'])} ms"
        )

    bad_databases = {
        name: status
        for name, status in report["database_integrity"].items()
        if status not in {"ok", "skipped"}
    }
    if bad_databases:
        print("\nSQLite 完整性错误：")
        for name, status in bad_databases.items():
            print(f"  {name}: {status}")

    if report["overall_status"] == "PASS":
        print("\n结论：关键话题、时间戳和帧间隔均通过当前阈值。")
    elif report["overall_status"] == "WARN":
        print("\n结论：bag 可读取，但存在时间间隔或相机同步警告，请查看 WARN 行。")
    else:
        print("\n结论：存在缺失话题、类型错误、时间戳错误、高丢帧或文件损坏。")


def main() -> int:
    args = parse_args()
    bag_dir = args.bag.expanduser().resolve()
    try:
        metadata, db_paths = read_metadata(bag_dir)
        types, samples, integrity = merge_samples(
            db_paths, run_quick_check=not args.no_quick_check
        )
    except Exception as exc:
        print(f"检查失败：{exc}", file=sys.stderr)
        return 2

    duration_ns = int((metadata.get("duration") or {}).get("nanoseconds", 0))
    start_ns = int(
        (metadata.get("starting_time") or {}).get("nanoseconds_since_epoch", 0)
    )
    if duration_ns <= 0:
        all_receive = [item[0] for values in samples.values() for item in values]
        if all_receive:
            start_ns = min(all_receive)
            duration_ns = max(all_receive) - start_ns

    topic_reports = [
        assess_topic(
            topic,
            spec,
            types.get(topic),
            samples[topic],
            start_ns,
            duration_ns,
            args,
        )
        for topic, spec in TOPICS.items()
    ]
    sync_reports = synchronization_report(samples, args.sync_warn_ms)

    statuses = [item["status"] for item in topic_reports]
    statuses.extend(item["status"] for item in sync_reports)
    statuses.extend(
        "FAIL" if status not in {"ok", "skipped"} else "PASS"
        for status in integrity.values()
    )
    overall = "FAIL" if "FAIL" in statuses else "WARN" if "WARN" in statuses else "PASS"

    report = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "bag": str(bag_dir),
        "overall_status": overall,
        "duration_s": duration_ns / 1e9,
        "selected_message_count": sum(len(values) for values in samples.values()),
        "size_gib": sum(path.stat().st_size for path in db_paths) / (1024**3),
        "database_integrity": integrity,
        "thresholds": {
            "gap_factor": args.gap_factor,
            "warn_drop_ratio": args.warn_drop_ratio,
            "fail_drop_ratio": args.fail_drop_ratio,
            "sync_warn_ms": args.sync_warn_ms,
            "warn_latency_ms": args.warn_latency_ms,
            "fail_latency_ms": args.fail_latency_ms,
        },
        "topics": topic_reports,
        "synchronization": sync_reports,
    }
    print_report(report)

    if args.json_path:
        json_path = args.json_path.expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"JSON 报告：{json_path}")

    return 0 if overall == "PASS" else 1 if overall == "WARN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
