#!/usr/bin/env python3
"""Offline quality checks for aligned HDF5 episodes before LeRobot conversion.

The checker validates exactly what ``tool/hdf5_to_lerobotv3.py`` requires
(dataset layout, shapes, dtypes) and adds data sanity checks that catch broken
recordings early -- most importantly ``action`` being a copy of ``qpos``, which
makes the episode useless for behaviour cloning.

Image payloads are only sampled (a handful of frames per camera), so a 9 GiB
file is checked in seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Schema:
    """Per-file layout, taken from the attributes the converter writes.

    ``tool/conversion_common.py`` stores ``state_dim``, ``state_names_json`` and
    the full ``profile_json`` on every file it produces, so the widths are read
    from the file rather than assumed.  Older or hand-made files without those
    attributes fall back to the observed ``action`` width.
    """

    state_dim: int
    arm_dim: Optional[int] = None
    state_names: Tuple[str, ...] = ()
    profile_name: Optional[str] = None
    # (name, kind, start, end) for each end effector, in state-vector order.
    end_effectors: List[Tuple[str, str, int, int]] = field(default_factory=list)
    source: str = "action shape"

    @property
    def diagnostic_dim(self) -> int:
        """Columns used for step/lag diagnostics: the arm when it is known."""
        return self.arm_dim if self.arm_dim else self.state_dim


def read_schema(source: Any, declared_dim: int) -> Schema:
    """Build a Schema from file attributes, falling back to the action width."""
    raw_profile = source.attrs.get("profile_json")
    attr_dim = source.attrs.get("state_dim")
    state_dim = int(attr_dim) if attr_dim is not None else declared_dim

    names: Tuple[str, ...] = ()
    raw_names = source.attrs.get("state_names_json")
    if raw_names is not None:
        try:
            parsed = json.loads(raw_names)
            if isinstance(parsed, list):
                names = tuple(str(item) for item in parsed)
        except (ValueError, TypeError):
            names = ()

    if raw_profile is None:
        return Schema(state_dim=state_dim, state_names=names,
                      source="state_dim attr" if attr_dim is not None else "action shape")

    try:
        profile = json.loads(raw_profile)
    except (ValueError, TypeError):
        return Schema(state_dim=state_dim, state_names=names, source="state_dim attr")

    arm_dim = len(profile.get("arm", {}).get("joint_names") or ()) or None
    effectors: List[Tuple[str, str, int, int]] = []
    offset = arm_dim or 0
    for entry in profile.get("end_effectors") or ():
        width = int(entry.get("dim", 0))
        if width <= 0:
            continue
        effectors.append((str(entry.get("name", "?")), str(entry.get("kind", "?")),
                          offset, offset + width))
        offset += width
    return Schema(
        state_dim=state_dim,
        arm_dim=arm_dim,
        state_names=names,
        profile_name=profile.get("name"),
        end_effectors=effectors,
        source="profile_json",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="HDF5 文件或包含 HDF5 的目录")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pattern", default="*.hdf5", help="目录模式下的文件通配符")
    parser.add_argument("--limit", type=int, default=0, help="最多检查前 N 个文件，0 表示不限制")
    parser.add_argument("--include-velocity", action=argparse.BooleanOptionalAction, default=True,
                        help="要求存在 observations/qvel（与转换脚本默认一致）")
    parser.add_argument("--fps", type=int, default=30,
                        help="用于把关节跳变换算成每秒角速度；文件带 fps 属性时以文件为准")
    parser.add_argument("--sample-frames", type=int, default=8, help="每个相机抽样检查的帧数，0 表示跳过图像内容检查")
    parser.add_argument("--identical-warn-ratio", type=float, default=0.9,
                        help="action 与 qpos 完全相同的行占比超过该值时 WARN")
    parser.add_argument("--max-step-rad", type=float, default=0.35,
                        help="相邻帧关节位置跳变超过该值（rad）时 WARN")
    parser.add_argument("--static-warn-ratio", type=float, default=0.98,
                        help="qpos 相邻帧完全不变的行占比超过该值时 WARN")
    parser.add_argument("--min-frames", type=int, default=30, help="少于该帧数的 episode 判为 WARN")
    parser.add_argument("--max-lag", type=int, default=5, help="action 相对 qpos 的滞后/超前搜索范围（帧）")
    parser.add_argument("--json-path", type=Path, default=None, help="把完整报告写入 JSON 文件")
    parser.add_argument("--quiet", action="store_true", help="只打印每个文件一行摘要")
    parser.add_argument("--layout", action="store_true",
                        help="打印文件内所有数据集（对应话题）的形状、dtype、chunk、压缩与占用空间")
    parser.add_argument("--layout-only", action="store_true",
                        help="只打印布局，跳过全部质量检查（最快）")
    return parser.parse_args()


def collect_files(root: Path, pattern: str, recursive: bool, limit: int) -> List[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"输入不存在：{root}")
    files = sorted(root.rglob(pattern) if recursive else root.glob(pattern))
    if limit > 0:
        files = files[:limit]
    return files


def fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def sample_indices(frames: int, count: int) -> np.ndarray:
    if count <= 0 or frames <= 0:
        return np.empty(0, dtype=int)
    count = min(count, frames)
    return np.unique(np.linspace(0, frames - 1, count).astype(int))


def check_arrays(report: Dict[str, Any], source: Any, args: argparse.Namespace) -> None:
    """Shape/dtype/value checks on action, qpos and qvel."""
    issues: List[str] = report["issues"]

    required = ["action", "observations/qpos", "observations/images"]
    if args.include_velocity:
        required.append("observations/qvel")
    missing = [name for name in required if name not in source]
    if missing:
        issues.append(f"FAIL 缺少数据集 {missing}")
        return

    action_ds = source["action"]
    qpos_ds = source["observations/qpos"]
    frames = int(action_ds.shape[0])
    report["frames"] = frames

    if frames < 1 or action_ds.ndim != 2:
        issues.append(f"FAIL action 形状应为 (T,D)，实际 {action_ds.shape}")
        return

    # The expected width comes from the file's own attributes, so a 16-dim
    # gripper episode and a 54-dim dexhand episode are both checked correctly.
    schema = read_schema(source, int(action_ds.shape[1]))
    state_dim = schema.state_dim
    report["schema"] = {
        "state_dim": state_dim,
        "arm_dim": schema.arm_dim,
        "profile": schema.profile_name,
        "source": schema.source,
    }

    if action_ds.shape != (frames, state_dim):
        issues.append(
            f"FAIL action 形状应为 (T,{state_dim})（来自 {schema.source}），实际 {action_ds.shape}"
        )
        return
    if qpos_ds.shape != (frames, state_dim):
        issues.append(f"FAIL qpos 形状应为 (T,{state_dim})，实际 {qpos_ds.shape}")
        return
    if args.include_velocity and source["observations/qvel"].shape != (frames, state_dim):
        issues.append(f"FAIL qvel 形状应为 (T,{state_dim})，实际 {source['observations/qvel'].shape}")
        return
    if schema.state_names and len(schema.state_names) != state_dim:
        issues.append(
            f"FAIL state_names_json 有 {len(schema.state_names)} 个名称，但 state_dim 为 {state_dim}"
        )
    if frames < args.min_frames:
        issues.append(f"WARN 帧数过少：{frames} < {args.min_frames}")

    action = np.asarray(action_ds[:], dtype=np.float64)
    qpos = np.asarray(qpos_ds[:], dtype=np.float64)
    # Prefer the rate the file was written at over the CLI default.
    fps = args.fps
    try:
        fps = int(source.attrs["fps"])
    except (KeyError, TypeError, ValueError):
        pass
    report["fps"] = fps
    report["duration_s"] = frames / fps if fps > 0 else None

    for name, array, dataset in (("action", action, action_ds), ("qpos", qpos, qpos_ds)):
        if dataset.dtype != np.float32:
            issues.append(f"WARN {name} dtype 为 {dataset.dtype}，转换脚本期望 float32")
        bad = int(np.count_nonzero(~np.isfinite(array)))
        if bad:
            issues.append(f"FAIL {name} 含 {bad} 个 NaN/Inf")

    # --- the decisive check: action must not be a copy of the observed state ---
    identical_rows = int(np.count_nonzero(np.all(action == qpos, axis=1)))
    identical_ratio = identical_rows / frames
    max_abs_diff = float(np.abs(action - qpos).max())
    report["action_vs_qpos"] = {
        "identical_rows": identical_rows,
        "identical_ratio": identical_ratio,
        "max_abs_diff": max_abs_diff,
        "per_dim_max_abs_diff": np.abs(action - qpos).max(axis=0).round(6).tolist(),
    }
    if identical_rows == frames:
        issues.append("FAIL action 与 qpos 完全相同，转换脚本会拒绝该文件（数据无监督信号）")
    elif identical_ratio > args.identical_warn_ratio:
        issues.append(f"WARN action 有 {100 * identical_ratio:.1f}% 的行与 qpos 完全相同")

    # Diagnostic: a real command leads the measured state, so the best match
    # should be at a positive lag, not at lag 0.  Restricted to the arm columns
    # when the profile identifies them, since end effectors move on their own
    # timescale (and an unmeasured one is a command echo by construction).
    arm_cols = schema.diagnostic_dim
    lags: Dict[int, float] = {}
    for lag in range(0, max(args.max_lag, 0) + 1):
        if frames - lag < 2:
            break
        lags[lag] = float(np.abs(action[: frames - lag, :arm_cols] - qpos[lag:, :arm_cols]).mean())
    if lags:
        best_lag = min(lags, key=lambda key: lags[key])
        report["best_action_lag"] = {"lag_frames": best_lag, "mean_abs_diff": lags[best_lag],
                                     "by_lag": {str(k): v for k, v in lags.items()}}
        if best_lag == 0 and lags[0] < 1e-6 and identical_rows != frames:
            issues.append("WARN action 与同帧 qpos 几乎一致（lag=0），疑似用状态回填了指令")

    def label(index: int) -> str:
        if schema.state_names and index < len(schema.state_names):
            return f"{index}:{schema.state_names[index]}"
        return str(index)

    # Constant (dead) columns usually mean an unplugged joint or end effector.
    for name, array in (("action", action), ("qpos", qpos)):
        dead = [int(i) for i in np.where(array.max(axis=0) == array.min(axis=0))[0]]
        if dead:
            shown = [label(i) for i in dead[:12]]
            more = f" 等 {len(dead)} 维" if len(dead) > 12 else ""
            issues.append(f"WARN {name} 的常量维度 {shown}{more}（全程无变化）")
        report.setdefault("dead_dims", {})[name] = dead

    # Discontinuities and frozen state.
    steps = np.abs(np.diff(qpos[:, :arm_cols], axis=0)) if frames > 1 else np.zeros((0, arm_cols))
    max_step = float(steps.max()) if steps.size else 0.0
    report["qpos_max_step_rad"] = max_step
    report["qpos_max_step_rad_per_s"] = max_step * fps if fps > 0 else None
    if max_step > args.max_step_rad:
        dim = int(np.unravel_index(np.argmax(steps), steps.shape)[1])
        frame = int(np.unravel_index(np.argmax(steps), steps.shape)[0])
        issues.append(
            f"WARN qpos 最大帧间跳变 {max_step:.3f} rad（维度 {label(dim)}，帧 {frame}）超过阈值"
        )

    if frames > 1:
        frozen = int(np.count_nonzero(np.all(np.diff(qpos, axis=0) == 0, axis=1)))
        frozen_ratio = frozen / (frames - 1)
        report["qpos_frozen_ratio"] = frozen_ratio
        if frozen_ratio > args.static_warn_ratio:
            issues.append(f"WARN qpos 有 {100 * frozen_ratio:.1f}% 的相邻帧完全相同，疑似状态未更新")

    # Per end effector, not a fixed qpos[:, 14:16] gripper slice: a dexhand
    # occupies 20 columns and a gripper 1, both taken from the profile.
    ranges: Dict[str, Any] = {}
    for name, kind, start, end in schema.end_effectors:
        if end > state_dim:
            continue
        block = qpos[:, start:end]
        ranges[name] = {
            "kind": kind,
            "dims": [start, end],
            "min": float(block.min()),
            "max": float(block.max()),
            "moving_dims": int(np.count_nonzero(block.max(axis=0) != block.min(axis=0))),
        }
        if ranges[name]["moving_dims"] == 0:
            issues.append(f"WARN 末端执行器 {name}（{kind}，{end - start} 维）全程无变化")
    if ranges:
        report["end_effector_range"] = ranges


def check_images(report: Dict[str, Any], source: Any, args: argparse.Namespace) -> None:
    issues: List[str] = report["issues"]
    frames = report.get("frames")
    if frames is None or "observations/images" not in source:
        return

    cameras: Dict[str, Any] = {}
    group = source["observations/images"]
    if not len(group):
        issues.append("FAIL observations/images 为空，没有任何相机")
        return

    for name, dataset in group.items():
        entry: Dict[str, Any] = {"shape": list(dataset.shape), "dtype": str(dataset.dtype)}
        if dataset.ndim != 4 or dataset.shape[0] != frames or dataset.shape[-1] != 3:
            issues.append(f"FAIL 相机 {name} 形状非法：{dataset.shape}（应为 (T,H,W,3) 且 T={frames}）")
            cameras[name] = entry
            continue
        if dataset.dtype != np.uint8:
            issues.append(f"FAIL 相机 {name} dtype 为 {dataset.dtype}，应为 uint8")

        indices = sample_indices(frames, args.sample_frames)
        if indices.size:
            sampled = np.stack([np.asarray(dataset[int(i)]) for i in indices])
            means = sampled.reshape(sampled.shape[0], -1).mean(axis=1)
            stds = sampled.reshape(sampled.shape[0], -1).std(axis=1)
            entry["sampled_frames"] = indices.tolist()
            entry["sample_mean"] = float(means.mean())
            entry["sample_min_std"] = float(stds.min())
            blank = int(np.count_nonzero(stds < 1.0))
            if blank:
                issues.append(f"WARN 相机 {name} 有 {blank}/{len(indices)} 个抽样帧近似纯色（可能黑屏或掉流）")
            if sampled.shape[0] > 1:
                duplicates = int(sum(1 for i in range(1, sampled.shape[0])
                                     if np.array_equal(sampled[i], sampled[i - 1])))
                entry["duplicate_sampled_frames"] = duplicates
                if duplicates:
                    issues.append(f"WARN 相机 {name} 有 {duplicates} 对相邻抽样帧完全相同（可能画面冻结）")
        cameras[name] = entry

    report["cameras"] = cameras


def collect_layout(source: Any) -> List[Dict[str, Any]]:
    """Every dataset in the file with its on-disk storage properties."""
    import h5py

    entries: List[Dict[str, Any]] = []

    def visit(name: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        filters = []
        if obj.compression:
            filters.append(f"{obj.compression}{obj.compression_opts if obj.compression_opts is not None else ''}")
        if obj.shuffle:
            filters.append("shuffle")
        if obj.scaleoffset is not None:
            filters.append(f"scaleoffset{obj.scaleoffset}")
        entries.append({
            "path": name,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "chunks": list(obj.chunks) if obj.chunks else None,
            "layout": "chunked" if obj.chunks else ("compact" if obj.nbytes < 64 * 1024 else "contiguous"),
            "filters": filters,
            "nbytes": int(obj.nbytes),
            "storage_bytes": int(obj.id.get_storage_size()),
            "attrs": {key: str(value) for key, value in obj.attrs.items()},
        })

    source.visititems(visit)
    entries.sort(key=lambda entry: entry["path"])
    return entries


def print_layout(entries: Sequence[Dict[str, Any]]) -> None:
    def human(size: float) -> str:
        for unit in ("B", "KiB", "MiB", "GiB"):
            if size < 1024 or unit == "GiB":
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}GiB"

    print(f"       {'数据集':<34}{'形状':<24}{'dtype':<9}{'chunk':<20}{'压缩':<12}{'占用':>9}  {'压缩比':>6}")
    for entry in entries:
        shape = "x".join(str(v) for v in entry["shape"])
        chunk = "x".join(str(v) for v in entry["chunks"]) if entry["chunks"] else entry["layout"]
        filters = ",".join(entry["filters"]) or "-"
        stored = entry["storage_bytes"]
        ratio = f"{entry['nbytes'] / stored:.2f}x" if stored else "-"
        print(f"       {entry['path']:<34}{shape:<24}{entry['dtype']:<9}{chunk:<20}{filters:<12}"
              f"{human(stored):>9}  {ratio:>6}")
        if entry["attrs"]:
            print(f"         attrs: {entry['attrs']}")


def check_file(path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    import h5py

    report: Dict[str, Any] = {
        "file": str(path),
        "size_gib": path.stat().st_size / (1024**3),
        "issues": [],
    }
    try:
        with h5py.File(path, "r") as source:
            report["attrs"] = {key: str(value) for key, value in source.attrs.items()}
            if args.layout or args.layout_only:
                report["layout"] = collect_layout(source)
            if args.layout_only:
                action = next((e for e in report["layout"] if e["path"] == "action"), None)
                if action and action["shape"]:
                    report["frames"] = int(action["shape"][0])
                    try:
                        fps = int(source.attrs["fps"])
                    except (KeyError, TypeError, ValueError):
                        fps = args.fps
                    report["duration_s"] = report["frames"] / fps if fps > 0 else None
                return {**report, "status": "INFO"}
            check_arrays(report, source, args)
            if args.sample_frames > 0:
                check_images(report, source, args)
    except Exception as exc:  # unreadable / truncated file
        report["issues"].append(f"FAIL 无法读取：{exc}")

    issues = report["issues"]
    report["status"] = "FAIL" if any(i.startswith("FAIL") for i in issues) else "WARN" if issues else "PASS"
    return report


def print_file_report(report: Dict[str, Any], quiet: bool) -> None:
    name = Path(report["file"]).name
    frames = report.get("frames")
    duration = report.get("duration_s")
    head = (f"{report['status']:<4} {name}  帧数={frames if frames is not None else '-'}"
            f"  时长={fmt(duration, 1)}s  大小={report['size_gib']:.2f} GiB")
    print(head)
    schema = report.get("schema")
    if schema:
        print(f"       维度：state_dim={schema['state_dim']} arm_dim={schema['arm_dim'] or '未知'}"
              f" profile={schema['profile'] or '未知'}（来自 {schema['source']}）")
    if report.get("attrs"):
        print(f"       文件属性：{report['attrs']}")
    if report.get("layout"):
        print_layout(report["layout"])
    if quiet:
        for issue in report["issues"]:
            print(f"       {issue}")
        return

    compare = report.get("action_vs_qpos")
    if compare:
        print(f"       action vs qpos: 相同行 {compare['identical_rows']}"
              f"（{100 * compare['identical_ratio']:.1f}%）最大差值 {compare['max_abs_diff']:.6f}")
    lag = report.get("best_action_lag")
    if lag:
        by_lag = "  ".join(f"lag{k}={v:.4f}" for k, v in lag["by_lag"].items())
        print(f"       最佳滞后 = {lag['lag_frames']} 帧（{by_lag}）")
    if report.get("qpos_max_step_rad") is not None:
        print(f"       qpos 最大帧间跳变 {fmt(report['qpos_max_step_rad'])} rad"
              f" | 冻结帧占比 {fmt(100 * report.get('qpos_frozen_ratio', 0.0), 1)}%")
    for name, entry in (report.get("end_effector_range") or {}).items():
        start, end = entry["dims"]
        print(f"       末端执行器 {name}（{entry['kind']}，维度 {start}-{end - 1}）"
              f" 取值 [{entry['min']:.3f},{entry['max']:.3f}]"
              f" 活动维度 {entry['moving_dims']}/{end - start}")
    for camera, entry in (report.get("cameras") or {}).items():
        shape = "x".join(str(v) for v in entry["shape"])
        extra = ""
        if "sample_mean" in entry:
            extra = f" 抽样均值={entry['sample_mean']:.1f} 最小方差={entry['sample_min_std']:.1f}"
        print(f"       相机 {camera}: {shape} {entry['dtype']}{extra}")
    for issue in report["issues"]:
        print(f"       {issue}")


def write_json(args: argparse.Namespace, reports: Sequence[Dict[str, Any]], overall: str,
               counts: Dict[str, int]) -> None:
    if not args.json_path:
        return
    json_path = args.json_path.expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "input": str(args.input),
        "overall_status": overall,
        "counts": counts,
        "thresholds": {
            "identical_warn_ratio": args.identical_warn_ratio,
            "max_step_rad": args.max_step_rad,
            "static_warn_ratio": args.static_warn_ratio,
            "min_frames": args.min_frames,
            "sample_frames": args.sample_frames,
            "fps": args.fps,
        },
        "files": list(reports),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"JSON 报告：{json_path}")


def main() -> int:
    args = parse_args()
    try:
        import h5py  # noqa: F401
    except ImportError:
        print("缺少依赖：pip install h5py", file=sys.stderr)
        return 2

    try:
        files = collect_files(args.input.expanduser().resolve(), args.pattern, args.recursive, args.limit)
    except Exception as exc:
        print(f"检查失败：{exc}", file=sys.stderr)
        return 2
    if not files:
        print(f"未找到匹配 {args.pattern} 的 HDF5 文件", file=sys.stderr)
        return 2

    reports: List[Dict[str, Any]] = []
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] ", end="")
        report = check_file(path, args)
        reports.append(report)
        print_file_report(report, args.quiet)

    counts = {status: sum(1 for r in reports if r["status"] == status) for status in ("PASS", "WARN", "FAIL")}
    overall = "FAIL" if counts["FAIL"] else "WARN" if counts["WARN"] else "PASS"

    if args.layout_only:
        print(f"\n共列出 {len(reports)} 个文件的布局（未执行质量检查）。")
        write_json(args, reports, "INFO", counts)
        return 0

    print()
    print(f"总计 {len(reports)} 个文件：PASS {counts['PASS']} | WARN {counts['WARN']} | FAIL {counts['FAIL']}")
    if counts["FAIL"]:
        print("FAIL 文件：")
        for report in reports:
            if report["status"] == "FAIL":
                print(f"  {Path(report['file']).name}: {'; '.join(report['issues'])}")
    if overall == "PASS":
        print("结论：全部文件满足 hdf5_to_lerobotv3.py 的输入要求。")
    elif overall == "WARN":
        print("结论：可以转换，但存在数据质量警告，请逐条确认 WARN 行。")
    else:
        print("结论：存在会被转换脚本拒绝的文件，需要在生成 HDF5 的上游修复。")

    write_json(args, reports, overall, counts)

    return 0 if overall == "PASS" else 1 if overall == "WARN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
