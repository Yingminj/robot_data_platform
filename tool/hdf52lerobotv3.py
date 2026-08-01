#!/usr/bin/env python3
"""Convert an ALOHA-style HDF5 episode to a lossless LeRobot Dataset v3.

RGB arrays can be stored either as embedded PNG images (``use_videos=False``)
or as lossless H.264 RGB videos (``libx264rgb``, RGB24, CRF 0). The converter
decodes the result and verifies every feature before publishing the output.

Expected HDF5 layout::

    /action                         (frames, action_dim)
    /observations/qpos              (frames, state_dim)
    /observations/qvel              (frames, velocity_dim)
    /observations/images/<camera>   (frames, height, width, 3), uint8 RGB

The source file has no timestamps or task text. LeRobot timestamps are therefore
generated from ``--fps`` and the task is supplied by ``--task``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("/home/kewei/YING/robot_data_platform/rosbag2_2026_07_29-17_40_29.hdf5")
SOURCE_TO_LEROBOT = {
    "action": "action",
    "observations/qpos": "observation.state",
    "observations/qvel": "observation.velocity",
}


@dataclass(frozen=True)
class SourceSchema:
    frame_count: int
    numeric_paths: tuple[str, ...]
    image_paths: tuple[str, ...]
    path_mapping: dict[str, str]
    features: dict[str, dict[str, Any]]
    datasets: dict[str, dict[str, Any]]
    file_attributes: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将 ALOHA 风格 HDF5 转换为无损 LeRobot Dataset v3；RGB 可使用内嵌 PNG "
            "或 libx264rgb 无损视频。"
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="源 HDF5 文件")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "输出目录；PNG 默认 <stem>_lerobot_v3，"
            "无损视频默认 <stem>_lerobot_v3_video"
        ),
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="写入 LeRobot 元数据的 repo_id；默认是 local/<源文件名>",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="用于生成 LeRobot timestamp 的帧率；源 HDF5 不包含时间戳（默认 30）",
    )
    parser.add_argument(
        "--task",
        default="Unspecified task",
        help="本 episode 的任务文本；源 HDF5 不包含任务文本",
    )
    parser.add_argument(
        "--robot-type",
        default="aloha",
        help="写入 meta/info.json 的 robot_type（默认 aloha）",
    )
    parser.add_argument(
        "--image-storage",
        choices=("png", "lossless-video"),
        default="png",
        help=(
            "RGB 存储方式：png=内嵌 Parquet 的无损 PNG；"
            "lossless-video=libx264rgb/CRF 0 无损 MP4（默认 png）"
        ),
    )
    parser.add_argument(
        "--no-source-file-hash",
        action="store_true",
        help="不计算整个源 HDF5 的 SHA-256（逐字段无损校验仍会执行）",
    )
    return parser.parse_args()


def load_dependencies():
    missing: list[str] = []
    modules: dict[str, Any] = {}
    for module_name in ("av", "h5py", "numpy", "pyarrow.parquet", "PIL.Image"):
        try:
            modules[module_name] = __import__(module_name, fromlist=["*"])
        except ImportError:
            missing.append(module_name.split(".")[0])

    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError:
        missing.append("lerobot[dataset]")
        LeRobotDataset = None

    if missing:
        names = ", ".join(sorted(set(missing)))
        raise RuntimeError(
            f"缺少依赖：{names}。请在目标环境执行：\n"
            "  python -m pip install 'lerobot[dataset]>=0.6,<0.7' 'h5py>=3.12,<4'"
        )

    version = importlib.metadata.version("lerobot")
    major_minor = tuple(int(part) for part in version.split(".")[:2])
    if not (0, 6) <= major_minor < (0, 7):
        raise RuntimeError(
            f"脚本按 lerobot 0.6.x 的 Dataset v3 API 编写并验证，当前版本是 {version}"
        )

    return (
        modules["av"],
        modules["h5py"],
        modules["numpy"],
        modules["pyarrow.parquet"],
        modules["PIL.Image"],
        LeRobotDataset,
        version,
    )


def json_value(value: Any, np: Any) -> Any:
    """Convert HDF5 attributes to reversible JSON-compatible values."""
    if isinstance(value, np.generic):
        return json_value(value.item(), np)
    if isinstance(value, np.ndarray):
        return [json_value(item, np) for item in value.tolist()]
    if isinstance(value, bytes):
        return {"type": "bytes", "hex": value.hex()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": type(value).__name__, "repr": repr(value)}


def dataset_description(dataset: Any, np: Any) -> dict[str, Any]:
    return {
        "shape": list(dataset.shape),
        "dtype": np.dtype(dataset.dtype).str,
        "chunks": list(dataset.chunks) if dataset.chunks is not None else None,
        "compression": dataset.compression,
        "compression_opts": json_value(dataset.compression_opts, np),
        "attributes": {key: json_value(value, np) for key, value in dataset.attrs.items()},
    }


def inspect_source(h5_file: Any, np: Any) -> SourceSchema:
    dataset_paths: list[str] = []

    def collect(name: str, obj: Any) -> None:
        if hasattr(obj, "shape") and hasattr(obj, "dtype"):
            dataset_paths.append(name)

    h5_file.visititems(collect)
    dataset_path_set = set(dataset_paths)

    required = set(SOURCE_TO_LEROBOT)
    missing = sorted(required - dataset_path_set)
    if missing:
        raise ValueError(f"HDF5 缺少必需字段：{missing}")

    image_prefix = "observations/images/"
    image_paths = sorted(path for path in dataset_paths if path.startswith(image_prefix))
    if not image_paths:
        raise ValueError("HDF5 中没有 observations/images/<camera> 图像字段")

    expected = required | set(image_paths)
    unexpected = sorted(dataset_path_set - expected)
    if unexpected:
        raise ValueError(
            "发现尚未映射的 HDF5 dataset；为避免静默丢数据，转换已停止："
            f"{unexpected}"
        )

    first = h5_file["action"]
    if len(first.shape) != 2 or first.shape[0] <= 0:
        raise ValueError(f"action 必须是非空二维数组，实际 shape={first.shape}")
    frame_count = int(first.shape[0])

    path_mapping = dict(SOURCE_TO_LEROBOT)
    features: dict[str, dict[str, Any]] = {}
    descriptions: dict[str, dict[str, Any]] = {}

    for source_path, target_key in SOURCE_TO_LEROBOT.items():
        dataset = h5_file[source_path]
        if len(dataset.shape) != 2 or dataset.shape[0] != frame_count:
            raise ValueError(
                f"{source_path} 应为 ({frame_count}, dim)，实际 shape={dataset.shape}"
            )
        if not np.issubdtype(dataset.dtype, np.number):
            raise ValueError(f"{source_path} 必须是数值类型，实际 dtype={dataset.dtype}")
        features[target_key] = {
            "dtype": np.dtype(dataset.dtype).name,
            "shape": (int(dataset.shape[1]),),
            "names": None,
        }
        descriptions[source_path] = dataset_description(dataset, np)

    used_camera_keys: set[str] = set()
    for source_path in image_paths:
        dataset = h5_file[source_path]
        if (
            len(dataset.shape) != 4
            or dataset.shape[0] != frame_count
            or dataset.shape[-1] != 3
            or dataset.dtype != np.uint8
        ):
            raise ValueError(
                f"{source_path} 必须是 ({frame_count}, H, W, 3) uint8 RGB，"
                f"实际 shape={dataset.shape}, dtype={dataset.dtype}"
            )
        camera = source_path.removeprefix(image_prefix)
        if "/" in camera:
            raise ValueError(f"暂不支持嵌套相机名：{camera}")
        camera = re.sub(r"[^A-Za-z0-9_.-]", "_", camera)
        target_key = f"observation.images.{camera}"
        if target_key in used_camera_keys:
            raise ValueError(f"相机名规范化后发生冲突：{target_key}")
        used_camera_keys.add(target_key)
        path_mapping[source_path] = target_key
        features[target_key] = {
            "dtype": "image",
            "shape": tuple(int(x) for x in dataset.shape[1:]),
            "names": ["height", "width", "channels"],
        }
        descriptions[source_path] = dataset_description(dataset, np)

    return SourceSchema(
        frame_count=frame_count,
        numeric_paths=tuple(SOURCE_TO_LEROBOT),
        image_paths=tuple(image_paths),
        path_mapping=path_mapping,
        features=features,
        datasets=descriptions,
        file_attributes={key: json_value(value, np) for key, value in h5_file.attrs.items()},
    )


def features_for_storage(schema: SourceSchema, image_storage: str) -> dict[str, dict[str, Any]]:
    features = {key: dict(feature) for key, feature in schema.features.items()}
    if image_storage == "lossless-video":
        for source_path in schema.image_paths:
            features[schema.path_mapping[source_path]]["dtype"] = "video"
    return features


def create_lossless_video_encoder() -> Any:
    """Create an exact-RGB encoder unsupported by LeRobot's stock whitelist.

    LeRobot 0.6's public encoder configuration omits ``libx264rgb`` even when
    the bundled FFmpeg provides it. Extending the process-local frozenset lets
    the official writer use the codec without modifying the installed package.
    """
    import lerobot.configs.video as video_config
    from lerobot.configs import RGBEncoderConfig

    video_config.VALID_VIDEO_CODECS = video_config.VALID_VIDEO_CODECS | {"libx264rgb"}
    return RGBEncoderConfig(
        vcodec="libx264rgb",
        pix_fmt="rgb24",
        g=2,
        crf=0,
        preset="medium",
    )


def update_array_hash(digest: Any, array: Any, np: Any) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(contiguous.view(np.uint8).tobytes())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def progress(frame_index: int, frame_count: int, started_at: float) -> None:
    interval = max(1, frame_count // 100)
    done = frame_index + 1
    if done != frame_count and done % interval != 0:
        return
    elapsed = max(time.monotonic() - started_at, 1e-9)
    rate = done / elapsed
    remaining = (frame_count - done) / rate if rate else 0.0
    print(
        f"\r转换帧 {done}/{frame_count} ({done / frame_count:6.2%}) "
        f"{rate:5.1f} frame/s，预计剩余 {remaining:6.1f}s",
        end="\n" if done == frame_count else "",
        flush=True,
    )


def convert_frames(
    source: Any,
    schema: SourceSchema,
    dataset: Any,
    task: str,
    np: Any,
) -> dict[str, str]:
    digests = {target: hashlib.sha256() for target in schema.path_mapping.values()}
    started_at = time.monotonic()

    for frame_index in range(schema.frame_count):
        frame: dict[str, Any] = {}
        for source_path in schema.numeric_paths:
            value = source[source_path][frame_index]
            target = schema.path_mapping[source_path]
            update_array_hash(digests[target], value, np)
            frame[target] = value
        for source_path in schema.image_paths:
            image = source[source_path][frame_index]
            target = schema.path_mapping[source_path]
            update_array_hash(digests[target], image, np)
            frame[target] = image
        frame["task"] = task
        dataset.add_frame(frame)
        progress(frame_index, schema.frame_count, started_at)

    dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    return {key: digest.hexdigest() for key, digest in digests.items()}


def image_array(item: Any, output_root: Path, Image: Any, np: Any) -> Any:
    if not isinstance(item, dict):
        raise ValueError(f"Parquet image 单元应为 dict，实际是 {type(item).__name__}")
    image_bytes = item.get("bytes")
    image_path = item.get("path")
    if image_bytes is not None:
        with Image.open(io.BytesIO(image_bytes)) as image:
            array = np.asarray(image).copy()
    elif image_path:
        with Image.open(output_root / image_path) as image:
            array = np.asarray(image).copy()
    else:
        raise ValueError("Parquet image 单元同时缺少 bytes 和 path")
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"解码后的 PNG 不是 uint8 RGB：shape={array.shape}, dtype={array.dtype}")
    return array


def verify_output(
    output_root: Path,
    schema: SourceSchema,
    source_hashes: dict[str, str],
    image_storage: str,
    av: Any,
    pq: Any,
    Image: Any,
    np: Any,
) -> dict[str, Any]:
    info_path = output_root / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"缺少 LeRobot 元数据：{info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if not str(info.get("codebase_version", "")).startswith("v3."):
        raise ValueError(f"输出不是 LeRobot Dataset v3：{info.get('codebase_version')}")
    if info.get("total_frames") != schema.frame_count or info.get("total_episodes") != 1:
        raise ValueError(
            "输出计数错误："
            f"frames={info.get('total_frames')}, episodes={info.get('total_episodes')}"
        )
    uses_video = image_storage == "lossless-video"
    if uses_video and info.get("video_path") is None:
        raise ValueError("无损视频输出缺少 video_path")
    if not uses_video and info.get("video_path") is not None:
        raise ValueError("PNG 输出不应包含 video_path")

    parquet_files = sorted((output_root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise ValueError("输出中没有 data/*.parquet")

    reverse_mapping = {target: source for source, target in schema.path_mapping.items()}
    output_digests = {target: hashlib.sha256() for target in reverse_mapping}
    image_targets = {schema.path_mapping[path] for path in schema.image_paths}
    parquet_targets = [
        target for target in reverse_mapping if not uses_video or target not in image_targets
    ]
    rows = 0

    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(
            batch_size=8,
            columns=parquet_targets,
        ):
            for row in batch.to_pylist():
                for target in parquet_targets:
                    source_path = reverse_mapping[target]
                    if source_path in schema.image_paths:
                        value = image_array(row[target], output_root, Image, np)
                        expected_shape = tuple(schema.datasets[source_path]["shape"][1:])
                        if value.shape != expected_shape:
                            raise ValueError(
                                f"{target} shape 不一致：{value.shape} != {expected_shape}"
                            )
                    else:
                        dtype = np.dtype(schema.datasets[source_path]["dtype"])
                        value = np.asarray(row[target], dtype=dtype)
                        expected_shape = tuple(schema.datasets[source_path]["shape"][1:])
                        if value.shape != expected_shape:
                            raise ValueError(
                                f"{target} shape 不一致：{value.shape} != {expected_shape}"
                            )
                    update_array_hash(output_digests[target], value, np)
                rows += 1

    if rows != schema.frame_count:
        raise ValueError(f"Parquet 行数不一致：{rows} != {schema.frame_count}")

    video_streams: dict[str, list[dict[str, Any]]] = {}
    if uses_video:
        for target in sorted(image_targets):
            source_path = reverse_mapping[target]
            expected_shape = tuple(schema.datasets[source_path]["shape"][1:])
            video_files = sorted((output_root / "videos" / target).rglob("*.mp4"))
            if not video_files:
                raise ValueError(f"{target} 没有生成 MP4 文件")

            decoded_frames = 0
            stream_details: list[dict[str, Any]] = []
            for video_path in video_files:
                with av.open(str(video_path), "r") as container:
                    stream = container.streams.video[0]
                    details = {
                        "path": str(video_path.relative_to(output_root)),
                        "codec": stream.codec.canonical_name,
                        "decoder": stream.codec_context.name,
                        "pixel_format": stream.pix_fmt,
                        "width": int(stream.width),
                        "height": int(stream.height),
                        "fps": str(stream.base_rate),
                    }
                    if details["codec"] != "h264" or details["pixel_format"] != "gbrp":
                        raise ValueError(
                            f"{target} 不是预期的 H.264 RGB/gbrp 无损流：{details}"
                        )
                    stream_details.append(details)
                    for decoded_frame in container.decode(stream):
                        value = decoded_frame.to_ndarray(format="rgb24")
                        if value.shape != expected_shape or value.dtype != np.uint8:
                            raise ValueError(
                                f"{target} 解码帧格式错误："
                                f"shape={value.shape}, dtype={value.dtype}"
                            )
                        update_array_hash(output_digests[target], value, np)
                        decoded_frames += 1

            if decoded_frames != schema.frame_count:
                raise ValueError(
                    f"{target} 解码帧数不一致：{decoded_frames} != {schema.frame_count}"
                )
            video_streams[target] = stream_details

    output_hashes = {key: digest.hexdigest() for key, digest in output_digests.items()}
    mismatches = {
        key: {"source": source_hashes[key], "output": output_hashes.get(key)}
        for key in source_hashes
        if output_hashes.get(key) != source_hashes[key]
    }
    if mismatches:
        raise ValueError(f"逐字段 SHA-256 无损校验失败：{mismatches}")

    return {
        "status": "passed",
        "method": "SHA-256 over decoded arrays in frame-major C order",
        "verified_frames": rows,
        "feature_sha256": output_hashes,
        "video_streams": video_streams,
    }


def safe_repo_id(source_path: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "-", source_path.stem).strip(".-")
    return f"local/{name or 'hdf5-dataset'}"


def run(args: argparse.Namespace) -> Path:
    av, h5py, np, pq, Image, LeRobotDataset, lerobot_version = load_dependencies()

    source_path = args.input.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"源 HDF5 不存在：{source_path}")
    if args.fps <= 0:
        raise ValueError(f"--fps 必须为正整数，实际是 {args.fps}")
    if not args.task.strip():
        raise ValueError("--task 不能为空")

    uses_video = args.image_storage == "lossless-video"
    default_output_suffix = "_lerobot_v3_video" if uses_video else "_lerobot_v3"
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else source_path.with_name(f"{source_path.stem}{default_output_suffix}")
    )
    if output_path.exists():
        raise FileExistsError(f"输出路径已存在，不会覆盖：{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = output_path.with_name(f".{output_path.name}.incomplete-{os.getpid()}")
    if staging_path.exists():
        raise FileExistsError(f"临时输出路径已存在：{staging_path}")

    repo_id = args.repo_id or safe_repo_id(source_path)
    storage_description = (
        "libx264rgb/CRF 0 无损 MP4" if uses_video else "无损 PNG（不使用视频）"
    )
    print(f"源文件：{source_path}")
    print(f"输出目录：{output_path}")
    print(f"LeRobot：{lerobot_version}；图像存储：{storage_description}")
    print(f"FPS：{args.fps}（源文件无 timestamp，由 frame_index/fps 生成）")

    dataset = None
    try:
        with h5py.File(source_path, "r") as source:
            schema = inspect_source(source, np)
            print(
                f"检测到 1 个 episode、{schema.frame_count} 帧、"
                f"{len(schema.image_paths)} 路 RGB 相机"
            )
            rgb_encoder = create_lossless_video_encoder() if uses_video else None
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                root=staging_path,
                fps=args.fps,
                robot_type=args.robot_type,
                features=features_for_storage(schema, args.image_storage),
                use_videos=uses_video,
                image_writer_processes=0,
                image_writer_threads=0,
                rgb_encoder=rgb_encoder,
            )
            source_hashes = convert_frames(source, schema, dataset, args.task, np)

        verification = verify_output(
            staging_path,
            schema,
            source_hashes,
            args.image_storage,
            av,
            pq,
            Image,
            np,
        )
        source_file_hash = None
        if not args.no_source_file_hash:
            print("计算源 HDF5 文件 SHA-256…", flush=True)
            source_file_hash = sha256_file(source_path)

        manifest = {
            "format": "hdf5-to-lerobot-v3-lossless",
            "converter": str(Path(__file__).resolve()),
            "lerobot_version": lerobot_version,
            "source": {
                "path": str(source_path),
                "size_bytes": source_path.stat().st_size,
                "file_sha256": source_file_hash,
                "file_attributes": schema.file_attributes,
                "datasets": schema.datasets,
            },
            "conversion": {
                "repo_id": repo_id,
                "robot_type": args.robot_type,
                "task": args.task,
                "fps": args.fps,
                "timestamp_source": "generated as frame_index / fps; source HDF5 has no timestamps",
                "episodes": 1,
                "frames": schema.frame_count,
                "path_mapping": schema.path_mapping,
                "image_storage": (
                    "H.264 RGB MP4 (libx264rgb, rgb24 input, CRF 0, GOP 2, lossless); "
                    "use_videos=True"
                    if uses_video
                    else "PNG embedded in Parquet (lossless); use_videos=False"
                ),
            },
            "verification": verification,
        }
        manifest_path = staging_path / "meta" / "conversion_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        staging_path.rename(output_path)
    except Exception:
        if dataset is not None:
            # Do not finalize a failed, partially buffered episode.
            try:
                dataset.writer.stop_image_writer()
            except Exception:
                pass
        if staging_path.exists():
            print(f"\n转换未完成；诊断用临时目录保留在：{staging_path}", file=sys.stderr)
        raise

    print(f"无损校验通过：{schema.frame_count} 帧、{len(schema.path_mapping)} 个字段")
    print(f"转换完成：{output_path}")
    return output_path


def main() -> int:
    try:
        run(parse_args())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n用户中断转换。", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
