#!/usr/bin/env python3
"""列出一个目录下所有 LeRobot 数据集的 repo_id / total_episodes / rgb_encoder。

数据集识别方式：任何包含 meta/info.json 的目录（会递归查找，所以
package_head_lerobot/batch_1 这类嵌套批次也能各自列出）。

repo_id 与 rgb_encoder 优先取 meta/conversion_manifest.json；没有 manifest 的
数据集（例如 LeRobot 自己录制的）则回退到 info.json 里视频特征的编码参数，
repo_id 回退为相对路径。

用法:
    tool/scripts/list_datasets.py /mnt/robot_platform/datasets
    tool/scripts/list_datasets.py /mnt/robot_platform/datasets --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def find_datasets(root: Path) -> list[Path]:
    """返回所有含 meta/info.json 的数据集目录，按路径排序。"""
    if (root / "meta" / "info.json").is_file():
        return [root]
    return sorted({p.parent.parent for p in root.rglob("meta/info.json")})


def load_json(path: Path):
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] 无法读取 {path}: {exc}", file=sys.stderr)
        return None


def encoder_from_info(info: dict) -> dict | None:
    """没有 manifest 时，从 info.json 的第一个 video 特征还原编码参数。"""
    for key, feat in (info.get("features") or {}).items():
        if feat.get("dtype") != "video":
            continue
        vi = feat.get("info") or {}
        return {
            "codec": vi.get("video.codec"),
            "pixel_format": vi.get("video.pix_fmt"),
            "crf": vi.get("video.crf"),
            "gop": vi.get("video.g"),
            "preset": vi.get("video.preset"),
            "fast_decode": vi.get("video.fast_decode"),
            "_source": f"info.json:{key}",
        }
    return None


def fmt_encoder(enc: dict | None) -> str:
    if not enc:
        return "-"
    parts = [str(enc.get("codec") or "?")]
    for label, key in (("crf", "crf"), ("gop", "gop"), ("preset", "preset"),
                       ("pix", "pixel_format")):
        val = enc.get(key)
        if val is None:
            continue
        if isinstance(val, float) and val.is_integer():
            val = int(val)
        parts.append(f"{label}={val}")
    return " ".join(parts)


def collect(root: Path) -> list[dict]:
    rows = []
    for ds in find_datasets(root):
        info = load_json(ds / "meta" / "info.json")
        if info is None:
            continue
        manifest_path = ds / "meta" / "conversion_manifest.json"
        manifest = load_json(manifest_path) if manifest_path.is_file() else None

        rows.append({
            "path": str(ds),
            "name": str(ds.relative_to(root)) if ds != root else ds.name,
            "repo_id": (manifest or {}).get("repo_id")
                       or info.get("repo_id")
                       or (str(ds.relative_to(root)) if ds != root else ds.name),
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "fps": info.get("fps"),
            "robot_type": info.get("robot_type"),
            "rgb_encoder": (manifest or {}).get("rgb_encoder") or encoder_from_info(info),
            "has_manifest": manifest is not None,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="数据集根目录（会递归查找）")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="输出完整 JSON，便于脚本消费")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"错误: 目录不存在: {root}", file=sys.stderr)
        return 2

    rows = collect(root)
    if not rows:
        print(f"在 {root} 下没有找到 LeRobot 数据集（meta/info.json）", file=sys.stderr)
        return 1

    if args.as_json:
        json.dump(rows, sys.stdout, indent=2, ensure_ascii=False)
        print()
        return 0

    w_repo = max(len("repo_id"), max(len(r["repo_id"]) for r in rows))
    w_enc = max(len("rgb_encoder"), max(len(fmt_encoder(r["rgb_encoder"])) for r in rows))
    print(f"{'repo_id':<{w_repo}}  {'episodes':>8}  {'rgb_encoder':<{w_enc}}")
    print(f"{'-' * w_repo}  {'-' * 8}  {'-' * w_enc}")
    total = 0
    for r in rows:
        eps = r["total_episodes"]
        total += eps if isinstance(eps, int) else 0
        mark = "" if r["has_manifest"] else " *"
        print(f"{r['repo_id']:<{w_repo}}  {str(eps if eps is not None else '?'):>8}  "
              f"{fmt_encoder(r['rgb_encoder']):<{w_enc}}{mark}")
    print(f"\n{len(rows)} 个数据集, 共 {total} 条 episode")
    if any(not r["has_manifest"] for r in rows):
        print("* 无 conversion_manifest.json，编码参数取自 info.json 的视频特征")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
