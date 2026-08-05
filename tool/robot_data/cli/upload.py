"""``rdp upload`` -- push a LeRobot v3 dataset directory to a Hugging Face repo.

Uploads in bounded batches (meta, data, one video key at a time) instead of a
single huge commit, so an interrupted run can simply be restarted: files that
already exist on the Hub are skipped by the server-side dedup check.

Example:
    export HF_TOKEN=hf_xx            # or: hf auth login
    rdp upload --local-dir /media/kewei/DATA-S2/tea_2_lerobot \
        --repo-id yyyyywv/egocentric --path-in-repo gripper
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HELP = "把 LeRobot v3 数据集分批上传到 Hugging Face"

# Never upload these, whatever the batch.
GLOBAL_IGNORE = ["**/.DS_Store", "**/.ipynb_checkpoints/**", "**/__pycache__/**", "**/*.tmp"]


def human(nbytes: int) -> str:
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def build_batches(local_dir: Path, max_files: int, max_bytes: int) -> list[dict]:
    """Group files under local_dir into upload batches.

    Grouping is by top-level layout (meta/, data/, videos/<video_key>/), then
    split further so no batch exceeds max_files or max_bytes.
    """
    groups: dict[str, list[Path]] = {}
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir)
        parts = rel.parts
        if parts[0] == "videos" and len(parts) > 2:
            key = f"videos/{parts[1]}"
        elif parts[0] == "images" and len(parts) > 2:
            key = f"images/{parts[1]}"
        else:
            key = parts[0] if len(parts) > 1 else "."
        groups.setdefault(key, []).append(path)

    batches: list[dict] = []
    for key, files in groups.items():
        if not files:
            continue
        current: list[Path] = []
        current_bytes = 0
        for path in files:
            size = path.stat().st_size
            over = current and (len(current) >= max_files or current_bytes + size > max_bytes)
            if over:
                batches.append({"key": key, "files": current, "bytes": current_bytes})
                current, current_bytes = [], 0
            current.append(path)
            current_bytes += size
        batches.append({"key": key, "files": current, "bytes": current_bytes})

    # Metadata last: the dataset only looks "complete" once meta/ lands.
    batches.sort(key=lambda b: (b["key"] == "meta", b["key"]))
    return batches


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--local-dir", required=True, type=Path, help="Local LeRobot dataset root")
    parser.add_argument("--repo-id", required=True, help="Target repo, e.g. yyyyywv/egocentric")
    parser.add_argument("--path-in-repo", default="", help="Subfolder inside the repo, e.g. gripper")
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model", "space"])
    parser.add_argument("--revision", default="main")
    parser.add_argument("--private", action="store_true", help="Create the repo private if it does not exist")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Defaults to $HF_TOKEN or cached login")
    parser.add_argument("--max-files-per-commit", type=int, default=16)
    parser.add_argument("--max-gb-per-commit", type=float, default=8.0)
    parser.add_argument("--skip-empty-dirs", action="store_true", default=True)
    parser.add_argument("--only", action="append", default=None,
                        help="Only upload batches whose group matches this prefix (repeatable), "
                             "e.g. --only data --only videos/observation.images.top")
    parser.add_argument("--dry-run", action="store_true", help="List what would be uploaded and exit")


def run(args: argparse.Namespace) -> int:
    # Imported here so that `rdp --help` works without huggingface_hub installed;
    # it is a dependency of publishing, not of converting.
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    local_dir = args.local_dir.expanduser().resolve()
    if not local_dir.is_dir():
        print(f"error: {local_dir} is not a directory", file=sys.stderr)
        return 1
    if not (local_dir / "meta" / "info.json").is_file():
        print(f"warning: {local_dir}/meta/info.json not found — is this a LeRobot dataset root?", file=sys.stderr)

    prefix = args.path_in_repo.strip("/")
    max_bytes = int(args.max_gb_per_commit * 1024**3)
    batches = build_batches(local_dir, args.max_files_per_commit, max_bytes)
    if args.only:
        batches = [b for b in batches if any(b["key"].startswith(p.strip("/")) for p in args.only)]
    if not batches:
        print("nothing to upload")
        return 0

    total_files = sum(len(b["files"]) for b in batches)
    total_bytes = sum(b["bytes"] for b in batches)
    dest = f"{args.repo_id}/{prefix}" if prefix else args.repo_id
    print(f"source : {local_dir}")
    print(f"target : https://huggingface.co/datasets/{dest}  ({args.repo_type}, rev {args.revision})")
    print(f"payload: {total_files} files, {human(total_bytes)} in {len(batches)} commits\n")
    for i, batch in enumerate(batches, 1):
        print(f"  [{i:>2}/{len(batches)}] {batch['key']:<40} {len(batch['files']):>3} files  {human(batch['bytes']):>10}")
    print()

    if args.dry_run:
        print("dry run — nothing uploaded")
        return 0

    api = HfApi(token=args.token)
    try:
        who = api.whoami()
        print(f"authenticated as {who.get('name')}")
    except Exception as exc:  # noqa: BLE001 - surface any auth failure the same way
        print(f"error: not authenticated ({exc}). Run `hf auth login` or set HF_TOKEN.", file=sys.stderr)
        return 1

    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)

    done_bytes = 0
    for i, batch in enumerate(batches, 1):
        key = batch["key"]
        rels = [str(p.relative_to(local_dir)) for p in batch["files"]]
        print(f"\n[{i}/{len(batches)}] uploading {key}: {len(rels)} files, {human(batch['bytes'])}")
        try:
            api.upload_folder(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                revision=args.revision,
                folder_path=str(local_dir),
                path_in_repo=prefix or None,
                allow_patterns=rels,
                ignore_patterns=GLOBAL_IGNORE,
                commit_message=f"Upload {prefix + '/' if prefix else ''}{key} ({i}/{len(batches)})",
            )
        except HfHubHTTPError as exc:
            print(f"error: batch {key} failed: {exc}", file=sys.stderr)
            print("Re-run the same command; completed files are skipped on retry.", file=sys.stderr)
            return 1
        done_bytes += batch["bytes"]
        print(f"    ok — {human(done_bytes)} / {human(total_bytes)} ({100 * done_bytes / max(total_bytes, 1):.1f}%)")

    print(f"\ndone: https://huggingface.co/datasets/{args.repo_id}/tree/{args.revision}/{prefix}")
    return 0
