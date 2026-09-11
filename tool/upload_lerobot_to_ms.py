#!/usr/bin/env python3
"""Upload a LeRobot v3 dataset directory to a ModelScope dataset repo.

Uploads in bounded batches (meta, data, one video key at a time) instead of a
single huge commit, so an interrupted run can simply be restarted: files that
already exist on the Hub are skipped by the server-side dedup check.

Example:
    export MODELSCOPE_API_TOKEN=ms-xxxxx   # https://modelscope.cn/my/myaccesstoken
    python tool/upload_lerobot_to_ms.py \
        --local-dir /media/kewei/DATA-S2/tea_2_lerobot \
        --repo-id yingminj/lerobot_test_yingminj \
        --path-in-repo gripper
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import modelscope.hub.api as ms_hub_api
from modelscope.hub.api import HubApi

# Never upload these, whatever the batch.
GLOBAL_IGNORE = ["**/.DS_Store", "**/.ipynb_checkpoints/**", "**/__pycache__/**", "**/*.tmp"]

# ModelScope caps a repo at 200 commits per rolling hour. upload_folder() splits
# its input into internal "commit batches" and issues one commit per batch, so a
# 16-file call can silently become 16 commits. Force: 1 upload_folder call == 1
# commit, with the batch size we chose ourselves.
RATE_LIMIT_RE = re.compile(r"try again after\s+(\d+)\s+seconds", re.IGNORECASE)


def force_one_commit_per_call(files_per_commit: int) -> None:
    """Neutralise ModelScope's hidden commit amplification.

    - UPLOAD_ADAPTIVE_BATCH_SIZE slices a small batch into total//10 pieces
      (16 files -> 16 commits), which alone can burn the hourly quota.
    - UPLOAD_REACT_ENABLED re-commits failed files in parallel "rounds" behind
      our back; every one of those rounds is another commit. Our own retry loop
      already handles failures, so keep the SDK out of it.

    Net effect: one upload_folder() call == one commit.
    """
    ms_hub_api.UPLOAD_ADAPTIVE_BATCH_SIZE = False
    ms_hub_api.UPLOAD_COMMIT_BATCH_SIZE = max(1, files_per_commit)
    ms_hub_api.UPLOAD_REACT_ENABLED = False


class CommitLimiter:
    """Client-side token bucket for ModelScope's ~200 commits/hour per repo.

    Counting our own batches is not enough: the SDK also commits internally
    (batch commits, retry rounds). So we wrap HubApi.create_commit and count
    every request that actually leaves the process.
    """

    def __init__(self, per_hour: int = 180, window: float = 3600.0):
        self.per_hour = max(1, per_hour)
        self.window = window
        self.calls: list[float] = []
        self.block_until = 0.0

    def penalize(self, seconds: float) -> None:
        """Honour a server-sent cooldown ('try again after Ns')."""
        self.block_until = max(self.block_until, time.time() + seconds)

    def acquire(self) -> None:
        wait = self.block_until - time.time()
        if wait > 0:
            self._sleep(wait)
            self.block_until = 0.0

        while True:
            cutoff = time.time() - self.window
            self.calls = [t for t in self.calls if t > cutoff]
            if len(self.calls) < self.per_hour:
                break
            self._sleep(self.calls[0] - cutoff + 1)
        self.calls.append(time.time())

    @staticmethod
    def _sleep(seconds: float) -> None:
        seconds = max(0.0, seconds)
        if seconds < 1:
            return
        print(f"    commit quota — sleeping {seconds:.0f}s "
              f"(resume ~{time.strftime('%H:%M:%S', time.localtime(time.time() + seconds))})", flush=True)
        time.sleep(seconds)

    def attach(self, api: HubApi) -> None:
        original = api.create_commit

        def counted_create_commit(*args, **kwargs):
            self.acquire()
            return original(*args, **kwargs)

        api.create_commit = counted_create_commit  # instance attr shadows the class method


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
        if any(part.startswith(".") for part in rel.parts):
            continue  # dotfiles, incl. ModelScope's own .ms_upload_cache
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


def run_batch(api: HubApi, *, limiter: CommitLimiter, max_wait: int, attempts: int, **kwargs) -> None:
    """Commit one batch, honouring ModelScope's 429 cooldown instead of dying."""
    for attempt in range(1, attempts + 1):
        try:
            api.upload_folder(**kwargs)
            return
        except Exception as exc:  # noqa: BLE001 - any failure gets the same treatment
            text = str(exc)
            rate_limited = "429" in text or "frequency limit" in text.lower()
            if not rate_limited or attempt == attempts:
                raise
            print(f"    commit rejected (attempt {attempt}/{attempts}): {text}", flush=True)
            match = RATE_LIMIT_RE.search(text)
            if match:
                limiter.penalize(min(int(match.group(1)) + 5, max_wait))
            else:
                limiter.penalize(min(30 * 2 ** (attempt - 1), max_wait))
    raise RuntimeError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local-dir", required=True, type=Path, help="Local LeRobot dataset root")
    parser.add_argument("--repo-id", required=True, help="Target repo, e.g. yyyyywv/egocentric")
    parser.add_argument("--path-in-repo", default="", help="Subfolder inside the repo, e.g. gripper")
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    parser.add_argument("--private", action="store_true", help="Create the repo private if it does not exist")
    parser.add_argument("--token", default=os.environ.get("MODELSCOPE_API_TOKEN"),
                        help="Defaults to $MODELSCOPE_API_TOKEN or cached login")
    parser.add_argument("--max-files-per-commit", type=int, default=512,
                        help="Files per commit; also the number of commits is ~files/this "
                             "(ModelScope allows ~200 commits/hour per repo)")
    parser.add_argument("--max-gb-per-commit", type=float, default=8.0)
    parser.add_argument("--max-rate-limit-wait", type=int, default=3900,
                        help="Cap on how long to sleep when the server says 'try again after Ns'")
    parser.add_argument("--rate-limit-attempts", type=int, default=3,
                        help="Retries per batch after a 429 before giving up")
    parser.add_argument("--commit-interval", type=float, default=0.0,
                        help="Extra sleep between successful commits, to stay under the hourly quota")
    parser.add_argument("--max-commits-per-hour", type=int, default=180,
                        help="Client-side cap; stays below the server's 200/h so we never see a 429")
    parser.add_argument("--cooldown", type=int, default=0,
                        help="Sleep this many seconds before the first commit, e.g. the N in "
                             "'try again after N seconds' from an earlier failed run")
    parser.add_argument("--only", action="append", default=None,
                        help="Only upload batches whose group matches this prefix (repeatable), "
                             "e.g. --only data --only videos/observation.images.top")
    parser.add_argument("--dry-run", action="store_true", help="List what would be uploaded and exit")
    args = parser.parse_args()

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
    print(f"target : https://modelscope.cn/{args.repo_type}s/{dest}  ({args.repo_type})")
    print(f"payload: {total_files} files, {human(total_bytes)} in {len(batches)} commits\n")
    for i, batch in enumerate(batches, 1):
        print(f"  [{i:>2}/{len(batches)}] {batch['key']:<40} {len(batch['files']):>3} files  {human(batch['bytes']):>10}")
    print()

    if args.dry_run:
        print("dry run — nothing uploaded")
        return 0

    force_one_commit_per_call(args.max_files_per_commit)
    if len(batches) > 150:
        print(f"warning: {len(batches)} commits planned — ModelScope allows ~200/hour per repo. "
              f"Raise --max-files-per-commit if this run aborts on a 429.", file=sys.stderr)

    api = HubApi()
    limiter = CommitLimiter(per_hour=args.max_commits_per_hour)
    limiter.penalize(args.cooldown)  # no-op when 0
    limiter.attach(api)  # counts every commit request, including the SDK's own
    try:
        api.login(args.token)  # caches the token; also accepts None for cached login
    except Exception as exc:  # noqa: BLE001 - surface any auth failure the same way
        print(f"error: not authenticated ({exc}). Set MODELSCOPE_API_TOKEN or pass --token.", file=sys.stderr)
        return 1

    visibility = "private" if args.private else None
    try:
        api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, visibility=visibility, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        print(f"error: failed to create repo {args.repo_id}: {exc}", file=sys.stderr)
        return 1

    done_bytes = 0
    for i, batch in enumerate(batches, 1):
        key = batch["key"]
        rels = [str(p.relative_to(local_dir)) for p in batch["files"]]
        print(f"\n[{i}/{len(batches)}] uploading {key}: {len(rels)} files, {human(batch['bytes'])}")
        try:
            run_batch(
                api,
                limiter=limiter,
                max_wait=args.max_rate_limit_wait,
                attempts=args.rate_limit_attempts,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                folder_path=str(local_dir),
                path_in_repo=prefix,
                commit_message=f"Upload {prefix + '/' if prefix else ''}{key} ({i}/{len(batches)})",
                allow_patterns=rels,
                ignore_patterns=GLOBAL_IGNORE,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"error: batch {key} failed: {exc}", file=sys.stderr)
            print("Re-run the same command; completed files are skipped on retry.", file=sys.stderr)
            return 1
        done_bytes += batch["bytes"]
        if args.commit_interval > 0 and i < len(batches):
            time.sleep(args.commit_interval)
        print(f"    ok — {human(done_bytes)} / {human(total_bytes)} ({100 * done_bytes / max(total_bytes, 1):.1f}%)")

    print(f"\ndone: https://modelscope.cn/{args.repo_type}s/{args.repo_id}/files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
