"""Finding source episodes on disk and putting them in a stable order."""

from __future__ import annotations

from pathlib import Path


def discover_rosbags(source: Path, recursive: bool = True) -> list[Path]:
    """Find rosbag2 episode directories, for both sqlite3 and MCAP storage."""
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        if source.suffix not in {".db3", ".mcap"}:
            raise ValueError(f"Expected a rosbag directory, .db3 or .mcap, got {source}")
        return [source.parent]
    if (source / "metadata.yaml").is_file() or any(source.glob("*.db3")) or any(source.glob("*.mcap")):
        return [source]
    bags: set[Path] = set()
    patterns = ("**/metadata.yaml", "**/*.db3", "**/*.mcap")
    if not recursive:
        patterns = ("*/metadata.yaml", "*/*.db3", "*/*.mcap")
    for pattern in patterns:
        bags.update(path.parent for path in source.glob(pattern))
    return sorted(bags, key=lambda path: str(path))


def discover_hdf5(source: Path, recursive: bool = True) -> list[Path]:
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        return [source]
    patterns = ("**/*.hdf5", "**/*.h5") if recursive else ("*.hdf5", "*.h5")
    files: set[Path] = set()
    for pattern in patterns:
        files.update(source.glob(pattern))
    return sorted(files, key=lambda path: str(path))


def sort_and_limit(paths: list[Path], sort_by: str, limit: int) -> list[Path]:
    if sort_by == "mtime":
        paths = sorted(paths, key=lambda path: (path.stat().st_mtime_ns, str(path)))
    elif sort_by == "name":
        paths = sorted(paths, key=lambda path: str(path))
    else:
        raise ValueError("sort_by must be name or mtime")
    return paths[:limit] if limit > 0 else paths


def bag_storage_kind(bag_dir: Path) -> str:
    """Which rosbag2 storage backend a bag directory uses."""
    if any(bag_dir.glob("*.db3")):
        return "sqlite3"
    if any(bag_dir.glob("*.mcap")):
        return "mcap"
    return "unknown"
