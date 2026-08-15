import logging
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

import pyarrow.parquet as pq
from fsspec import AbstractFileSystem

LOGGER = logging.getLogger(__name__)


def is_readable_parquet(path: Path) -> bool:
    """Return whether a local file has readable Parquet metadata."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        pq.ParquetFile(path)
    except (OSError, ValueError):
        return False
    return True


def cache_remote_shard(
    fs: AbstractFileSystem,
    remote_path: str,
    cache_dir: str | Path,
    shard_type: str,
    index: int,
    total: int,
) -> Path:
    """Download one Parquet shard atomically, or reuse a readable cached copy."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / Path(remote_path).name
    partial_path = local_path.with_name(f"{local_path.name}.part")

    if is_readable_parquet(local_path):
        LOGGER.info("Using cached %s shard %s/%s", shard_type, index, total)
        return local_path

    LOGGER.info("Downloading %s shard %s/%s", shard_type, index, total)
    with fs.open(remote_path, "rb") as source, partial_path.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)

    if not is_readable_parquet(partial_path):
        raise OSError(f"Downloaded shard is not a readable Parquet file: {remote_path}")

    os.replace(partial_path, local_path)
    return local_path


def cache_remote_shards(
    fs: AbstractFileSystem,
    remote_paths: Iterable[str],
    cache_dir: str | Path,
    shard_type: str,
) -> list[Path]:
    """Cache remote shards sequentially and return their local paths."""
    remote_paths = list(remote_paths)
    return [
        cache_remote_shard(fs, path, cache_dir, shard_type, index, len(remote_paths))
        for index, path in enumerate(remote_paths, start=1)
    ]
