from io import BytesIO

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quality_intelligence.data.cache import cache_remote_shard


def make_parquet_bytes():
    destination = pa.BufferOutputStream()
    pq.write_table(pa.table({"value": [1]}), destination)
    return destination.getvalue().to_pybytes()


class FakeRemoteFileSystem:
    def __init__(self, content=None):
        self.content = make_parquet_bytes() if content is None else content
        self.opened_paths = []

    def open(self, path, mode):
        self.opened_paths.append((path, mode))
        return BytesIO(self.content)


def test_existing_cached_file_is_reused(tmp_path):
    cached_file = tmp_path / "shard.parquet"
    cached_content = make_parquet_bytes()
    cached_file.write_bytes(cached_content)
    fs = FakeRemoteFileSystem()

    result = cache_remote_shard(fs, "remote/shard.parquet", tmp_path, "metadata", 1, 1)

    assert result == cached_file
    assert cached_file.read_bytes() == cached_content
    assert fs.opened_paths == []


def test_missing_file_is_downloaded(tmp_path):
    downloaded_content = make_parquet_bytes()
    fs = FakeRemoteFileSystem(downloaded_content)

    result = cache_remote_shard(fs, "remote/shard.parquet", tmp_path, "review", 1, 1)

    assert fs.opened_paths == [("remote/shard.parquet", "rb")]
    assert result.read_bytes() == downloaded_content


def test_partial_file_is_not_treated_as_complete(tmp_path):
    partial_file = tmp_path / "shard.parquet.part"
    partial_file.write_bytes(b"interrupted data")
    complete_content = make_parquet_bytes()
    fs = FakeRemoteFileSystem(complete_content)

    result = cache_remote_shard(fs, "remote/shard.parquet", tmp_path, "review", 1, 1)

    assert fs.opened_paths == [("remote/shard.parquet", "rb")]
    assert result.read_bytes() == complete_content


def test_successful_download_renames_partial_file(tmp_path):
    fs = FakeRemoteFileSystem()

    result = cache_remote_shard(fs, "remote/shard.parquet", tmp_path, "metadata", 1, 1)

    assert result == tmp_path / "shard.parquet"
    assert result.exists()
    assert not (tmp_path / "shard.parquet.part").exists()


def test_corrupt_cached_file_is_replaced(tmp_path):
    cached_file = tmp_path / "shard.parquet"
    cached_file.write_bytes(b"corrupt")
    downloaded_content = make_parquet_bytes()
    fs = FakeRemoteFileSystem(downloaded_content)

    result = cache_remote_shard(fs, "remote/shard.parquet", tmp_path, "review", 1, 1)

    assert fs.opened_paths == [("remote/shard.parquet", "rb")]
    assert result.read_bytes() == downloaded_content


def test_invalid_download_is_not_promoted_to_final_file(tmp_path):
    fs = FakeRemoteFileSystem(b"not parquet")

    with pytest.raises(OSError, match="not a readable Parquet"):
        cache_remote_shard(fs, "remote/shard.parquet", tmp_path, "review", 1, 1)

    assert not (tmp_path / "shard.parquet").exists()
    assert not (tmp_path / "shard.parquet.part").exists()
