import os

import pytest

from quality_intelligence.data.output import replace_output_directory


def test_replace_output_directory_swaps_staged_output(tmp_path):
    output_path = tmp_path / "reviews.parquet"
    staged_path = tmp_path / "reviews.parquet.part"
    output_path.mkdir()
    staged_path.mkdir()
    (output_path / "old").write_text("old")
    (staged_path / "new").write_text("new")

    replace_output_directory(staged_path, output_path)

    assert not staged_path.exists()
    assert not (tmp_path / "reviews.parquet.backup").exists()
    assert (output_path / "new").read_text() == "new"


def test_replace_output_directory_restores_old_output_when_swap_fails(tmp_path, monkeypatch):
    output_path = tmp_path / "reviews.parquet"
    staged_path = tmp_path / "reviews.parquet.part"
    output_path.mkdir()
    staged_path.mkdir()
    (output_path / "old").write_text("old")
    real_replace = os.replace

    def fail_staged_swap(source, destination):
        if source == staged_path:
            raise OSError("swap failed")
        real_replace(source, destination)

    monkeypatch.setattr("quality_intelligence.data.output.os.replace", fail_staged_swap)

    with pytest.raises(OSError, match="swap failed"):
        replace_output_directory(staged_path, output_path)

    assert (output_path / "old").read_text() == "old"
    assert staged_path.exists()


def test_replace_output_directory_recovers_orphaned_backup(tmp_path):
    output_path = tmp_path / "reviews.parquet"
    backup_path = tmp_path / "reviews.parquet.backup"
    staged_path = tmp_path / "reviews.parquet.part"
    backup_path.mkdir()
    staged_path.mkdir()
    (backup_path / "old").write_text("old")
    (staged_path / "new").write_text("new")

    replace_output_directory(staged_path, output_path)

    assert (output_path / "new").read_text() == "new"
    assert not backup_path.exists()
