from pathlib import Path

import pytest

from scripts import label_reviews


def test_parse_args_accepts_pipeline_configuration(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "label_reviews.py",
            "--input",
            "input.parquet",
            "--output",
            "output.parquet",
            "--sample-size",
            "4",
            "--master",
            "local[1]",
            "--driver-memory",
            "2g",
        ],
    )

    args = label_reviews.parse_args()

    assert args.input == Path("input.parquet")
    assert args.output == Path("output.parquet")
    assert args.sample_size == 4
    assert args.master == "local[1]"
    assert args.driver_memory == "2g"


def test_write_labeled_reviews_removes_partial_output_after_failure(tmp_path):
    output_path = tmp_path / "reviews.parquet"

    class FailingWriter:
        def parquet(self, path):
            Path(path).mkdir()
            raise RuntimeError("write failed")

    class FailingDataFrame:
        write = FailingWriter()

    with pytest.raises(RuntimeError, match="write failed"):
        label_reviews.write_labeled_reviews(FailingDataFrame(), output_path)

    assert not output_path.with_name("reviews.parquet.part").exists()


def test_run_labeling_unpersists_cached_dataframe_after_failure(tmp_path, monkeypatch):
    class FakeReader:
        def parquet(self, path):
            return object()

    class FakeSpark:
        read = FakeReader()

    class FakeLabeledDataFrame:
        unpersisted = False

        def cache(self):
            return self

        def unpersist(self):
            self.unpersisted = True

    labeled_df = FakeLabeledDataFrame()
    monkeypatch.setattr(label_reviews, "add_weak_labels", lambda df: labeled_df)

    def fail_summary(df, sample_size):
        raise RuntimeError("summary failed")

    monkeypatch.setattr(label_reviews, "show_labeling_summary", fail_summary)

    with pytest.raises(RuntimeError, match="summary failed"):
        label_reviews.run_labeling(
            FakeSpark(), tmp_path / "input.parquet", tmp_path / "output.parquet", 10
        )

    assert labeled_df.unpersisted
