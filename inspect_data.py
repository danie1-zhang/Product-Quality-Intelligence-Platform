import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REVISION = "ac9d3ad3342d6f00bf6ad8caa2668a3f830e2dee"
METADATA_PATH = (
    f"datasets/McAuley-Lab/Amazon-Reviews-2023@{REVISION}/"
    "raw_meta_Electronics/full-00000-of-00010.parquet"
)


def main():
    fs = HfFileSystem()

    with fs.open(METADATA_PATH, "rb") as file:
        parquet_file = pq.ParquetFile(file)
        print(parquet_file.schema_arrow)

        first_batch = next(parquet_file.iter_batches(batch_size=5000), None)
        if first_batch is None:
            return

        for row in first_batch.to_pylist():
            categories = row["categories"] or []
            if any("Headphones" in category for category in categories):
                print(row["title"])
                print(categories)
                print()


if __name__ == "__main__":
    main()
