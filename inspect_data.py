import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

fs = HfFileSystem()

path = (
    "datasets/McAuley-Lab/Amazon-Reviews-2023@"
    "ac9d3ad3342d6f00bf6ad8caa2668a3f830e2dee/"
    "raw_meta_Electronics/full-00000-of-00010.parquet"
)

with fs.open(path, "rb") as f:
    parquet_file = pq.ParquetFile(f)

    print(parquet_file.schema_arrow)

    first_batch = next(parquet_file.iter_batches(batch_size=5000))

    for row in first_batch.to_pylist():
        categories = row["categories"] or []

        if any("Headphones" in category for category in categories):
            print(row["title"])
            print(categories)
            print()