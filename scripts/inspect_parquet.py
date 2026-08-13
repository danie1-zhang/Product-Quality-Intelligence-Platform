from pathlib import Path

import pyarrow.parquet as pq

PARQUET_PATH = Path("data/raw/headphone_reviews.parquet")


def main():
    parquet_file = pq.ParquetFile(PARQUET_PATH)
    metadata = parquet_file.metadata

    print("Schema:")
    print(parquet_file.schema_arrow)
    print(f"Total rows: {metadata.num_rows}")
    print(f"Row groups: {metadata.num_row_groups}")
    print(f"File size: {PARQUET_PATH.stat().st_size / (1024 * 1024):.2f} MB")

    print("First 5 rows:")
    first_batch = next(parquet_file.iter_batches(batch_size=5), None)
    if first_batch is None:
        print("No rows found.")
    else:
        for row in first_batch.to_pylist():
            print(row)


if __name__ == "__main__":
    main()
