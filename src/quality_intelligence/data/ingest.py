import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq

CANONICAL_REVIEW_SCHEMA = pa.schema(
    [
        ("review_id", pa.string()),
        ("product_id", pa.string()),
        ("review_title", pa.string()),
        ("review_text", pa.string()),
        ("rating", pa.float64()),
        ("timestamp", pa.int64()),
        ("helpful_votes", pa.int64()),
        ("verified_purchase", pa.bool_()),
    ]
)


def rows_to_table(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=CANONICAL_REVIEW_SCHEMA)


def write_rows_to_parquet(rows, output_path: str, batch_size: int = 5000):
    """
    Write canonical review rows to Parquet in bounded batches.
    """
    buffer = []

    with pq.ParquetWriter(output_path, CANONICAL_REVIEW_SCHEMA) as writer:
        for row in rows:
            buffer.append(row)

            if len(buffer) == batch_size:
                table = rows_to_table(buffer)
                writer.write_table(table)
                buffer.clear()

        if buffer:
            table = rows_to_table(buffer)
            writer.write_table(table)


def iter_parquet_rows(fs, parquet_paths, batch_size=5000):
    """
    Yield raw review rows lazily from one or more Parquet files.
    """
    for path in parquet_paths:
        with fs.open(path, "rb") as f:
            parquet_file = pq.ParquetFile(f)

            for batch in parquet_file.iter_batches(batch_size=batch_size):
                yield from batch.to_pylist()


def ingest_headphone_reviews(
    fs,
    metadata_paths,
    review_paths,
    output_path: str,
    read_batch_size: int = 5000,
    write_batch_size: int = 5000,
):
    """Ingest headphone reviews from remote Parquet inputs into canonical output."""
    metadata_rows = iter_parquet_rows(fs, metadata_paths, batch_size=read_batch_size)
    headphone_ids = get_headphone_product_ids(metadata_rows)
    review_rows = iter_parquet_rows(fs, review_paths, batch_size=read_batch_size)
    canonical_rows = filter_and_transform_rows(review_rows, headphone_ids)
    write_rows_to_parquet(canonical_rows, output_path, batch_size=write_batch_size)


def is_headphone_product(categories: list[str] | None) -> bool:
    """
    This function takes in a the categories a product is associated with and returns True if the product is a headphones
    or earbuds product and returns False otherwise.
    """
    if not categories:
        return False
    return "Headphones & Earbuds" in categories


def generate_review_id(parent_asin: str, user_id: str, timestamp: int, review_text: str) -> str:
    """
    Generate a deterministic review ID from stable source fields.
    """
    source_fields = [parent_asin, user_id, timestamp, review_text]
    s = json.dumps(source_fields).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def transform_review(raw_review: dict) -> dict:
    """
    Transform one raw Amazon review into the canonical ingestion schema.
    """
    rating = raw_review["rating"]
    review_title = raw_review["title"]
    product_id = raw_review["parent_asin"]
    review_text = raw_review["text"]
    timestamp = raw_review["timestamp"]
    user_id = raw_review["user_id"]
    helpful_votes = raw_review["helpful_vote"]
    verified_purchase = raw_review["verified_purchase"]
    review_id = generate_review_id(product_id, user_id, timestamp, review_text)

    return {
        "review_id": review_id,
        "review_title": review_title,
        "review_text": review_text,
        "product_id": product_id,
        "rating": rating,
        "timestamp": timestamp,
        "helpful_votes": helpful_votes,
        "verified_purchase": verified_purchase,
    }


def get_headphone_product_ids(metadata_rows) -> set[str]:
    """
    Return the unique parent ASINs for valid headphone and earbud products.
    """
    headphone_ids = set()
    for row in metadata_rows:
        if row["parent_asin"] and is_headphone_product(row["categories"]):
            headphone_ids.add(row["parent_asin"])
    return headphone_ids


def filter_and_transform_rows(review_rows, headphone_ids: set[str]):
    """
    Filter reviews to headphone products and lazily yield transformed rows.
    """
    for review in review_rows:
        if review["parent_asin"] in headphone_ids:
            yield transform_review(review)
