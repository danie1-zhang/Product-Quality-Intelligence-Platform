import hashlib
import json
import os
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem

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

HEADPHONE_TITLE_TERMS = [
    "headphone",
    "headphones",
    "earbud",
    "earbuds",
    "earphone",
    "earphones",
    "headset",
    "in-ear",
    "over-ear",
    "on-ear",
]
HEADPHONE_TITLE_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9])(?:{'|'.join(map(re.escape, HEADPHONE_TITLE_TERMS))})(?![A-Za-z0-9])",
    re.IGNORECASE,
)
STANDARD_ASIN_PATTERN = re.compile(r"^B[A-Z0-9]{9}$", re.IGNORECASE)


def rows_to_table(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    """Convert canonical review mappings to a table with the canonical schema."""
    return pa.Table.from_pylist(rows, schema=CANONICAL_REVIEW_SCHEMA)


def write_rows_to_parquet(
    rows: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    batch_size: int = 5000,
) -> None:
    """
    Write canonical review rows to Parquet in bounded batches.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f"{output_path.name}.part")
    buffer: list[Mapping[str, Any]] = []

    with pq.ParquetWriter(partial_path, CANONICAL_REVIEW_SCHEMA) as writer:
        for row in rows:
            buffer.append(row)

            if len(buffer) == batch_size:
                table = rows_to_table(buffer)
                writer.write_table(table)
                buffer.clear()

        if buffer:
            table = rows_to_table(buffer)
            writer.write_table(table)

    os.replace(partial_path, output_path)


def iter_parquet_rows(
    fs: AbstractFileSystem,
    parquet_paths: Iterable[str | Path],
    batch_size: int = 5000,
) -> Iterator[dict[str, Any]]:
    """
    Yield raw review rows lazily from one or more Parquet files.
    """
    for path in parquet_paths:
        with fs.open(path, "rb") as f:
            parquet_file = pq.ParquetFile(f)

            for batch in parquet_file.iter_batches(batch_size=batch_size):
                yield from batch.to_pylist()


def ingest_headphone_reviews(
    fs: AbstractFileSystem,
    metadata_paths: Iterable[str | Path],
    review_paths: Iterable[str | Path],
    output_path: str | Path,
    read_batch_size: int = 5000,
    write_batch_size: int = 5000,
) -> None:
    """
    Ingest metadata and review Parquet files into atomic canonical output.
    """
    metadata_rows = iter_parquet_rows(fs, metadata_paths, batch_size=read_batch_size)
    headphone_ids = get_headphone_product_ids(metadata_rows)
    review_rows = iter_parquet_rows(fs, review_paths, batch_size=read_batch_size)
    canonical_rows = filter_and_transform_rows(review_rows, headphone_ids)
    write_rows_to_parquet(canonical_rows, output_path, batch_size=write_batch_size)


def is_headphone_product(categories: list[str] | None) -> bool:
    """
    Return whether categories contain the exact headphone domain category.
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


def transform_review(raw_review: Mapping[str, Any]) -> dict[str, Any]:
    """
    Transform one raw Amazon review into the canonical ingestion schema.

    Missing required Amazon review fields raise KeyError rather than silently
    producing an incomplete canonical row.
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


def is_headphone_title(title: str | None) -> bool:
    """
    Check whether a product title contains a strong headphone-related term.
    """
    return bool(title and HEADPHONE_TITLE_PATTERN.search(title))


def is_standard_asin(p_asin: str | None) -> bool:
    """
    Check for a 10-character alphanumeric ASIN beginning with "B".
    """
    return bool(p_asin and STANDARD_ASIN_PATTERN.fullmatch(p_asin))


def get_headphone_product_ids(metadata_rows: Iterable[Mapping[str, Any]]) -> set[str]:
    """
    Return the unique parent ASINs for valid headphone and earbud products.

    Each raw metadata row must include 'parent_asin', 'categories', and 'title'.
    """
    headphone_ids = set()
    for row in metadata_rows:
        if (
            row["parent_asin"]
            and is_headphone_product(row["categories"])
            and is_headphone_title(row["title"])
            and is_standard_asin(row["parent_asin"])
        ):
            headphone_ids.add(row["parent_asin"])
    return headphone_ids


def filter_and_transform_rows(
    review_rows: Iterable[Mapping[str, Any]], headphone_ids: set[str]
) -> Iterator[dict[str, Any]]:
    """
    Filter reviews to headphone products and lazily yield transformed rows.
    """
    for review in review_rows:
        if review["parent_asin"] in headphone_ids:
            yield transform_review(review)
