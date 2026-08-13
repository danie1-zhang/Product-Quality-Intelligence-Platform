import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fsspec.implementations.local import LocalFileSystem

from quality_intelligence.data.ingest import (
    CANONICAL_REVIEW_SCHEMA,
    filter_and_transform_rows,
    generate_review_id,
    get_headphone_product_ids,
    ingest_headphone_reviews,
    is_headphone_product,
    iter_parquet_rows,
    rows_to_table,
    transform_review,
    write_rows_to_parquet,
)


@pytest.fixture
def raw_review():
    return {
        "rating": 5.0,
        "title": "Excellent sound and battery life",
        "text": "These headphones sound great and last all day.",
        "parent_asin": "B0CHEADPHN",
        "asin": "B0CVARIANT1",
        "user_id": "AGX7EXAMPLEUSER",
        "timestamp": 1711929600000,
        "helpful_vote": 12,
        "verified_purchase": True,
        "images": [{"small_image_url": "https://example.com/review-image.jpg"}],
    }


@pytest.fixture
def local_fs():
    return LocalFileSystem()


def make_canonical_reviews(raw_review, count):
    return [
        transform_review({**raw_review, "text": f"Review number {number}"})
        for number in range(1, count + 1)
    ]


# is_headphone_product tests
def test_categories_is_none():
    categories = None
    assert not is_headphone_product(categories)


def test_categories_is_empty():
    categories = []
    assert not is_headphone_product(categories)


def test_categories_does_not_contain_headphones_earbuds():
    categories = ["Electronics"]
    assert not is_headphone_product(categories)


def test_categories_contains_headphones_earbuds():
    categories = ["Headphones, Earbuds & Accessories", "Headphones & Earbuds", "Electronics"]
    assert is_headphone_product(categories)


def test_categories_contains_similar():
    categories = ["Headphones, Earbuds & Accessories"]
    assert not is_headphone_product(categories)


# generate_review_id tests
def test_two_different_reviews():
    parent_asin1, user_id1, timestamp1, review_text1 = "airpods", "12jsj4", 213234234, "ok"
    parent_asin2, user_id2, timestamp2, review_text2 = "cheese", "123984j4", 99089, "bad"
    id1 = generate_review_id(parent_asin1, user_id1, timestamp1, review_text1)
    id2 = generate_review_id(parent_asin2, user_id2, timestamp2, review_text2)
    assert id1 != id2


def test_same_source_fields_produce_same_id():
    parent_asin, user_id, timestamp, review_text = "airpods", "12jsj4", 213234234, "ok"
    id1 = generate_review_id(parent_asin, user_id, timestamp, review_text)
    id2 = generate_review_id(parent_asin, user_id, timestamp, review_text)
    assert id1 == id2


def test_one_character_change_in_review_text_produces_different_id():
    parent_asin, user_id, timestamp = "airpods", "12jsj4", 213234234
    id1 = generate_review_id(parent_asin, user_id, timestamp, "ok")
    id2 = generate_review_id(parent_asin, user_id, timestamp, "oh")
    assert id1 != id2


# transform_review tests
def test_transform_review_output_has_expected_keys(raw_review):
    transformed = transform_review(raw_review)
    assert set(transformed) == {
        "review_id",
        "product_id",
        "review_title",
        "review_text",
        "rating",
        "timestamp",
        "helpful_votes",
        "verified_purchase",
    }


def test_transform_review_maps_raw_fields_to_canonical_fields(raw_review):
    transformed = transform_review(raw_review)
    assert transformed["product_id"] == raw_review["parent_asin"]
    assert transformed["review_title"] == raw_review["title"]
    assert transformed["review_text"] == raw_review["text"]
    assert transformed["rating"] == raw_review["rating"]
    assert transformed["timestamp"] == raw_review["timestamp"]
    assert transformed["helpful_votes"] == raw_review["helpful_vote"]
    assert transformed["verified_purchase"] == raw_review["verified_purchase"]


def test_transform_review_generates_expected_review_id(raw_review):
    transformed = transform_review(raw_review)
    expected_id = generate_review_id(
        raw_review["parent_asin"],
        raw_review["user_id"],
        raw_review["timestamp"],
        raw_review["text"],
    )
    assert transformed["review_id"] == expected_id


def test_transform_review_excludes_extra_raw_fields(raw_review):
    transformed = transform_review(raw_review)
    assert "images" not in transformed
    assert "asin" not in transformed


# get_headphone_product_ids tests
def test_get_headphone_product_ids_includes_headphone_product():
    metadata_rows = [
        {"parent_asin": "B0CHEADPHN", "categories": ["Electronics", "Headphones & Earbuds"]},
    ]
    assert get_headphone_product_ids(metadata_rows) == {"B0CHEADPHN"}


def test_get_headphone_product_ids_excludes_non_headphone_product():
    metadata_rows = [
        {"parent_asin": "B0CSPEAKER", "categories": ["Electronics", "Portable Speakers"]},
    ]
    assert get_headphone_product_ids(metadata_rows) == set()


def test_get_headphone_product_ids_removes_duplicate_parent_asins():
    metadata_rows = [
        {"parent_asin": "B0CHEADPHN", "categories": ["Headphones & Earbuds"]},
        {"parent_asin": "B0CHEADPHN", "categories": ["Electronics", "Headphones & Earbuds"]},
    ]
    assert get_headphone_product_ids(metadata_rows) == {"B0CHEADPHN"}


def test_get_headphone_product_ids_excludes_none_parent_asin():
    metadata_rows = [
        {"parent_asin": None, "categories": ["Headphones & Earbuds"]},
    ]
    assert get_headphone_product_ids(metadata_rows) == set()


# filter_and_transform_rows tests
def test_filter_and_transform_rows_yields_matching_review(raw_review):
    results = list(filter_and_transform_rows([raw_review], {raw_review["parent_asin"]}))
    assert len(results) == 1


def test_filter_and_transform_rows_skips_non_matching_review(raw_review):
    results = list(filter_and_transform_rows([raw_review], {"B0COTHER"}))
    assert results == []


def test_filter_and_transform_rows_yields_canonical_schema(raw_review):
    results = list(filter_and_transform_rows([raw_review], {raw_review["parent_asin"]}))
    assert set(results[0]) == {
        "review_id",
        "product_id",
        "review_title",
        "review_text",
        "rating",
        "timestamp",
        "helpful_votes",
        "verified_purchase",
    }


def test_filter_and_transform_rows_with_empty_rows_yields_nothing():
    results = list(filter_and_transform_rows([], {"B0CHEADPHN"}))
    assert results == []


# rows_to_table tests
def test_rows_to_table_uses_canonical_review_schema(raw_review):
    canonical_review = transform_review(raw_review)
    table = rows_to_table([canonical_review])
    assert table.schema == CANONICAL_REVIEW_SCHEMA


def test_rows_to_table_preserves_review_values(raw_review):
    canonical_review = transform_review(raw_review)
    table = rows_to_table([canonical_review])
    assert table.to_pylist() == [canonical_review]


# write_rows_to_parquet tests
def test_write_rows_to_parquet_creates_readable_file_with_canonical_schema(raw_review, tmp_path):
    canonical_review = transform_review(raw_review)
    output_path = tmp_path / "reviews.parquet"

    write_rows_to_parquet([canonical_review], output_path)

    table = pq.read_table(output_path)
    assert table.schema == CANONICAL_REVIEW_SCHEMA


def test_write_rows_to_parquet_writes_all_batches_and_final_partial_batch(raw_review, tmp_path):
    canonical_reviews = make_canonical_reviews(raw_review, 3)
    output_path = tmp_path / "batched-reviews.parquet"

    write_rows_to_parquet(iter(canonical_reviews), output_path, batch_size=2)

    table = pq.read_table(output_path)
    assert table.to_pylist() == canonical_reviews


# iter_parquet_rows tests
def test_iter_parquet_rows_yields_all_rows(raw_review, tmp_path, local_fs):
    canonical_reviews = make_canonical_reviews(raw_review, 3)
    parquet_path = tmp_path / "reviews.parquet"
    pq.write_table(rows_to_table(canonical_reviews), parquet_path)

    results = list(iter_parquet_rows(local_fs, [parquet_path]))

    assert results == canonical_reviews


def test_iter_parquet_rows_yields_rows_across_record_batches(raw_review, tmp_path, local_fs):
    canonical_reviews = make_canonical_reviews(raw_review, 5)
    parquet_path = tmp_path / "batched-reviews.parquet"
    pq.write_table(rows_to_table(canonical_reviews), parquet_path)

    results = list(iter_parquet_rows(local_fs, [parquet_path], batch_size=2))

    assert results == canonical_reviews


def test_iter_parquet_rows_yields_rows_from_multiple_files(raw_review, tmp_path, local_fs):
    first_reviews = make_canonical_reviews(raw_review, 2)
    second_reviews = make_canonical_reviews({**raw_review, "user_id": "SECONDUSER"}, 2)
    first_path = tmp_path / "first.parquet"
    second_path = tmp_path / "second.parquet"
    pq.write_table(rows_to_table(first_reviews), first_path)
    pq.write_table(rows_to_table(second_reviews), second_path)

    results = list(iter_parquet_rows(local_fs, [first_path, second_path]))

    assert results == first_reviews + second_reviews


def test_iter_parquet_rows_empty_file_yields_no_rows(tmp_path, local_fs):
    parquet_path = tmp_path / "empty.parquet"
    empty_table = pa.Table.from_pylist([], schema=CANONICAL_REVIEW_SCHEMA)
    pq.write_table(empty_table, parquet_path)

    results = list(iter_parquet_rows(local_fs, [parquet_path]))

    assert results == []


# ingest_headphone_reviews tests
def test_ingest_headphone_reviews_filters_and_transforms_end_to_end(raw_review, tmp_path, local_fs):
    metadata_path = tmp_path / "metadata.parquet"
    reviews_path = tmp_path / "reviews.parquet"
    output_path = tmp_path / "headphone-reviews.parquet"
    metadata_rows = [
        {
            "parent_asin": raw_review["parent_asin"],
            "categories": ["Electronics", "Headphones & Earbuds"],
        },
        {"parent_asin": "B0CSPEAKER", "categories": ["Electronics", "Portable Speakers"]},
    ]
    non_headphone_review = {
        **raw_review,
        "parent_asin": "B0CSPEAKER",
        "title": "Good portable speaker",
        "text": "Loud enough for a small room.",
    }
    pq.write_table(pa.Table.from_pylist(metadata_rows), metadata_path)
    pq.write_table(pa.Table.from_pylist([raw_review, non_headphone_review]), reviews_path)

    ingest_headphone_reviews(
        local_fs,
        [metadata_path],
        [reviews_path],
        output_path,
        read_batch_size=1,
        write_batch_size=1,
    )

    output_table = pq.read_table(output_path)
    assert output_table.schema == CANONICAL_REVIEW_SCHEMA
    assert output_table.to_pylist() == [transform_review(raw_review)]
