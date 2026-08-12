import pytest

from quality_intelligence.data.ingest import (
    filter_and_transform_rows,
    generate_review_id,
    get_headphone_product_ids,
    is_headphone_product,
    transform_review,
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
