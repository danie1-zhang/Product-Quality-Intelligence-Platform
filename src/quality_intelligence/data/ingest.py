from typing import Any
import json
import hashlib


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
        "verified_purchase": verified_purchase
    }

