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
