
def is_headphone_product(categories: list[str] | None) -> bool:
    """
    This function takes in a the categories a product is associated with and returns True if the product is a headphones
    or earbuds product and returns False otherwise.
    """

    if not categories:
        return False

    return "Headphones & Earbuds" in categories