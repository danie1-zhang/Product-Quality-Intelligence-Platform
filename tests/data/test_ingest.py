from quality_intelligence.data.ingest import is_headphone_product

def categories_is_none():
    categories = None
    assert not is_headphone_product(categories) 

def categories_is_empty():
    categories = []
    assert not is_headphone_product(categories) 

def categories_does_not_contain_headphones_earbuds():
    categories = ["Electronics"]
    assert not is_headphone_product(categories)

def categories_contains_headphones_earbuds():
    categories = ["Headphones, Earbuds & Accessories", "Headphones & Earbuds", "Electronics"]
    assert is_headphone_product(categories)

def categories_contains_similar():
    categories = ["Headphones, Earbuds & Accessories"]
    assert not is_headphone_product(categories)