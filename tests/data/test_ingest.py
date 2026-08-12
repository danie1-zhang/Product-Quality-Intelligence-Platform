from quality_intelligence.data.ingest import is_headphone_product, generate_review_id

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
