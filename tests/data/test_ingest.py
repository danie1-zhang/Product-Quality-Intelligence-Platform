from quality_intelligence.data.ingest import is_headphone_product

def test1():
    categories = None
    assert not is_headphone_product(categories) 

def test2():
    categories = []
    assert not is_headphone_product(categories) 

def test3():
    categories = ["Electronics"]
    assert not is_headphone_product(categories)

def test4():
    categories = ["Headphones, Earbuds & Accessories", "Headphones & Earbuds", "Electronics"]
    assert is_headphone_product(categories)

def test5():
    categories = ["Headphones, Earbuds & Accessories"]
    assert not is_headphone_product(categories)