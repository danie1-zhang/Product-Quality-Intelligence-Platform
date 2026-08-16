def pytest_collection_modifyitems(items):
    """Group tests that require a local Spark gateway as integration tests."""
    for item in items:
        if item.path.name in {"test_preprocess.py", "test_weak_labeling_spark.py"}:
            item.add_marker("integration")
