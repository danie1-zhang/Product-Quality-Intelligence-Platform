import pytest
from pyspark.sql import SparkSession

from quality_intelligence.data.weak_labeling_spark import add_weak_labels


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[1]")
        .appName("test-weak-labeling")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_add_weak_labels_resolves_labels_statuses_and_nulls(spark):
    rows = [
        ("functionality", "It STOPPED WORKING yesterday.", 1.0),
        ("build", "The headband cracked.", 2.0),
        ("shipping", "The box was crushed.", 1.0),
        ("fit", "These are too tight.", 2.0),
        ("setup", "There were no instructions.", 2.0),
        ("positive", "This was a great purchase.", 4.0),
        ("abstain", "It arrived yesterday.", 5.0),
        ("low-positive", "This was a great purchase.", 3.0),
        ("complaint-conflict", "It stopped working and the headband cracked.", 1.0),
        ("positive-conflict", "It works great but stopped working.", 5.0),
        ("null-text", None, 5.0),
        ("null-rating", "This works great.", None),
    ]
    df = spark.createDataFrame(rows, "review_id string, cleaned_review_text string, rating double")

    results = {row.review_id: row for row in add_weak_labels(df).collect()}

    assert (results["functionality"].weak_label, results["functionality"].weak_label_status) == (
        "FUNCTIONALITY",
        "LABELED",
    )
    assert (results["build"].weak_label, results["build"].weak_label_status) == (
        "BUILD_QUALITY",
        "LABELED",
    )
    assert (results["shipping"].weak_label, results["shipping"].weak_label_status) == (
        "SHIPPING",
        "LABELED",
    )
    assert (results["fit"].weak_label, results["fit"].weak_label_status) == (
        "FIT_COMPATIBILITY",
        "LABELED",
    )
    assert (results["setup"].weak_label, results["setup"].weak_label_status) == (
        "USABILITY_SETUP",
        "LABELED",
    )
    assert (results["positive"].weak_label, results["positive"].weak_label_status) == (
        "NO_COMPLAINT",
        "LABELED",
    )
    for review_id in ("abstain", "low-positive", "null-text", "null-rating"):
        assert (results[review_id].weak_label, results[review_id].weak_label_status) == (
            None,
            "ABSTAIN",
        )
    for review_id in ("complaint-conflict", "positive-conflict"):
        assert (results[review_id].weak_label, results[review_id].weak_label_status) == (
            None,
            "CONFLICT",
        )


def test_add_weak_labels_preserves_input_columns(spark):
    df = spark.createDataFrame(
        [("review-1", "No matching language.", 3.0)],
        "review_id string, cleaned_review_text string, rating double",
    )

    result = add_weak_labels(df)

    assert result.columns == [*df.columns, "weak_label", "weak_label_status"]


def test_add_weak_labels_replaces_existing_output_columns(spark):
    df = spark.createDataFrame(
        [("It stopped working.", 1.0, "OLD", "OLD")],
        "cleaned_review_text string, rating double, weak_label string, weak_label_status string",
    )

    result = add_weak_labels(df).first()

    assert result.weak_label == "FUNCTIONALITY"
    assert result.weak_label_status == "LABELED"
