from datetime import date

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, DoubleType, IntegerType, LongType, TimestampType

from quality_intelligence.data.preprocess import (
    add_product_historical_features,
    add_review_features,
    clean_review_text,
    deduplicate_reviews,
    normalize_timestamps,
    preprocess_reviews,
)


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[1]")
        .appName("test-preprocess")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    yield session
    session.stop()


def make_reviews(spark, rows):
    return spark.createDataFrame(rows, ["review_id", "helpful_votes", "review_text"])


def make_review_texts(spark, *texts):
    return spark.createDataFrame([(text,) for text in texts], "review_text string")


def make_timestamped_reviews(spark, rows):
    return spark.createDataFrame(rows, "review_id string, timestamp long, review_text string")


def make_cleaned_reviews(spark, rows):
    return spark.createDataFrame(
        rows,
        "review_id string, cleaned_review_text string, rating int",
    )


def make_product_reviews(spark, rows):
    return spark.createDataFrame(
        rows,
        "review_id string, product_id string, timestamp long, rating double",
    )


def make_canonical_spark_reviews(spark, rows):
    return spark.createDataFrame(
        rows,
        "review_id string, product_id string, review_title string, review_text string, "
        "rating double, timestamp long, helpful_votes long, verified_purchase boolean",
    )


def test_no_duplicates(spark):
    reviews = make_reviews(spark, [("r1", 2, "first"), ("r2", 5, "second")])

    result = deduplicate_reviews(reviews)

    assert {tuple(row) for row in result.collect()} == {
        ("r1", 2, "first"),
        ("r2", 5, "second"),
    }


def test_duplicate_id_keeps_highest_helpful_votes(spark):
    reviews = make_reviews(spark, [("r1", 2, "less helpful"), ("r1", 8, "more helpful")])

    result = deduplicate_reviews(reviews).collect()

    assert len(result) == 1
    assert tuple(result[0]) == ("r1", 8, "more helpful")


def test_duplicate_id_tie_is_resolved_deterministically(spark):
    rows = [("r1", 8, "zulu"), ("r1", 8, "alpha")]

    forward = tuple(deduplicate_reviews(make_reviews(spark, rows)).first())
    reverse = tuple(deduplicate_reviews(make_reviews(spark, reversed(rows))).first())

    assert forward == reverse


def test_exact_duplicate_rows_leave_one_row(spark):
    reviews = make_reviews(spark, [("r1", 3, "same"), ("r1", 3, "same")])

    result = deduplicate_reviews(reviews).collect()

    assert len(result) == 1
    assert tuple(result[0]) == ("r1", 3, "same")


def test_three_rows_sharing_one_review_id_leave_one_row(spark):
    reviews = make_reviews(
        spark,
        [("r1", 1, "low"), ("r1", 9, "high"), ("r1", 4, "middle")],
    )

    result = deduplicate_reviews(reviews).collect()

    assert len(result) == 1
    assert tuple(result[0]) == ("r1", 9, "high")


def test_different_review_ids_remain_separate(spark):
    reviews = make_reviews(spark, [("r1", 4, "first"), ("r2", 4, "second")])

    result = deduplicate_reviews(reviews)

    assert result.count() == 2
    assert {row.review_id for row in result.collect()} == {"r1", "r2"}


def test_output_schema_excludes_temporary_row_number_column(spark):
    reviews = make_reviews(spark, [("r1", 1, "low"), ("r1", 2, "high")])

    result = deduplicate_reviews(reviews)

    assert result.columns == reviews.columns
    assert "rn" not in result.columns


def test_clean_review_text_drops_none(spark):
    result = clean_review_text(make_review_texts(spark, None))

    assert result.count() == 0


def test_clean_review_text_drops_empty_string(spark):
    result = clean_review_text(make_review_texts(spark, ""))

    assert result.count() == 0


def test_clean_review_text_drops_spaces_only(spark):
    result = clean_review_text(make_review_texts(spark, "   "))

    assert result.count() == 0


def test_clean_review_text_drops_newline_and_tab_only(spark):
    result = clean_review_text(make_review_texts(spark, "\n\t\n"))

    assert result.count() == 0


def test_clean_review_text_drops_only_html_line_breaks(spark):
    result = clean_review_text(make_review_texts(spark, "<br /><br />"))

    assert result.count() == 0


def test_clean_review_text_keeps_short_meaningful_text(spark):
    result = clean_review_text(make_review_texts(spark, "bad")).first()

    assert result.cleaned_review_text == "bad"


def test_clean_review_text_keeps_normal_review(spark):
    text = "Great Sound! I would buy it again."

    result = clean_review_text(make_review_texts(spark, text)).first()

    assert result.cleaned_review_text == text


@pytest.mark.parametrize("line_break", ["<br>", "<br/>", "<br />"])
def test_clean_review_text_replaces_html_line_break_variants(spark, line_break):
    result = clean_review_text(make_review_texts(spark, f"good{line_break}sound")).first()

    assert result.cleaned_review_text == "good sound"


def test_clean_review_text_collapses_repeated_whitespace(spark):
    result = clean_review_text(make_review_texts(spark, "good   sound\n\nwith\t\tbass")).first()

    assert result.cleaned_review_text == "good sound with bass"


def test_clean_review_text_trims_leading_and_trailing_whitespace(spark):
    result = clean_review_text(make_review_texts(spark, " \n good sound \t ")).first()

    assert result.cleaned_review_text == "good sound"


def test_clean_review_text_preserves_original_review_text(spark):
    text = "  Great<br />sound!\n"

    result = clean_review_text(make_review_texts(spark, text)).first()

    assert result.review_text == text


def test_clean_review_text_adds_cleaned_column_to_schema(spark):
    reviews = make_review_texts(spark, "good")

    result = clean_review_text(reviews)

    assert result.columns == ["review_text", "cleaned_review_text"]


def test_normalize_timestamps_converts_milliseconds_to_datetime(spark):
    reviews = make_timestamped_reviews(spark, [("r1", 1704067200000, "good")])

    result = (
        normalize_timestamps(reviews)
        .select(F.date_format("review_datetime", "yyyy-MM-dd HH:mm:ss").alias("review_datetime"))
        .first()
    )

    assert result.review_datetime == "2024-01-01 00:00:00"


def test_normalize_timestamps_derives_review_date(spark):
    reviews = make_timestamped_reviews(spark, [("r1", 1704153599000, "good")])

    result = (
        normalize_timestamps(reviews)
        .select(
            F.date_format("review_datetime", "yyyy-MM-dd HH:mm:ss").alias("review_datetime"),
            "review_date",
        )
        .first()
    )

    assert result.review_datetime == "2024-01-01 23:59:59"
    assert result.review_date == date(2024, 1, 1)


def test_normalize_timestamps_preserves_original_timestamp(spark):
    timestamp = 1704067200123
    reviews = make_timestamped_reviews(spark, [("r1", timestamp, "good")])

    result = normalize_timestamps(reviews).first()

    assert result.timestamp == timestamp


def test_normalize_timestamps_adds_timestamp_typed_datetime_column(spark):
    reviews = make_timestamped_reviews(spark, [("r1", 1704067200000, "good")])

    result = normalize_timestamps(reviews)

    assert isinstance(result.schema["review_datetime"].dataType, TimestampType)


def test_normalize_timestamps_adds_date_typed_date_column(spark):
    reviews = make_timestamped_reviews(spark, [("r1", 1704067200000, "good")])

    result = normalize_timestamps(reviews)

    assert isinstance(result.schema["review_date"].dataType, DateType)


def test_normalize_timestamps_preserves_unrelated_columns(spark):
    reviews = make_timestamped_reviews(spark, [("r1", 1704067200000, "Great sound!")])

    result = normalize_timestamps(reviews).first()

    assert result.review_id == "r1"
    assert result.review_text == "Great sound!"


def test_normalize_timestamps_converts_multiple_rows_independently(spark):
    reviews = make_timestamped_reviews(
        spark,
        [("r1", 0, "first"), ("r2", 86400000, "second")],
    )

    normalized = normalize_timestamps(reviews).select(
        "review_id",
        F.date_format("review_datetime", "yyyy-MM-dd HH:mm:ss").alias("review_datetime"),
        "review_date",
    )
    result = {row.review_id: row for row in normalized.collect()}

    assert result["r1"].review_datetime == "1970-01-01 00:00:00"
    assert result["r1"].review_date == date(1970, 1, 1)
    assert result["r2"].review_datetime == "1970-01-02 00:00:00"
    assert result["r2"].review_date == date(1970, 1, 2)


def test_add_review_features_counts_three_words(spark):
    reviews = make_cleaned_reviews(spark, [("r1", "Battery died quickly", 1)])

    result = add_review_features(reviews).first()

    assert result.review_word_count == 3


def test_add_review_features_counts_characters_exactly(spark):
    text = "Good sound quality"
    reviews = make_cleaned_reviews(spark, [("r1", text, 5)])

    result = add_review_features(reviews).first()

    assert result.review_length_chars == len(text)


def test_add_review_features_counts_one_word(spark):
    reviews = make_cleaned_reviews(spark, [("r1", "Excellent", 5)])

    result = add_review_features(reviews).first()

    assert result.review_word_count == 1


def test_add_review_features_counts_punctuation_as_characters_not_words(spark):
    text = "Wow! Really good."
    reviews = make_cleaned_reviews(spark, [("r1", text, 5)])

    result = add_review_features(reviews).first()

    assert result.review_length_chars == len(text)
    assert result.review_word_count == 3


def test_add_review_features_calculates_each_row_independently(spark):
    reviews = make_cleaned_reviews(
        spark,
        [("r1", "Great", 5), ("r2", "Not very good", 2)],
    )

    result = {row.review_id: row for row in add_review_features(reviews).collect()}

    assert result["r1"].review_length_chars == 5
    assert result["r1"].review_word_count == 1
    assert result["r2"].review_length_chars == 13
    assert result["r2"].review_word_count == 3


def test_add_review_features_preserves_cleaned_review_text(spark):
    text = "Still has punctuation!"
    reviews = make_cleaned_reviews(spark, [("r1", text, 4)])

    result = add_review_features(reviews).first()

    assert result.cleaned_review_text == text


def test_add_review_features_preserves_unrelated_columns(spark):
    reviews = make_cleaned_reviews(spark, [("r1", "Good", 5)])

    result = add_review_features(reviews).first()

    assert result.review_id == "r1"
    assert result.rating == 5


def test_add_review_features_adds_integer_feature_columns(spark):
    reviews = make_cleaned_reviews(spark, [("r1", "Good sound", 5)])

    result = add_review_features(reviews)

    assert isinstance(result.schema["review_length_chars"].dataType, IntegerType)
    assert isinstance(result.schema["review_word_count"].dataType, IntegerType)


def test_product_history_first_review_has_no_prior_reviews(spark):
    reviews = make_product_reviews(spark, [("r1", "p1", 1000, 5.0)])

    result = add_product_historical_features(reviews).first()

    assert result.product_prior_review_count == 0
    assert result.product_prior_average_rating is None


def test_product_history_sequential_reviews(spark):
    reviews = make_product_reviews(
        spark,
        [("r1", "p1", 1000, 5.0), ("r2", "p1", 2000, 3.0), ("r3", "p1", 3000, 1.0)],
    )

    result = {row.review_id: row for row in add_product_historical_features(reviews).collect()}

    assert result["r1"].product_prior_review_count == 0
    assert result["r1"].product_prior_average_rating is None
    assert result["r2"].product_prior_review_count == 1
    assert result["r2"].product_prior_average_rating == 5.0
    assert result["r3"].product_prior_review_count == 2
    assert result["r3"].product_prior_average_rating == 4.0


def test_product_history_same_timestamp_reviews_do_not_see_each_other(spark):
    reviews = make_product_reviews(
        spark,
        [
            ("r1", "p1", 1000, 5.0),
            ("r2", "p1", 2000, 3.0),
            ("r3", "p1", 2000, 4.0),
            ("r4", "p1", 3000, 2.0),
        ],
    )

    result = {row.review_id: row for row in add_product_historical_features(reviews).collect()}

    assert result["r2"].product_prior_review_count == 1
    assert result["r2"].product_prior_average_rating == 5.0
    assert result["r3"].product_prior_review_count == 1
    assert result["r3"].product_prior_average_rating == 5.0
    assert result["r4"].product_prior_review_count == 3
    assert result["r4"].product_prior_average_rating == 4.0


def test_product_history_is_independent_for_each_product(spark):
    reviews = make_product_reviews(
        spark,
        [
            ("r1", "p1", 1000, 5.0),
            ("r2", "p2", 1000, 1.0),
            ("r3", "p1", 2000, 3.0),
            ("r4", "p2", 2000, 4.0),
        ],
    )

    result = {row.review_id: row for row in add_product_historical_features(reviews).collect()}

    assert result["r3"].product_prior_review_count == 1
    assert result["r3"].product_prior_average_rating == 5.0
    assert result["r4"].product_prior_review_count == 1
    assert result["r4"].product_prior_average_rating == 1.0


def test_product_history_excludes_current_rating(spark):
    reviews = make_product_reviews(
        spark,
        [("r1", "p1", 1000, 2.0), ("r2", "p1", 2000, 10.0)],
    )

    result = {row.review_id: row for row in add_product_historical_features(reviews).collect()}

    assert result["r2"].product_prior_average_rating == 2.0


def test_product_history_averages_multiple_earlier_reviews(spark):
    reviews = make_product_reviews(
        spark,
        [
            ("r1", "p1", 1000, 1.0),
            ("r2", "p1", 2000, 2.0),
            ("r3", "p1", 3000, 3.0),
            ("r4", "p1", 4000, 5.0),
        ],
    )

    result = {row.review_id: row for row in add_product_historical_features(reviews).collect()}

    assert result["r4"].product_prior_review_count == 3
    assert result["r4"].product_prior_average_rating == 2.0


def test_product_history_preserves_existing_columns(spark):
    source = ("r1", "p1", 1000, 4.0)
    reviews = make_product_reviews(spark, [source])

    result = add_product_historical_features(reviews).first()

    assert (result.review_id, result.product_id, result.timestamp, result.rating) == source


def test_product_history_adds_integer_prior_count_column(spark):
    reviews = make_product_reviews(spark, [("r1", "p1", 1000, 5.0)])

    result = add_product_historical_features(reviews)

    assert isinstance(result.schema["product_prior_review_count"].dataType, LongType)


def test_product_history_adds_floating_point_prior_average_column(spark):
    reviews = make_product_reviews(spark, [("r1", "p1", 1000, 5.0)])

    result = add_product_historical_features(reviews)

    assert isinstance(result.schema["product_prior_average_rating"].dataType, DoubleType)


def test_preprocess_reviews_runs_full_pipeline_on_mixed_input(spark):
    reviews = make_canonical_spark_reviews(
        spark,
        [
            ("r1", "p1", "old", "discarded", 1.0, 1000, 1, False),
            ("r1", "p1", "kept", "  Great<br /> sound  ", 5.0, 1000, 8, True),
            ("r2", "p1", "second", "Battery   died\nquickly", 3.0, 2000, 2, True),
            ("r3", "p1", "third", "Not bad", 1.0, 3000, 0, False),
        ],
    )

    result = {row.review_id: row for row in preprocess_reviews(reviews).collect()}

    assert set(result) == {"r1", "r2", "r3"}
    assert result["r1"].helpful_votes == 8
    assert result["r1"].cleaned_review_text == "Great sound"
    assert result["r1"].review_length_chars == len("Great sound")
    assert result["r1"].review_word_count == 2
    assert result["r2"].cleaned_review_text == "Battery died quickly"
    assert result["r2"].product_prior_review_count == 1
    assert result["r2"].product_prior_average_rating == 5.0
    assert result["r3"].product_prior_review_count == 2
    assert result["r3"].product_prior_average_rating == 4.0


def test_preprocess_reviews_deduplicates_before_product_history(spark):
    reviews = make_canonical_spark_reviews(
        spark,
        [
            ("r1", "p1", "old", "old duplicate", 1.0, 1000, 1, False),
            ("r1", "p1", "kept", "kept duplicate", 5.0, 1000, 9, True),
            ("r2", "p1", "later", "later review", 3.0, 2000, 0, True),
        ],
    )

    result = {row.review_id: row for row in preprocess_reviews(reviews).collect()}

    assert len(result) == 2
    assert result["r2"].product_prior_review_count == 1
    assert result["r2"].product_prior_average_rating == 5.0


def test_preprocess_reviews_cleans_text_before_text_features(spark):
    raw_text = "  Great<br />   battery\n\tlife!  "
    reviews = make_canonical_spark_reviews(
        spark,
        [("r1", "p1", "title", raw_text, 5.0, 1000, 1, True)],
    )

    result = preprocess_reviews(reviews).first()

    assert result.cleaned_review_text == "Great battery life!"
    assert result.review_length_chars == len("Great battery life!")
    assert result.review_word_count == 3


def test_preprocess_reviews_removes_unusable_text_before_product_history(spark):
    reviews = make_canonical_spark_reviews(
        spark,
        [
            ("r1", "p1", "empty", "<br /><br />", 1.0, 1000, 0, False),
            ("r2", "p1", "valid", "Useful review", 5.0, 2000, 0, True),
        ],
    )

    result = preprocess_reviews(reviews).collect()

    assert len(result) == 1
    assert result[0].review_id == "r2"
    assert result[0].product_prior_review_count == 0
    assert result[0].product_prior_average_rating is None


def test_preprocess_reviews_output_contains_raw_and_derived_columns(spark):
    reviews = make_canonical_spark_reviews(
        spark,
        [("r1", "p1", "title", "Good review", 5.0, 1000, 1, True)],
    )

    result = preprocess_reviews(reviews)

    assert set(result.columns) == {
        "review_id",
        "product_id",
        "review_title",
        "review_text",
        "rating",
        "timestamp",
        "helpful_votes",
        "verified_purchase",
        "cleaned_review_text",
        "review_datetime",
        "review_date",
        "review_length_chars",
        "review_word_count",
        "product_prior_review_count",
        "product_prior_average_rating",
    }


def test_preprocess_reviews_preserves_original_text_and_timestamp(spark):
    raw_text = "  Good<br />review  "
    timestamp = 1704067200123
    reviews = make_canonical_spark_reviews(
        spark,
        [("r1", "p1", "title", raw_text, 5.0, timestamp, 1, True)],
    )

    result = preprocess_reviews(reviews).first()

    assert result.review_text == raw_text
    assert result.timestamp == timestamp


def test_preprocess_reviews_same_timestamp_reviews_do_not_see_each_other(spark):
    reviews = make_canonical_spark_reviews(
        spark,
        [
            ("r1", "p1", "first", "First", 5.0, 1000, 0, True),
            ("r2", "p1", "second", "Second", 3.0, 2000, 0, True),
            ("r3", "p1", "third", "Third", 4.0, 2000, 0, True),
        ],
    )

    result = {row.review_id: row for row in preprocess_reviews(reviews).collect()}

    assert result["r2"].product_prior_review_count == 1
    assert result["r2"].product_prior_average_rating == 5.0
    assert result["r3"].product_prior_review_count == 1
    assert result["r3"].product_prior_average_rating == 5.0
