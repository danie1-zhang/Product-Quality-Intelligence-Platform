from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def deduplicate_reviews(df: DataFrame) -> DataFrame:
    """
    Keep one row per 'review_id', preferring the greatest 'helpful_votes' value.
    """
    w = Window.partitionBy("review_id").orderBy(F.col("helpful_votes").desc())
    df = df.withColumn("rn", F.row_number().over(w))
    df = df.filter(F.col("rn") == 1)
    df = df.drop("rn")
    return df


def clean_review_text(df: DataFrame) -> DataFrame:
    """
    Normalize review text and remove rows without usable text.

    The cleaned text is stored in 'cleaned_review_text' after HTML line-break
    tags are replaced with spaces, repeated whitespace is collapsed, and leading
    and trailing whitespace is removed. Rows whose cleaned text is null or empty
    are excluded from the returned DataFrame.
    """
    df = df.withColumn("cleaned_review_text", F.regexp_replace(F.col("review_text"), r"(?i)<br\s*/?>", " "))
    df = df.withColumn("cleaned_review_text", F.regexp_replace(F.col("cleaned_review_text"), r"\s+", " "))
    df = df.withColumn("cleaned_review_text", F.trim(F.col("cleaned_review_text")))
    df = df.filter(F.col("cleaned_review_text").isNotNull() & (F.col("cleaned_review_text") != ""))
    return df


def normalize_timestamps(df: DataFrame) -> DataFrame:
    """
    Add normalized datetime and date columns while preserving the source timestamp.

    The 'timestamp' column is interpreted as Unix epoch milliseconds and converted
    to 'review_datetime'. The corresponding calendar date is stored in
    'review_date'. All original columns remain unchanged.
    """
    df = df.withColumn("review_datetime", F.timestamp_millis(F.col("timestamp")))
    df = df.withColumn("review_date", F.to_date(F.col("review_datetime")))
    return df


def add_review_features(df: DataFrame) -> DataFrame:
    """
    Add character-length and word-count features for cleaned review text.

    'review_length_chars' contains the number of characters in
    'cleaned_review_text', while 'review_word_count' contains the number of
    space-delimited words. All existing columns are preserved.
    """
    df = df.withColumn("review_length_chars", F.length(F.col("cleaned_review_text")))
    df = df.withColumn("review_word_count", F.size(F.split(F.col("cleaned_review_text"), " ")))
    return df


def add_product_historical_features(df: DataFrame) -> DataFrame:
    """
    Add historical review-count and average-rating features for each product.

    'product_prior_review_count' counts earlier reviews for the same 'product_id',
    while 'product_prior_average_rating' contains their average 'rating'. Only
    reviews with an earlier 'timestamp' contribute. All existing columns are preserved.
    """
    w = Window.partitionBy("product_id").orderBy("timestamp").rangeBetween(Window.unboundedPreceding, -1)
    df = df.withColumn("product_prior_review_count", F.count("*").over(w))
    df = df.withColumn("product_prior_average_rating", F.avg(F.col("rating")).over(w))
    return df


def preprocess_reviews(df: DataFrame) -> DataFrame:
    """
    Apply the full review preprocessing pipeline and return a processed DataFrame.

    Required canonical columns are consumed by the individual stages. Spark
    analysis errors surface normally when any required column is absent.
    """
    df = deduplicate_reviews(df)
    df = clean_review_text(df)
    df = normalize_timestamps(df)
    df = add_review_features(df)
    df = add_product_historical_features(df)
    return df
