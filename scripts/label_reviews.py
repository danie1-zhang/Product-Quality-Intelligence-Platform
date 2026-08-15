from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructField, StructType, StringType

from quality_intelligence.data.labeling import label_review


INPUT_PATH = "data/processed/reviews_clean.parquet"
OUTPUT_PATH = "data/processed/reviews_labeled.parquet"

LABEL_RESULT_SCHEMA = StructType([
    StructField("weak_label", StringType(), True),
    StructField("weak_label_status", StringType(), False),
])


def label_review_for_spark(text: str, rating: float) -> tuple[str | None, str]:
    label, status = label_review(text, rating)
    if label:
        str_label = label.value
    else:
        str_label = None
    str_status = status.value
    return (str_label, str_status)


label_review_udf = F.udf(label_review_for_spark, LABEL_RESULT_SCHEMA)

def add_weak_labels(df):
    df = df.withColumn("label_result", label_review_udf(F.col("cleaned_review_text"), F.col("rating")))
    df = df.withColumn("weak_label", F.col("label_result.weak_label"))
    df = df.withColumn("weak_label_status", F.col("label_result.weak_label_status"))
    return df.drop("label_result")


LABELS_TO_INSPECT = [
    "FUNCTIONALITY",
    "NO_COMPLAINT",
    "FIT_COMPATIBILITY",
    "BUILD_QUALITY",
    "USABILITY_SETUP",
    "SHIPPING",
]


def main():
    spark = (
        SparkSession.builder
        .appName("product-quality-weak-labeling")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print(f"Reading processed reviews from: {INPUT_PATH}")
    df = spark.read.parquet(INPUT_PATH)

    print("Applying weak-labeling rules...")

    # Cache because we will reuse this labeled DataFrame for
    # several actions below.
    labeled_df = add_weak_labels(df).cache()

    # First action: computes the labels and populates the cache.
    total_count = labeled_df.count()

    print(f"\nTotal reviews: {total_count:,}")

    print("\nWeak-label status distribution:")
    (
        labeled_df
        .groupBy("weak_label_status")
        .count()
        .orderBy(F.desc("count"))
        .show(truncate=False)
    )

    print("\nLabel distribution among labeled reviews:")
    (
        labeled_df
        .filter(F.col("weak_label_status") == "LABELED")
        .groupBy("weak_label")
        .count()
        .orderBy(F.desc("count"))
        .show(truncate=False)
    )

    print("\nRandom examples by weak label:")

    for label in LABELS_TO_INSPECT:
        print(f"\n--- {label} ---")

        (
            labeled_df
            .filter(F.col("weak_label") == label)
            .orderBy(F.rand(seed=42))
            .select(
                "cleaned_review_text",
                "rating",
                "weak_label",
            )
            .limit(10)
            .show(
                10,
                truncate=False,
            )
        )

    print(f"\nWriting labeled reviews to: {OUTPUT_PATH}")
    labeled_df.write.mode("overwrite").parquet(OUTPUT_PATH)
    print("Write complete.")



    written_df = spark.read.parquet(OUTPUT_PATH)

    print(f"Written rows: {written_df.count():,}")
    written_df.printSchema()

    written_df.groupBy("weak_label_status").count().show()

    labeled_df.unpersist()

    spark.stop()


if __name__ == "__main__":
    main()