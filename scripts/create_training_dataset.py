from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


LABELED_REVIEWS_PATH = "data/processed/reviews_labeled.parquet"
HUMAN_EVAL_PATH = Path("data/annotation/human_labeled_reviews.csv")
OUTPUT_PATH = "data/processed/reviews_weak_train.parquet"


def main() -> None:
    spark = (
        SparkSession.builder
        .appName("create-training-dataset")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print(f"Reading weak-labeled reviews from: {LABELED_REVIEWS_PATH}")
    df = spark.read.parquet(LABELED_REVIEWS_PATH)

    print(f"Total rows before filtering: {df.count():,}")

    # Only accepted weak labels are eligible for supervised training.
    weak_train_candidates = df.filter(
        F.col("weak_label_status") == "LABELED"
    )

    candidate_count = weak_train_candidates.count()

    print(
        f"Weak-labeled training candidates: {candidate_count:,}"
    )

    # Read the trusted human evaluation IDs.
    human_eval_pdf = pd.read_csv(
        HUMAN_EVAL_PATH,
        usecols=["review_id"],
    )

    if human_eval_pdf["review_id"].duplicated().any():
        raise ValueError(
            "Duplicate review_id values found in human evaluation set"
        )

    eval_ids_df = spark.createDataFrame(
        human_eval_pdf
    ).select("review_id")

    print(
        f"Human evaluation review IDs to exclude: "
        f"{eval_ids_df.count():,}"
    )

    # Anti-join removes any review appearing in the evaluation set.
    train_df = weak_train_candidates.join(
        eval_ids_df,
        on="review_id",
        how="left_anti",
    ).cache()

    train_count = train_df.count()

    removed_count = candidate_count - train_count

    print(
        f"Evaluation rows removed from training candidates: "
        f"{removed_count:,}"
    )

    print(
        f"Final weak-supervised training rows: "
        f"{train_count:,}"
    )

    # Explicit leakage validation.
    leakage_count = (
        train_df
        .join(
            eval_ids_df,
            on="review_id",
            how="inner",
        )
        .count()
    )

    if leakage_count != 0:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: "
            f"{leakage_count} evaluation reviews remain in training"
        )

    print("Human evaluation leakage check: PASS")

    # Basic integrity checks.
    null_label_count = train_df.filter(
        F.col("weak_label").isNull()
    ).count()

    if null_label_count != 0:
        raise ValueError(
            f"Found {null_label_count} null weak labels "
            "in training dataset"
        )

    invalid_status_count = train_df.filter(
        F.col("weak_label_status") != "LABELED"
    ).count()

    if invalid_status_count != 0:
        raise ValueError(
            f"Found {invalid_status_count} non-LABELED rows "
            "in training dataset"
        )

    print("Training label integrity: PASS")

    print("\nTraining class distribution:")

    (
        train_df
        .groupBy("weak_label")
        .count()
        .orderBy(F.desc("count"))
        .show(50, truncate=False)
    )

    print(f"\nWriting training dataset to: {OUTPUT_PATH}")

    (
        train_df
        .write
        .mode("overwrite")
        .parquet(OUTPUT_PATH)
    )

    # Re-read the actual written artifact and validate it.
    written_df = spark.read.parquet(OUTPUT_PATH)

    written_count = written_df.count()

    if written_count != train_count:
        raise ValueError(
            "Written training row count does not match "
            f"in-memory count: {written_count} != {train_count}"
        )

    written_leakage_count = (
        written_df
        .join(
            eval_ids_df,
            on="review_id",
            how="inner",
        )
        .count()
    )

    if written_leakage_count != 0:
        raise ValueError(
            "Evaluation leakage detected in written training artifact"
        )

    print("\nWritten training dataset validation: PASS")
    print(f"Written rows: {written_count:,}")
    print("Evaluation IDs present: 0")

    train_df.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()