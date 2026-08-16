from pathlib import Path
from typing import cast

import pandas as pd
from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F

INPUT_PATH = "data/processed/reviews_labeled.parquet"
OUTPUT_DIR = Path("data/annotation")

ANNOTATION_OUTPUT = OUTPUT_DIR / "human_label_candidates.csv"
METADATA_OUTPUT = OUTPUT_DIR / "human_label_candidates_metadata.csv"

RANDOM_SEED = 42


WEAK_LABEL_SAMPLE_COUNTS = {
    "FUNCTIONALITY": 60,
    "BUILD_QUALITY": 60,
    "SHIPPING": 60,
    "FIT_COMPATIBILITY": 60,
    "USABILITY_SETUP": 60,
    "NO_COMPLAINT": 60,
}

ABSTAIN_COUNT = 100
CONFLICT_COUNT = 20
GLOBAL_RANDOM_COUNT = 20


def sample_rows(df: DataFrame, condition: Column, count: int) -> DataFrame:
    return df.filter(condition).orderBy(F.rand(seed=RANDOM_SEED)).limit(count)


def main() -> None:
    spark = (
        SparkSession.builder.appName("create-human-label-sample")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

    df: DataFrame | None = None
    try:
        spark.sparkContext.setLogLevel("WARN")
        print(f"Reading labeled reviews from: {INPUT_PATH}")
        df = spark.read.parquet(INPUT_PATH).cache()
        print(f"Total available reviews: {df.count():,}")

        sampled_dfs: list[DataFrame] = []

        # 60 examples from each accepted weak-label class
        for label, count in WEAK_LABEL_SAMPLE_COUNTS.items():
            print(f"Sampling {count} {label} reviews")
            sampled_dfs.append(
                sample_rows(
                    df,
                    (F.col("weak_label_status") == "LABELED") & (F.col("weak_label") == label),
                    count,
                )
            )

        print(f"Sampling {ABSTAIN_COUNT} ABSTAIN reviews")
        sampled_dfs.append(sample_rows(df, F.col("weak_label_status") == "ABSTAIN", ABSTAIN_COUNT))

        print(f"Sampling {CONFLICT_COUNT} CONFLICT reviews")
        sampled_dfs.append(
            sample_rows(df, F.col("weak_label_status") == "CONFLICT", CONFLICT_COUNT)
        )

        # Completely random reviews reduce sampling bias.
        print(f"Sampling {GLOBAL_RANDOM_COUNT} reviews from the overall dataset")
        sampled_dfs.append(df.orderBy(F.rand(seed=RANDOM_SEED + 1)).limit(GLOBAL_RANDOM_COUNT))

        combined_df = sampled_dfs[0]
        for sampled_df in sampled_dfs[1:]:
            combined_df = combined_df.unionByName(sampled_df)

        # The global sample can overlap with a stratified sample.
        combined_df = combined_df.dropDuplicates(["review_id"]).withColumn(
            "annotation_id",
            F.row_number().over(Window.orderBy(F.rand(seed=RANDOM_SEED + 2))),
        )

        metadata_df = combined_df.select(
            "annotation_id",
            "review_id",
            "cleaned_review_text",
            "rating",
            "weak_label",
            "weak_label_status",
        )
        annotation_df = (
            combined_df.select("annotation_id", "review_id", "cleaned_review_text", "rating")
            .withColumn("human_label", F.lit(""))
            .withColumn("notes", F.lit(""))
        )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        annotation_pdf = cast(pd.DataFrame, annotation_df.orderBy("annotation_id").toPandas())
        metadata_pdf = cast(pd.DataFrame, metadata_df.orderBy("annotation_id").toPandas())
        annotation_pdf.to_csv(ANNOTATION_OUTPUT, index=False)
        metadata_pdf.to_csv(METADATA_OUTPUT, index=False)

        print("\nHuman annotation sample created.")
        print(f"Annotation sheet: {ANNOTATION_OUTPUT}")
        print(f"Metadata sheet:   {METADATA_OUTPUT}")
        print(f"Final candidate count: {len(annotation_pdf):,}")
        print("\nCandidate weak-label provenance:")
        metadata_df.groupBy("weak_label_status", "weak_label").count().orderBy(
            "weak_label_status", F.desc("count")
        ).show(50, truncate=False)
    finally:
        if df is not None:
            df.unpersist()
        spark.stop()


if __name__ == "__main__":
    main()
