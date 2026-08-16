import argparse
import logging
import shutil
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from quality_intelligence.data.output import replace_output_directory
from quality_intelligence.data.weak_labeling_spark import add_weak_labels

DEFAULT_INPUT_PATH = Path("data/processed/reviews_clean.parquet")
DEFAULT_OUTPUT_PATH = Path("data/processed/reviews_labeled.parquet")
LABELS_TO_INSPECT = (
    "FUNCTIONALITY",
    "NO_COMPLAINT",
    "FIT_COMPATIBILITY",
    "BUILD_QUALITY",
    "USABILITY_SETUP",
    "SHIPPING",
)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply weak labels to processed reviews.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--master", default="local[*]")
    parser.add_argument("--driver-memory", default="4g")
    return parser.parse_args()


def show_labeling_summary(labeled_df: DataFrame, sample_size: int) -> None:
    LOGGER.info("Total reviews: %s", f"{labeled_df.count():,}")
    labeled_df.groupBy("weak_label_status").count().orderBy(F.desc("count")).show(truncate=False)
    labeled_df.filter(F.col("weak_label_status") == "LABELED").groupBy(
        "weak_label"
    ).count().orderBy(F.desc("count")).show(truncate=False)

    for label in LABELS_TO_INSPECT:
        LOGGER.info("Sample label: %s", label)
        labeled_df.filter(F.col("weak_label") == label).orderBy(F.rand(seed=42)).select(
            "cleaned_review_text", "rating", "weak_label"
        ).limit(sample_size).show(sample_size, truncate=False)


def write_labeled_reviews(labeled_df: DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged_path = output_path.with_name(f"{output_path.name}.part")
    if staged_path.exists():
        shutil.rmtree(staged_path)
    try:
        labeled_df.write.parquet(str(staged_path))
        replace_output_directory(staged_path, output_path)
    except Exception:
        if staged_path.exists():
            shutil.rmtree(staged_path)
        raise


def run_labeling(
    spark: SparkSession, input_path: Path, output_path: Path, sample_size: int
) -> None:
    LOGGER.info("Reading processed reviews from: %s", input_path)
    labeled_df = add_weak_labels(spark.read.parquet(str(input_path))).cache()
    try:
        show_labeling_summary(labeled_df, sample_size)
        write_labeled_reviews(labeled_df, output_path)
        written_df = spark.read.parquet(str(output_path))
        LOGGER.info("Written rows: %s", f"{written_df.count():,}")
        written_df.printSchema()
    finally:
        labeled_df.unpersist()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    spark = (
        SparkSession.builder.appName("product-quality-weak-labeling")
        .master(args.master)
        .config("spark.driver.memory", args.driver_memory)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        run_labeling(spark, args.input, args.output, args.sample_size)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
