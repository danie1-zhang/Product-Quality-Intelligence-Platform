import argparse
import logging
import shutil
from pathlib import Path

from pyspark.sql import SparkSession

from quality_intelligence.data.output import replace_output_directory
from quality_intelligence.data.preprocess import preprocess_reviews

DEFAULT_INPUT_PATH = Path("data/raw/headphone_reviews.parquet")
DEFAULT_OUTPUT_PATH = Path("data/processed/reviews_clean.parquet")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse paths while preserving repository-local defaults."""
    parser = argparse.ArgumentParser(description="Preprocess ingested headphone reviews.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staged_path = args.output.with_name(f"{args.output.name}.part")
    if staged_path.exists():
        shutil.rmtree(staged_path)

    spark = (
        SparkSession.builder.appName("preprocess-product-quality-reviews")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )

    try:
        raw_df = spark.read.parquet(str(args.input))
        processed_df = preprocess_reviews(raw_df)
        processed_df.write.parquet(str(staged_path))
        replace_output_directory(staged_path, args.output)
        LOGGER.info("Output written to: %s", args.output)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
