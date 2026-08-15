import argparse
import logging
from pathlib import Path

from fsspec.implementations.local import LocalFileSystem
from huggingface_hub import HfFileSystem

from quality_intelligence.data.cache import cache_remote_shards
from quality_intelligence.data.ingest import ingest_headphone_reviews

REVISION = "ac9d3ad3342d6f00bf6ad8caa2668a3f830e2dee"
METADATA_GLOB = (
    f"datasets/McAuley-Lab/Amazon-Reviews-2023@{REVISION}/raw_meta_Electronics/*.parquet"
)
REVIEW_GLOB = (
    f"datasets/McAuley-Lab/Amazon-Reviews-2023@{REVISION}/raw_review_Electronics/*.parquet"
)
OUTPUT_PATH = "data/raw/headphone_reviews.parquet"
DEFAULT_CACHE_PATH = Path("data/cache/amazon_reviews_2023")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse paths while preserving repository-local defaults."""
    parser = argparse.ArgumentParser(description="Ingest Amazon headphone reviews.")
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    remote_fs = HfFileSystem()

    metadata_paths = remote_fs.glob(METADATA_GLOB)
    review_paths = remote_fs.glob(REVIEW_GLOB)

    LOGGER.info("Metadata files found: %s", len(metadata_paths))
    LOGGER.info("Review files found: %s", len(review_paths))

    local_metadata_paths = cache_remote_shards(
        remote_fs,
        metadata_paths,
        args.cache_dir / "metadata",
        "metadata",
    )
    local_review_paths = cache_remote_shards(
        remote_fs,
        review_paths,
        args.cache_dir / "reviews",
        "review",
    )

    ingest_headphone_reviews(
        fs=LocalFileSystem(),
        metadata_paths=local_metadata_paths,
        review_paths=local_review_paths,
        output_path=args.output,
        read_batch_size=5000,
        write_batch_size=5000,
    )

    LOGGER.info("Output written to: %s", args.output)


if __name__ == "__main__":
    main()
