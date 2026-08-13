from huggingface_hub import HfFileSystem

from quality_intelligence.data.ingest import ingest_headphone_reviews

REVISION = "ac9d3ad3342d6f00bf6ad8caa2668a3f830e2dee"
METADATA_GLOB = (
    f"datasets/McAuley-Lab/Amazon-Reviews-2023@{REVISION}/raw_meta_Electronics/*.parquet"
)
REVIEW_GLOB = (
    f"datasets/McAuley-Lab/Amazon-Reviews-2023@{REVISION}/raw_review_Electronics/*.parquet"
)
OUTPUT_PATH = "data/raw/headphone_reviews.parquet"


def main():
    fs = HfFileSystem()

    metadata_paths = fs.glob(METADATA_GLOB)
    review_paths = fs.glob(REVIEW_GLOB)

    print(f"Metadata files found: {len(metadata_paths)}")
    print(f"Review files found: {len(review_paths)}")

    print("First metadata paths:")
    for path in metadata_paths[:2]:
        print(path)

    print("First review paths:")
    for path in review_paths[:2]:
        print(path)

    ingest_headphone_reviews(
        fs=fs,
        metadata_paths=metadata_paths,
        review_paths=review_paths,
        output_path=OUTPUT_PATH,
        read_batch_size=5000,
        write_batch_size=5000,
    )

    print(f"Output written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
