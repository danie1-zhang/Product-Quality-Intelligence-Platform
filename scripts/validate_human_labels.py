from pathlib import Path

import pandas as pd


HUMAN_LABELS_PATH = Path(
    "data/annotation/human_labeled_reviews.csv"
)

METADATA_PATH = Path(
    "data/annotation/human_label_candidates_metadata.csv"
)

MERGED_OUTPUT_PATH = Path(
    "data/annotation/human_labeled_reviews_with_metadata.csv"
)


VALID_LABELS = {
    "NO_COMPLAINT",
    "FUNCTIONALITY",
    "BUILD_QUALITY",
    "SHIPPING",
    "FIT_COMPATIBILITY",
    "USABILITY_SETUP",
    "OTHER",
}


def validate_human_labels(df: pd.DataFrame) -> None:
    print("Validating human-labeled dataset...\n")

    expected_columns = {
        "annotation_id",
        "review_id",
        "cleaned_review_text",
        "rating",
        "human_label",
        "notes",
    }

    missing_columns = expected_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {sorted(missing_columns)}"
        )

    print(f"Total rows: {len(df):,}")

    if len(df) != 500:
        raise ValueError(
            f"Expected 500 rows, found {len(df)}"
        )

    if df["annotation_id"].duplicated().any():
        duplicates = df.loc[
            df["annotation_id"].duplicated(keep=False),
            "annotation_id",
        ].tolist()

        raise ValueError(
            f"Duplicate annotation_id values found: {duplicates}"
        )

    print("annotation_id uniqueness: PASS")

    if df["review_id"].duplicated().any():
        duplicates = df.loc[
            df["review_id"].duplicated(keep=False),
            "review_id",
        ].tolist()

        raise ValueError(
            f"Duplicate review_id values found: {duplicates}"
        )

    print("review_id uniqueness: PASS")

    missing_labels = df["human_label"].isna() | (
        df["human_label"].astype(str).str.strip() == ""
    )

    if missing_labels.any():
        bad_ids = df.loc[
            missing_labels,
            "annotation_id",
        ].tolist()

        raise ValueError(
            f"Missing human labels for annotation IDs: {bad_ids}"
        )

    print("Missing labels: PASS")

    observed_labels = set(df["human_label"].unique())

    invalid_labels = observed_labels - VALID_LABELS

    if invalid_labels:
        raise ValueError(
            f"Invalid human labels found: {sorted(invalid_labels)}"
        )

    print("Allowed labels: PASS")

    print("\nHuman label distribution:")

    distribution = (
        df["human_label"]
        .value_counts()
        .rename_axis("human_label")
        .reset_index(name="count")
    )

    print(distribution.to_string(index=False))


def validate_metadata(
    human_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> None:
    print("\nValidating metadata alignment...\n")

    required_metadata_columns = {
        "annotation_id",
        "review_id",
        "cleaned_review_text",
        "rating",
        "weak_label",
        "weak_label_status",
    }

    missing_columns = (
        required_metadata_columns - set(metadata_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Metadata missing columns: {sorted(missing_columns)}"
        )

    if len(metadata_df) != 500:
        raise ValueError(
            f"Expected 500 metadata rows, found {len(metadata_df)}"
        )

    if metadata_df["annotation_id"].duplicated().any():
        raise ValueError(
            "Duplicate annotation_id values found in metadata"
        )

    if metadata_df["review_id"].duplicated().any():
        raise ValueError(
            "Duplicate review_id values found in metadata"
        )

    human_ids = set(human_df["review_id"])
    metadata_ids = set(metadata_df["review_id"])

    if human_ids != metadata_ids:
        missing_from_metadata = human_ids - metadata_ids
        missing_from_human = metadata_ids - human_ids

        raise ValueError(
            "Human and metadata review IDs do not match.\n"
            f"Missing from metadata: {len(missing_from_metadata)}\n"
            f"Missing from human labels: {len(missing_from_human)}"
        )

    print("Metadata row count: PASS")
    print("Metadata uniqueness: PASS")
    print("Review ID alignment: PASS")


def merge_datasets(
    human_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    metadata_subset = metadata_df[
        [
            "annotation_id",
            "review_id",
            "weak_label",
            "weak_label_status",
        ]
    ]

    merged_df = human_df.merge(
        metadata_subset,
        on=["annotation_id", "review_id"],
        how="left",
        validate="one_to_one",
    )

    if len(merged_df) != len(human_df):
        raise ValueError(
            "Merged row count does not match human dataset"
        )

    return merged_df


def print_weak_label_analysis(
    merged_df: pd.DataFrame,
) -> None:
    print("\nWeak-label status distribution:")

    print(
        merged_df["weak_label_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    weak_labeled = merged_df[
        merged_df["weak_label_status"] == "LABELED"
    ].copy()

    weak_labeled["weak_label_correct"] = (
        weak_labeled["weak_label"]
        == weak_labeled["human_label"]
    )

    overall_accuracy = (
        weak_labeled["weak_label_correct"].mean()
    )

    print(
        "\nWeak-label agreement with human ground truth "
        f"(LABELED rows only): {overall_accuracy:.2%}"
    )

    print("\nPer-class weak-label precision:")

    per_class = (
        weak_labeled
        .groupby("weak_label")
        .agg(
            total=("weak_label", "size"),
            correct=("weak_label_correct", "sum"),
        )
    )

    per_class["precision"] = (
        per_class["correct"] / per_class["total"]
    )

    print(per_class.to_string())

    print("\nHuman labels for weak ABSTAIN rows:")

    abstain_distribution = (
        merged_df[
            merged_df["weak_label_status"] == "ABSTAIN"
        ]["human_label"]
        .value_counts()
    )

    print(abstain_distribution.to_string())

    print("\nHuman labels for weak CONFLICT rows:")

    conflict_distribution = (
        merged_df[
            merged_df["weak_label_status"] == "CONFLICT"
        ]["human_label"]
        .value_counts()
    )

    print(conflict_distribution.to_string())


def main() -> None:
    human_df = pd.read_csv(HUMAN_LABELS_PATH)

    metadata_df = pd.read_csv(METADATA_PATH)

    validate_human_labels(human_df)

    validate_metadata(
        human_df,
        metadata_df,
    )

    merged_df = merge_datasets(
        human_df,
        metadata_df,
    )

    MERGED_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged_df.to_csv(
        MERGED_OUTPUT_PATH,
        index=False,
    )

    print(
        "\nMerged dataset written to: "
        f"{MERGED_OUTPUT_PATH}"
    )

    print_weak_label_analysis(
        merged_df
    )

    print(
        "\nHuman evaluation dataset validation: PASS"
    )


if __name__ == "__main__":
    main()