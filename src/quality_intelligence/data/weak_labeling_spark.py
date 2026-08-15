from collections.abc import Sequence

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from quality_intelligence.data.labeling import (
    BUILD_QUALITY_PATTERNS,
    FIT_COMPATIBILITY_PATTERNS,
    FUNCTIONALITY_PATTERNS,
    POSITIVE_PATTERNS,
    SHIPPING_PATTERNS,
    USABILITY_SETUP_PATTERNS,
    ComplaintLabel,
    WeakLabelStatus,
)


def _matches_any(text: Column, patterns: Sequence[str]) -> Column:
    matches = [text.contains(pattern) for pattern in patterns]
    result = matches[0]
    for match in matches[1:]:
        result = result | match
    return F.coalesce(result, F.lit(False))


def add_weak_labels(df: DataFrame) -> DataFrame:
    """Add null-safe weak-label and resolution-status columns using Spark expressions."""
    normalized_text = F.lower(F.col("cleaned_review_text"))
    category_rules = (
        (FUNCTIONALITY_PATTERNS, ComplaintLabel.FUNCTIONALITY),
        (BUILD_QUALITY_PATTERNS, ComplaintLabel.BUILD_QUALITY),
        (SHIPPING_PATTERNS, ComplaintLabel.SHIPPING),
        (FIT_COMPATIBILITY_PATTERNS, ComplaintLabel.FIT_COMPATIBILITY),
        (USABILITY_SETUP_PATTERNS, ComplaintLabel.USABILITY_SETUP),
    )
    label_candidates = [
        F.when(_matches_any(normalized_text, patterns), F.lit(label.value))
        for patterns, label in category_rules
    ]
    positive_match = _matches_any(normalized_text, POSITIVE_PATTERNS) & F.coalesce(
        F.col("rating") >= F.lit(4), F.lit(False)
    )
    label_candidates.append(F.when(positive_match, F.lit(ComplaintLabel.NO_COMPLAINT.value)))

    fired_labels = F.array_distinct(
        F.filter(F.array(*label_candidates), lambda label: label.isNotNull())
    )
    label_count = F.size(fired_labels)
    weak_label = F.when(label_count == 1, F.element_at(fired_labels, 1)).otherwise(
        F.lit(None).cast("string")
    )
    status = (
        F.when(label_count == 1, F.lit(WeakLabelStatus.LABELED.value))
        .when(label_count == 0, F.lit(WeakLabelStatus.ABSTAIN.value))
        .otherwise(F.lit(WeakLabelStatus.CONFLICT.value))
    )
    return df.withColumn("weak_label", weak_label).withColumn("weak_label_status", status)
