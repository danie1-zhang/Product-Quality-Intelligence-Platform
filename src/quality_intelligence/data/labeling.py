from enum import StrEnum


class ComplaintLabel(StrEnum):
    """
    Supported complaint categories.
    """

    NO_COMPLAINT = "NO_COMPLAINT"
    FUNCTIONALITY = "FUNCTIONALITY"
    BUILD_QUALITY = "BUILD_QUALITY"
    SHIPPING = "SHIPPING"
    FIT_COMPATIBILITY = "FIT_COMPATIBILITY"
    USABILITY_SETUP = "USABILITY_SETUP"
    OTHER = "OTHER"


class WeakLabelStatus(StrEnum):
    LABELED = "LABELED"
    ABSTAIN = "ABSTAIN"
    CONFLICT = "CONFLICT"


# Weak Labels
FUNCTIONALITY_PATTERNS = (
    "stopped working",
    "not working",
    "doesn't work",
    "won't turn on",
    "won't charge",
    "not charging",
    "keeps disconnecting",
    "won't connect",
    "microphone doesn't work",
)

BUILD_QUALITY_PATTERNS = (
    "headband cracked",
    "hinge broke",
    "ear pad fell off",
    "fell apart",
    "wire broke",
    "one earbud broke",
)


SHIPPING_PATTERNS = (
    "arrived broken",
    "box damaged",
    "damaged box",
    "box was crushed",
    "missing packaging",
)


FIT_COMPATIBILITY_PATTERNS = (
    "doesn't fit",
    "keeps falling out",
    "keep falling out",
    "start falling out",
    "starts falling out",
    "too tight",
    "too small",
    "too loose",
    "incompatible with",
    "not compatible with",
    "don't fit",
)


USABILITY_SETUP_PATTERNS = (
    "no instructions",
    "instruction unclear",
    "can't setup",
    "pairing process is confusing",
    "instructions confusing",
    "setup confusing",
)


POSITIVE_PATTERNS = (
    "works great",
    "great purchase",
    "no issues",
    "works perfectly",
    "awesome product",
    "seamless setup",
)


def _label_helper(
    text: str, patterns: tuple[str, ...], label: ComplaintLabel
) -> ComplaintLabel | None:
    normalized_text = text.lower()
    for pattern in patterns:
        if pattern in normalized_text:
            return label
    return None


def functionality_label(text: str) -> ComplaintLabel | None:
    return _label_helper(text, FUNCTIONALITY_PATTERNS, ComplaintLabel.FUNCTIONALITY)


def build_quality_label(text: str) -> ComplaintLabel | None:
    return _label_helper(text, BUILD_QUALITY_PATTERNS, ComplaintLabel.BUILD_QUALITY)


def shipping_label(text: str) -> ComplaintLabel | None:
    return _label_helper(text, SHIPPING_PATTERNS, ComplaintLabel.SHIPPING)


def fit_compatibility_label(text: str) -> ComplaintLabel | None:
    return _label_helper(text, FIT_COMPATIBILITY_PATTERNS, ComplaintLabel.FIT_COMPATIBILITY)


def usability_setup_label(text: str) -> ComplaintLabel | None:
    return _label_helper(text, USABILITY_SETUP_PATTERNS, ComplaintLabel.USABILITY_SETUP)


def no_complaint_label(
    text: str,
    rating: float,
) -> ComplaintLabel | None:
    normalized_text = text.lower()
    if rating >= 4:
        for pattern in POSITIVE_PATTERNS:
            if pattern in normalized_text:
                return ComplaintLabel.NO_COMPLAINT
    return None


# Label Functions
LABEL_FUNCTIONS = (
    functionality_label,
    build_quality_label,
    shipping_label,
    fit_compatibility_label,
    usability_setup_label,
)


def label_review(text: str, rating: float) -> tuple[ComplaintLabel | None, WeakLabelStatus]:
    labels: set[ComplaintLabel] = set()
    for labeling_function in LABEL_FUNCTIONS:
        label = labeling_function(text)
        if label:
            labels.add(label)

    label = no_complaint_label(text, rating)
    if label:
        labels.add(label)

    if len(labels) == 1:
        return (next(iter(labels)), WeakLabelStatus.LABELED)
    if not labels:
        return (None, WeakLabelStatus.ABSTAIN)
    return (None, WeakLabelStatus.CONFLICT)
