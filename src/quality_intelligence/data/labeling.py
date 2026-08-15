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
    "microphone doesn't work"
)


def functionality_label(text: str) -> ComplaintLabel | None:
    normalized_text = text.lower()
    for pattern in FUNCTIONALITY_PATTERNS:
        if pattern in normalized_text:
            return ComplaintLabel.FUNCTIONALITY
    return None