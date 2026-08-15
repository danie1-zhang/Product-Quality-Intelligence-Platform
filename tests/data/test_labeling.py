import pytest

from quality_intelligence.data.labeling import (
    FUNCTIONALITY_PATTERNS,
    ComplaintLabel,
    functionality_label,
)


def test_complaint_label_exposes_expected_labels():
    assert [label.value for label in ComplaintLabel] == [
        "NO_COMPLAINT",
        "FUNCTIONALITY",
        "BUILD_QUALITY",
        "SHIPPING",
        "FIT_COMPATIBILITY",
        "USABILITY_SETUP",
        "OTHER",
    ]


@pytest.mark.parametrize("pattern", FUNCTIONALITY_PATTERNS)
def test_each_functionality_pattern_returns_functionality_label(pattern):
    assert functionality_label(f"The device {pattern} after one day.") is ComplaintLabel.FUNCTIONALITY


def test_functionality_matching_is_case_insensitive():
    assert functionality_label("THE DEVICE WON'T TURN ON.") is ComplaintLabel.FUNCTIONALITY


def test_unrelated_positive_text_returns_none():
    assert functionality_label("This product is excellent and works perfectly.") is None


def test_unrelated_negative_text_returns_none():
    assert functionality_label("The color is disappointing and the price is too high.") is None


@pytest.mark.parametrize(
    "text",
    [
        "The device is working well.",
        "The charging cable is long.",
        "It stopped raining before delivery.",
        "The microphone quality is excellent.",
    ],
)
def test_partial_or_unrelated_wording_does_not_match(text):
    assert functionality_label(text) is None
