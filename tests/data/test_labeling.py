import pytest

from quality_intelligence.data.labeling import (
    BUILD_QUALITY_PATTERNS,
    FIT_COMPATIBILITY_PATTERNS,
    FUNCTIONALITY_PATTERNS,
    POSITIVE_PATTERNS,
    SHIPPING_PATTERNS,
    USABILITY_SETUP_PATTERNS,
    ComplaintLabel,
    _label_helper,
    build_quality_label,
    fit_compatibility_label,
    functionality_label,
    label_review,
    no_complaint_label,
    shipping_label,
    usability_setup_label,
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


def test_label_helper_returns_supplied_label_for_matching_pattern():
    assert (
        _label_helper(
            "The outer shell cracked yesterday.",
            ("shell cracked",),
            ComplaintLabel.BUILD_QUALITY,
        )
        is ComplaintLabel.BUILD_QUALITY
    )


def test_label_helper_matching_is_case_insensitive():
    assert (
        _label_helper("PAIRING FAILED", ("pairing failed",), ComplaintLabel.USABILITY_SETUP)
        is ComplaintLabel.USABILITY_SETUP
    )


def test_label_helper_returns_none_without_a_supplied_pattern():
    assert _label_helper("Everything is fine.", ("broken",), ComplaintLabel.BUILD_QUALITY) is None


def test_label_helper_uses_substring_matching():
    assert (
        _label_helper(
            "The device unexpectedly-disconnects sometimes.",
            ("disconnect",),
            ComplaintLabel.FUNCTIONALITY,
        )
        is ComplaintLabel.FUNCTIONALITY
    )


def test_functionality_mixed_case_matching():
    assert functionality_label("The device StOpPeD WoRkInG overnight.") is ComplaintLabel.FUNCTIONALITY


@pytest.mark.parametrize("pattern", BUILD_QUALITY_PATTERNS)
def test_each_build_quality_pattern_returns_build_quality_label(pattern):
    assert build_quality_label(f"Unfortunately, the {pattern} during normal use.") is ComplaintLabel.BUILD_QUALITY


def test_build_quality_matching_is_case_insensitive():
    assert build_quality_label("THE HEADBAND CRACKED THIS MORNING.") is ComplaintLabel.BUILD_QUALITY


def test_unrelated_text_returns_no_build_quality_label():
    assert build_quality_label("The headband is comfortable and adjustable.") is None


@pytest.mark.parametrize("pattern", SHIPPING_PATTERNS)
def test_each_shipping_pattern_returns_shipping_label(pattern):
    assert shipping_label(f"When the order arrived, the {pattern}.") is ComplaintLabel.SHIPPING


def test_shipping_matching_is_case_insensitive():
    assert shipping_label("THE BOX WAS CRUSHED DURING DELIVERY.") is ComplaintLabel.SHIPPING


def test_unrelated_text_returns_no_shipping_label():
    assert shipping_label("Delivery was prompt and the parcel was intact.") is None


@pytest.mark.parametrize("pattern", FIT_COMPATIBILITY_PATTERNS)
def test_each_fit_compatibility_pattern_returns_fit_compatibility_label(pattern):
    assert fit_compatibility_label(f"I found that it {pattern} my device.") is ComplaintLabel.FIT_COMPATIBILITY


def test_fit_compatibility_matching_is_case_insensitive():
    assert fit_compatibility_label("THE EARBUDS ARE TOO LOOSE.") is ComplaintLabel.FIT_COMPATIBILITY


def test_unrelated_text_returns_no_fit_compatibility_label():
    assert fit_compatibility_label("The earbuds fit comfortably.") is None


@pytest.mark.parametrize("pattern", USABILITY_SETUP_PATTERNS)
def test_each_usability_setup_pattern_returns_usability_setup_label(pattern):
    assert usability_setup_label(f"Setup was frustrating because {pattern}.") is ComplaintLabel.USABILITY_SETUP


def test_usability_setup_matching_is_case_insensitive():
    assert usability_setup_label("THERE WERE NO INSTRUCTIONS IN THE PACKAGE.") is ComplaintLabel.USABILITY_SETUP


def test_unrelated_text_returns_no_usability_setup_label():
    assert usability_setup_label("Setup was quick and clearly documented.") is None


@pytest.mark.parametrize("pattern", POSITIVE_PATTERNS)
@pytest.mark.parametrize("rating", [4, 5])
def test_positive_pattern_with_high_rating_returns_no_complaint(pattern, rating):
    assert no_complaint_label(f"Overall, this {pattern} for me.", rating) is ComplaintLabel.NO_COMPLAINT


@pytest.mark.parametrize("pattern", POSITIVE_PATTERNS)
@pytest.mark.parametrize("rating", [0, 3, 3.9])
def test_positive_pattern_with_rating_below_four_returns_none(pattern, rating):
    assert no_complaint_label(f"Overall, this {pattern} for me.", rating) is None


def test_rating_five_without_positive_pattern_returns_none():
    assert no_complaint_label("The product arrived yesterday.", 5) is None


def test_no_complaint_matching_is_case_insensitive():
    assert no_complaint_label("THIS WORKS GREAT FOR ME.", 4) is ComplaintLabel.NO_COMPLAINT


@pytest.mark.parametrize(
    ("text", "rating", "expected"),
    [
        ("The product arrived yesterday.", 3, None),
        ("It stopped working after one day.", 2, ComplaintLabel.FUNCTIONALITY),
        ("The headband cracked during normal use.", 2, ComplaintLabel.BUILD_QUALITY),
        ("The box was crushed in transit.", 2, ComplaintLabel.SHIPPING),
        ("These earbuds are too tight for me.", 2, ComplaintLabel.FIT_COMPATIBILITY),
        ("There were no instructions in the package.", 2, ComplaintLabel.USABILITY_SETUP),
        ("This was a great purchase for me.", 4, ComplaintLabel.NO_COMPLAINT),
        ("This was a great purchase for me.", 3, None),
        ("It stopped working and the headband cracked.", 2, None),
        ("It works great, but it stopped working today.", 5, None),
        ("THE BOX WAS CRUSHED IN TRANSIT.", 1, ComplaintLabel.SHIPPING),
    ],
)
def test_label_review_conflict_resolution(text, rating, expected):
    assert label_review(text, rating) is expected
