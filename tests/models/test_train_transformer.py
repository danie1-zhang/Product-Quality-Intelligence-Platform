from quality_intelligence.models import train_transformer


EXPECTED_LABEL_TO_ID = {
    "NO_COMPLAINT": 0,
    "FUNCTIONALITY": 1,
    "BUILD_QUALITY": 2,
    "SHIPPING": 3,
    "FIT_COMPATIBILITY": 4,
    "USABILITY_SETUP": 5,
}


def test_supported_labels_contains_exactly_six_classes():
    assert len(train_transformer.SUPPORTED_LABELS) == 6
    assert len(set(train_transformer.SUPPORTED_LABELS)) == 6


def test_label_mappings_are_exact_inverses():
    assert train_transformer.LABEL_TO_ID == {
        label: label_id for label_id, label in train_transformer.ID_TO_LABEL.items()
    }
    assert train_transformer.ID_TO_LABEL == {
        label_id: label for label, label_id in train_transformer.LABEL_TO_ID.items()
    }


def test_other_is_not_a_supported_label():
    assert "OTHER" not in train_transformer.LABEL_TO_ID


def test_label_to_id_mapping_is_stable():
    assert train_transformer.LABEL_TO_ID == EXPECTED_LABEL_TO_ID
