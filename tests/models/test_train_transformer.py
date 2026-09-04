from types import SimpleNamespace

import torch

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


class TinyTokenizer:
    """Tokenizer stub with the padding interface used by the HF collator."""

    def __call__(self, text, *, truncation, max_length, return_token_type_ids):
        token_ids = list(range(1, min(len(text.split()) + 2, max_length) + 1))
        return {"input_ids": token_ids, "attention_mask": [1] * len(token_ids)}

    def pad(self, features, *, padding, max_length, pad_to_multiple_of, return_tensors):
        batch_length = max(len(feature["input_ids"]) for feature in features)
        padded = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding_length = batch_length - len(feature["input_ids"])
            padded["input_ids"].append(feature["input_ids"] + [0] * padding_length)
            padded["attention_mask"].append(
                feature["attention_mask"] + [0] * padding_length
            )
            padded["labels"].append(feature["labels"])
        return {key: torch.tensor(value) for key, value in padded.items()}


def test_review_dataset_length_and_item_are_unpadded():
    dataset = train_transformer.ReviewDataset(
        ["battery stopped charging", "arrived late"],
        ["FUNCTIONALITY", "SHIPPING"],
        TinyTokenizer(),
        train_transformer.MAX_LENGTH,
    )

    item = dataset[0]

    assert len(dataset) == 2
    assert {"input_ids", "attention_mask", "labels"} <= item.keys()
    assert item["labels"] == 1
    assert len(item["input_ids"]) < train_transformer.MAX_LENGTH


def test_build_dataloader_dynamically_pads_batch():
    tokenizer = TinyTokenizer()
    dataset = train_transformer.ReviewDataset(
        ["battery failed", "package arrived extremely late"],
        ["FUNCTIONALITY", "SHIPPING"],
        tokenizer,
        train_transformer.MAX_LENGTH,
    )

    batch = next(
        iter(
            train_transformer.build_dataloader(
                dataset, tokenizer, batch_size=2, shuffle=False
            )
        )
    )

    assert isinstance(batch["input_ids"], torch.Tensor)
    assert batch["input_ids"].ndim == 2
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["labels"].tolist() == [1, 3]


def test_build_model_configures_six_classes(monkeypatch):
    def fake_from_pretrained(model_name, **config):
        return SimpleNamespace(config=SimpleNamespace(**config))

    monkeypatch.setattr(
        train_transformer.AutoModelForSequenceClassification,
        "from_pretrained",
        fake_from_pretrained,
    )

    model = train_transformer.build_model()

    assert model.config.num_labels == 6
    assert model.config.label2id == train_transformer.LABEL_TO_ID
    assert model.config.id2label == train_transformer.ID_TO_LABEL


def test_evaluate_returns_bounded_metrics():
    class PredictingModel:
        def eval(self):
            return self

        def __call__(self, **batch):
            logits = torch.zeros(len(batch["labels"]), 6)
            logits[range(len(batch["labels"])), batch["labels"]] = 1
            return SimpleNamespace(logits=logits)

    dataloader = [
        {
            "input_ids": torch.tensor([[1, 2], [3, 0]]),
            "attention_mask": torch.tensor([[1, 1], [1, 0]]),
            "labels": torch.tensor([1, 3]),
        }
    ]

    metrics = train_transformer.evaluate(PredictingModel(), dataloader)

    assert {"accuracy", "macro_f1"} <= metrics.keys()
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1
