import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from quality_intelligence.models import baseline, train_transformer

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

    metrics = train_transformer.evaluate(
        PredictingModel(), dataloader, torch.device("cpu")
    )

    assert {"accuracy", "macro_f1"} <= metrics.keys()
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1


def test_training_data_split_matches_baseline_contract(tmp_path):
    path = tmp_path / "weak-labels.parquet"
    rows = [
        {
            "cleaned_review_text": f"{label} review {index}",
            "weak_label": label,
        }
        for label in train_transformer.SUPPORTED_LABELS
        for index in range(10)
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)

    texts, labels = train_transformer.load_training_data(path)
    first = train_transformer.split_data(texts, labels)
    second = train_transformer.split_data(texts, labels)
    train_texts, validation_texts, train_labels, validation_labels = first

    assert train_transformer.TRAINING_DATA_PATH == baseline.TRAINING_DATA_PATH
    assert train_transformer.VALIDATION_FRACTION == baseline.VALIDATION_FRACTION
    assert train_transformer.RANDOM_STATE == baseline.RANDOM_STATE
    assert set(train_texts).isdisjoint(validation_texts)
    assert len(train_texts) + len(validation_texts) == len(texts)
    assert set(train_labels) | set(validation_labels) <= set(
        train_transformer.SUPPORTED_LABELS
    )
    for first_part, second_part in zip(first, second, strict=True):
        pd.testing.assert_series_equal(first_part, second_part)


def test_training_dataloaders_use_expected_samplers(tmp_path):
    path = tmp_path / "weak-labels.parquet"
    rows = [
        {
            "cleaned_review_text": f"{label} review {index}",
            "weak_label": label,
        }
        for label in train_transformer.SUPPORTED_LABELS
        for index in range(5)
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)

    train_loader, validation_loader = train_transformer.build_training_dataloaders(
        TinyTokenizer(), batch_size=4, path=path
    )

    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(validation_loader.sampler, SequentialSampler)
    assert len(train_loader.dataset) + len(validation_loader.dataset) == len(rows)
    assert set(train_loader.dataset.labels) | set(
        validation_loader.dataset.labels
    ) <= set(train_transformer.SUPPORTED_LABELS)


def test_stratified_subset_is_deterministic_and_preserves_proportions():
    class_counts = {
        "NO_COMPLAINT": 50,
        "FUNCTIONALITY": 30,
        "BUILD_QUALITY": 20,
        "SHIPPING": 15,
        "FIT_COMPATIBILITY": 10,
        "USABILITY_SETUP": 5,
    }
    labels = pd.Series(
        [label for label, count in class_counts.items() for _ in range(count)]
    )
    texts = pd.Series([f"review {index}" for index in range(len(labels))])

    first_texts, first_labels = train_transformer.stratified_subset(
        texts, labels, subset_size=60
    )
    second_texts, second_labels = train_transformer.stratified_subset(
        texts, labels, subset_size=60
    )

    assert len(first_texts) == len(first_labels) == 60
    pd.testing.assert_series_equal(first_texts, second_texts)
    pd.testing.assert_series_equal(first_labels, second_labels)
    assert set(first_labels) == set(train_transformer.SUPPORTED_LABELS)

    parent_proportions = labels.value_counts(normalize=True)
    subset_proportions = first_labels.value_counts(normalize=True)
    assert (parent_proportions - subset_proportions).abs().max() < 0.02


def test_smoke_dataloaders_subset_only_canonical_splits(tmp_path):
    path = tmp_path / "weak-labels.parquet"
    rows = [
        {
            "cleaned_review_text": f"{label} review {index}",
            "weak_label": label,
        }
        for label in train_transformer.SUPPORTED_LABELS
        for index in range(20)
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)
    texts, labels = train_transformer.load_training_data(path)
    canonical_before = train_transformer.split_data(texts, labels)

    train_loader, validation_loader = (
        train_transformer.build_smoke_training_dataloaders(
            TinyTokenizer(),
            batch_size=4,
            path=path,
            train_subset_size=48,
            validation_subset_size=12,
        )
    )
    canonical_after = train_transformer.split_data(texts, labels)

    assert len(train_loader.dataset) == 48
    assert len(validation_loader.dataset) == 12
    assert set(train_loader.dataset.texts) <= set(canonical_before[0])
    assert set(validation_loader.dataset.texts) <= set(canonical_before[1])
    assert set(train_loader.dataset.labels) == set(
        train_transformer.SUPPORTED_LABELS
    )
    assert set(validation_loader.dataset.labels) == set(
        train_transformer.SUPPORTED_LABELS
    )
    for before, after in zip(canonical_before, canonical_after, strict=True):
        pd.testing.assert_series_equal(before, after)


def test_scheduler_uses_calculated_training_and_warmup_steps(monkeypatch):
    captured = {}
    expected_scheduler = object()

    def fake_linear_scheduler(
        optimizer, *, num_warmup_steps, num_training_steps
    ):
        captured.update(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return expected_scheduler

    monkeypatch.setattr(
        train_transformer,
        "get_linear_schedule_with_warmup",
        fake_linear_scheduler,
    )
    optimizer = object()
    train_dataloader = [object()] * 8

    total_steps = train_transformer.calculate_training_steps(
        train_dataloader, epochs=5
    )
    scheduler = train_transformer.build_scheduler(
        optimizer, num_training_steps=total_steps, warmup_ratio=0.1
    )

    assert total_steps == 40
    assert captured == {
        "optimizer": optimizer,
        "num_warmup_steps": 4,
        "num_training_steps": 40,
    }
    assert scheduler is expected_scheduler


def test_train_one_epoch_steps_scheduler_after_each_optimizer_update():
    events = []

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, **batch):
            return SimpleNamespace(loss=self.weight * batch["labels"].float().mean())

    class RecordingOptimizer:
        def zero_grad(self):
            events.append("zero_grad")

        def step(self):
            events.append("optimizer")

    class RecordingScheduler:
        def step(self):
            events.append("scheduler")

    dataloader = [{"labels": torch.tensor([1])}, {"labels": torch.tensor([2])}]

    train_transformer.train_one_epoch(
        TinyModel(),
        dataloader,
        RecordingOptimizer(),
        torch.device("cpu"),
        scheduler=RecordingScheduler(),
    )

    assert events == [
        "zero_grad",
        "optimizer",
        "scheduler",
        "zero_grad",
        "optimizer",
        "scheduler",
    ]


def test_train_one_epoch_works_without_scheduler():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    dataloader = [{"input": torch.tensor([[1.0]])}]

    class LossModel(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, **batch):
            return SimpleNamespace(loss=self.layer(batch["input"]).sum())

    loss = train_transformer.train_one_epoch(
        LossModel(model), dataloader, optimizer, torch.device("cpu")
    )

    assert isinstance(loss, float)


def test_set_seed_reproducibly_seeds_supported_rngs():
    train_transformer.set_seed(train_transformer.RANDOM_STATE)
    first = (random.random(), np.random.random(), torch.rand(1))

    train_transformer.set_seed(train_transformer.RANDOM_STATE)
    second = (random.random(), np.random.random(), torch.rand(1))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_seeded_training_dataloaders_shuffle_reproducibly():
    class IndexDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 12

        def __getitem__(self, index):
            return {
                "input_ids": [index + 1],
                "attention_mask": [1],
                "labels": index,
            }

    tokenizer = TinyTokenizer()
    first_loader = train_transformer.build_dataloader(
        IndexDataset(), tokenizer, batch_size=3, shuffle=True
    )
    second_loader = train_transformer.build_dataloader(
        IndexDataset(), tokenizer, batch_size=3, shuffle=True
    )

    first_order = torch.cat([batch["labels"] for batch in first_loader]).tolist()
    second_order = torch.cat([batch["labels"] for batch in second_loader]).tolist()

    assert first_order == second_order
    assert first_order != list(range(12))


def test_log_experiment_parameters_logs_expected_configuration(monkeypatch):
    logged = []
    monkeypatch.setattr(train_transformer.mlflow, "log_params", logged.append)

    train_transformer.log_experiment_parameters(
        learning_rate=5e-5,
        batch_size=16,
        weight_decay=0.01,
        epochs=3,
        warmup_ratio=0.1,
        train_size=5_000,
        validation_size=1_000,
        device=torch.device("cpu"),
    )

    assert logged == [
        {
            "model_name": train_transformer.MODEL_NAME,
            "learning_rate": 5e-5,
            "batch_size": 16,
            "weight_decay": 0.01,
            "epochs": 3,
            "max_length": train_transformer.MAX_LENGTH,
            "warmup_ratio": 0.1,
            "random_seed": train_transformer.RANDOM_STATE,
            "train_size": 5_000,
            "validation_size": 1_000,
            "device": "cpu",
        }
    ]


def test_log_training_metrics_uses_epoch_steps_and_logs_best_summary(monkeypatch):
    logged = []

    def record_metrics(metrics, step=None):
        logged.append((metrics, step))

    monkeypatch.setattr(train_transformer.mlflow, "log_metrics", record_metrics)
    results = {
        "history": [
            {
                "epoch": 1,
                "train_loss": 0.5,
                "validation_accuracy": 0.8,
                "validation_macro_f1": 0.7,
            },
            {
                "epoch": 2,
                "train_loss": 0.3,
                "validation_accuracy": 0.85,
                "validation_macro_f1": 0.75,
            },
        ],
        "best_epoch": 2,
        "best_macro_f1": 0.75,
    }

    train_transformer.log_training_metrics(results)

    assert logged[:2] == [
        (
            {
                "train_loss": 0.5,
                "validation_accuracy": 0.8,
                "validation_macro_f1": 0.7,
            },
            1,
        ),
        (
            {
                "train_loss": 0.3,
                "validation_accuracy": 0.85,
                "validation_macro_f1": 0.75,
            },
            2,
        ),
    ]
    assert logged[2] == (
        {"best_epoch": 2, "best_validation_macro_f1": 0.75},
        None,
    )


def test_log_best_model_includes_model_and_tokenizer(monkeypatch):
    calls = []
    model = object()
    tokenizer = object()
    expected_info = SimpleNamespace(model_uri="runs:/run-id/model")

    def fake_log_model(**kwargs):
        calls.append(kwargs)
        return expected_info

    monkeypatch.setattr(
        train_transformer.mlflow_transformers, "log_model", fake_log_model
    )

    model_info = train_transformer.log_best_model(model, tokenizer)

    assert calls == [
        {
            "transformers_model": {"model": model, "tokenizer": tokenizer},
            "name": "model",
            "task": "text-classification",
            "pip_requirements": [
                f"mlflow=={train_transformer.version('mlflow')}",
                f"torch=={train_transformer.version('torch')}",
                f"transformers=={train_transformer.version('transformers')}",
            ],
        }
    ]
    assert model_info is expected_info


def test_experiment_logs_one_model_after_best_state_restoration(monkeypatch):
    events = []

    class FakeModel:
        restored = False

        def to(self, device):
            return self

    class FakeRun:
        info = SimpleNamespace(run_id="run-id")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    model = FakeModel()
    tokenizer = object()
    train_loader = SimpleNamespace(dataset=list(range(8)))
    validation_loader = SimpleNamespace(dataset=list(range(2)))

    monkeypatch.setattr(train_transformer, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        train_transformer, "get_device", lambda: torch.device("cpu")
    )
    monkeypatch.setattr(train_transformer, "build_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(
        train_transformer,
        "build_training_dataloaders",
        lambda tokenizer, batch_size, path: (train_loader, validation_loader),
    )
    monkeypatch.setattr(train_transformer, "build_model", lambda: model)
    monkeypatch.setattr(train_transformer, "build_optimizer", lambda *args: object())
    monkeypatch.setattr(
        train_transformer, "calculate_training_steps", lambda *args: 8
    )
    monkeypatch.setattr(train_transformer, "build_scheduler", lambda *args: object())
    monkeypatch.setattr(train_transformer.mlflow, "set_experiment", lambda name: None)
    monkeypatch.setattr(
        train_transformer.mlflow, "start_run", lambda run_name: FakeRun()
    )
    monkeypatch.setattr(
        train_transformer, "log_experiment_parameters", lambda **kwargs: None
    )
    monkeypatch.setattr(train_transformer, "log_training_metrics", lambda results: None)

    def fake_train_model(*args, **kwargs):
        model.restored = True
        events.append("best_state_restored")
        return {"history": [], "best_epoch": 1, "best_macro_f1": 0.7}

    def fake_log_best_model(logged_model, logged_tokenizer):
        assert logged_model.restored
        assert logged_tokenizer is tokenizer
        events.append("model_logged")
        return SimpleNamespace(model_uri="runs:/run-id/model")

    monkeypatch.setattr(train_transformer, "train_model", fake_train_model)
    monkeypatch.setattr(train_transformer, "log_best_model", fake_log_best_model)

    _, results = train_transformer.run_transformer_experiment(
        learning_rate=5e-5,
        batch_size=16,
        weight_decay=0.01,
        epochs=1,
        warmup_ratio=0.1,
    )

    assert events == ["best_state_restored", "model_logged"]
    assert results["mlflow_model_uri"] == "runs:/run-id/model"


def test_experiment_preserves_results_when_model_logging_fails(monkeypatch):
    class FakeModel:
        def to(self, device):
            return self

    class FakeRun:
        info = SimpleNamespace(run_id="run-id")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    loader = SimpleNamespace(dataset=list(range(2)))
    monkeypatch.setattr(train_transformer, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        train_transformer, "get_device", lambda: torch.device("cpu")
    )
    monkeypatch.setattr(train_transformer, "build_tokenizer", object)
    monkeypatch.setattr(
        train_transformer,
        "build_training_dataloaders",
        lambda tokenizer, batch_size, path: (loader, loader),
    )
    monkeypatch.setattr(train_transformer, "build_model", FakeModel)
    monkeypatch.setattr(train_transformer, "build_optimizer", lambda *args: object())
    monkeypatch.setattr(
        train_transformer, "calculate_training_steps", lambda *args: 2
    )
    monkeypatch.setattr(train_transformer, "build_scheduler", lambda *args: object())
    monkeypatch.setattr(train_transformer.mlflow, "set_experiment", lambda name: None)
    monkeypatch.setattr(
        train_transformer.mlflow, "start_run", lambda run_name: FakeRun()
    )
    monkeypatch.setattr(
        train_transformer, "log_experiment_parameters", lambda **kwargs: None
    )
    monkeypatch.setattr(train_transformer, "log_training_metrics", lambda results: None)
    monkeypatch.setattr(
        train_transformer,
        "train_model",
        lambda *args, **kwargs: {
            "history": [{"epoch": 1}],
            "best_epoch": 1,
            "best_macro_f1": 0.7,
        },
    )
    monkeypatch.setattr(
        train_transformer,
        "log_best_model",
        lambda *args: (_ for _ in ()).throw(ModuleNotFoundError("torchvision")),
    )

    _, results = train_transformer.run_transformer_experiment(
        learning_rate=5e-5,
        batch_size=2,
        weight_decay=0.01,
        epochs=1,
        warmup_ratio=0.1,
    )

    assert results["best_epoch"] == 1
    assert results["best_macro_f1"] == 0.7
    assert results["mlflow_run_id"] == "run-id"
    assert results["mlflow_model_logging_error"] == "torchvision"
