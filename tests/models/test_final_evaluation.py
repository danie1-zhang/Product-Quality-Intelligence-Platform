from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from quality_intelligence.models import final_evaluation, train_transformer


def test_final_transformer_configuration_is_exact():
    config = train_transformer.FINAL_TRANSFORMER_CONFIG

    assert config.model_name == "distilbert-base-uncased"
    assert config.selected_trial_number == 0
    assert config.optuna_best_validation_macro_f1 == 0.9766622479790312
    assert config.learning_rate == 1.827226177606625e-05
    assert config.weight_decay == 0.09507143064099162
    assert config.warmup_ratio == 0.10979909127171077
    assert config.batch_size == 8
    assert config.epochs == 2
    assert config.max_length == 256
    assert config.random_seed == 42


def test_final_training_wrapper_uses_selected_full_data_configuration(monkeypatch):
    captured = {}

    class RunContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def fake_run_transformer_experiment(**kwargs):
        captured.update(kwargs)
        return object(), {
            "best_epoch": 1,
            "best_macro_f1": 0.98,
            "mlflow_run_id": "run-id",
            "mlflow_model_uri": "models:/model-id",
        }

    monkeypatch.setattr(
        train_transformer,
        "run_transformer_experiment",
        fake_run_transformer_experiment,
    )
    monkeypatch.setattr(
        train_transformer.mlflow,
        "start_run",
        lambda run_id: RunContext(),
    )
    monkeypatch.setattr(train_transformer.mlflow, "log_params", lambda params: None)

    results = train_transformer.run_final_transformer_experiment(path="training.parquet")
    config = train_transformer.FINAL_TRANSFORMER_CONFIG

    assert captured == {
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "weight_decay": config.weight_decay,
        "epochs": config.epochs,
        "warmup_ratio": config.warmup_ratio,
        "smoke": False,
        "path": "training.parquet",
        "run_name": "distilbert-final-selected-trial-0",
        "max_length": config.max_length,
        "random_seed": config.random_seed,
    }
    assert results["best_epoch"] == 1
    assert results["best_validation_macro_f1"] == 0.98
    assert results["selected_hyperparameters"]["batch_size"] == 8


def test_final_training_does_not_access_human_evaluation(monkeypatch):
    monkeypatch.setattr(
        final_evaluation.baseline,
        "load_human_evaluation_data",
        lambda *args: pytest.fail("training accessed frozen human evaluation data"),
    )
    monkeypatch.setattr(
        train_transformer,
        "run_transformer_experiment",
        lambda **kwargs: (
            object(),
            {
                "best_epoch": 1,
                "best_macro_f1": 0.9,
                "mlflow_run_id": "run-id",
                "mlflow_model_uri": "models:/model-id",
            },
        ),
    )

    class RunContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(
        train_transformer.mlflow, "start_run", lambda run_id: RunContext()
    )
    monkeypatch.setattr(train_transformer.mlflow, "log_params", lambda params: None)

    train_transformer.run_final_transformer_experiment()


def test_human_evaluation_filters_other_and_writes_expected_artifacts(
    tmp_path, monkeypatch
):
    human_path = tmp_path / "human.csv"
    pd.DataFrame(
        {
            "cleaned_review_text": [f"review {index}" for index in range(7)],
            "human_label": train_transformer.SUPPORTED_LABELS + ["OTHER"],
        }
    ).to_csv(human_path, index=False)
    captured_texts = []

    def fake_predict(model_uri, texts, batch_size, device):
        captured_texts.extend(texts.tolist())
        return train_transformer.SUPPORTED_LABELS.copy()

    monkeypatch.setattr(
        final_evaluation, "predict_with_persisted_transformer", fake_predict
    )

    results = final_evaluation.evaluate_persisted_transformer_on_human_set(
        "models:/model-id", path=human_path, output_dir=tmp_path / "output"
    )

    assert len(captured_texts) == 6
    assert len(results["per_class_metrics"]) == 6
    assert [row["label"] for row in results["per_class_metrics"]] == (
        train_transformer.SUPPORTED_LABELS
    )
    assert len(results["confusion_matrix"]) == 6
    assert all(len(row) == 6 for row in results["confusion_matrix"])

    output = tmp_path / "output"
    assert {path.name for path in output.iterdir()} == {
        "metrics_summary.json",
        "per_class_metrics.csv",
        "confusion_matrix.csv",
        "predictions.csv",
    }
    assert list(pd.read_csv(output / "per_class_metrics.csv").columns) == [
        "label",
        "precision",
        "recall",
        "f1",
        "support",
    ]
    assert list(pd.read_csv(output / "confusion_matrix.csv").columns) == [
        "actual_label",
        *train_transformer.SUPPORTED_LABELS,
    ]
    assert list(pd.read_csv(output / "predictions.csv").columns) == [
        "review_text",
        "actual_label",
        "predicted_label",
    ]


def test_persisted_model_prediction_path_is_callable(monkeypatch):
    tokenizer_arguments = {}

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            tokenizer_arguments.update(kwargs)
            return {
                "input_ids": torch.ones((len(texts), 2), dtype=torch.long),
                "attention_mask": torch.ones((len(texts), 2), dtype=torch.long),
            }

    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **encoded):
            logits = torch.zeros((len(encoded["input_ids"]), 6))
            logits[:, 1] = 1
            return SimpleNamespace(logits=logits)

    monkeypatch.setattr(
        final_evaluation.mlflow_transformers,
        "load_model",
        lambda model_uri, return_type: {"model": Model(), "tokenizer": Tokenizer()},
    )

    predictions = final_evaluation.predict_with_persisted_transformer(
        "models:/model-id",
        ["battery stopped working"],
        device=torch.device("cpu"),
    )

    assert predictions == ["FUNCTIONALITY"]
    assert tokenizer_arguments["max_length"] == (
        train_transformer.FINAL_TRANSFORMER_CONFIG.max_length
    )


@pytest.mark.parametrize(
    ("baseline_f1", "transformer_f1", "champion"),
    [(0.7, 0.8, "distilbert"), (0.9, 0.8, "tfidf_logistic_regression")],
)
def test_comparison_selects_higher_human_macro_f1(
    baseline_f1, transformer_f1, champion
):
    comparison = final_evaluation.compare_human_metrics(
        {"accuracy": 0.8, "macro_f1": baseline_f1},
        {"accuracy": 0.85, "macro_f1": transformer_f1},
    )

    assert comparison["champion_model"] == champion
    assert comparison["macro_f1_delta"] == pytest.approx(
        transformer_f1 - baseline_f1
    )


def test_final_metadata_contains_training_and_human_results(tmp_path):
    metadata = final_evaluation.write_final_metadata(
        {
            "best_epoch": 1,
            "best_validation_macro_f1": 0.98,
            "mlflow_run_id": "run-id",
            "mlflow_model_uri": "models:/model-id",
        },
        {"accuracy": 0.9, "macro_f1": 0.88},
        output_dir=tmp_path,
    )

    assert metadata["selected_optuna_trial"] == 0
    assert metadata["full_validation_best_epoch"] == 1
    assert metadata["human_test_metrics"] == {"accuracy": 0.9, "macro_f1": 0.88}
    assert (tmp_path / "final_metadata.json").is_file()
