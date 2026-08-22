import os
import subprocess
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from quality_intelligence.models import baseline

EXPECTED_LABELS = (
    "NO_COMPLAINT",
    "FUNCTIONALITY",
    "BUILD_QUALITY",
    "SHIPPING",
    "FIT_COMPATIBILITY",
    "USABILITY_SETUP",
)


@pytest.fixture
def tiny_reviews():
    texts = []
    labels = []
    phrases = {
        "NO_COMPLAINT": "works perfectly",
        "FUNCTIONALITY": "battery stopped charging",
        "BUILD_QUALITY": "plastic case cracked",
        "SHIPPING": "package arrived damaged",
        "FIT_COMPATIBILITY": "does not fit device",
        "USABILITY_SETUP": "instructions are confusing",
    }
    for label, phrase in phrases.items():
        for index in range(10):
            texts.append(f"{phrase} example {index}")
            labels.append(label)
    return pd.Series(texts), pd.Series(labels)


def test_baseline_labels_are_exactly_the_six_supported_classes():
    assert baseline.BASELINE_LABELS == EXPECTED_LABELS


def test_module_import_is_side_effect_free():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    result = subprocess.run(
        [sys.executable, "-c", "import quality_intelligence.models.baseline"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.stdout == ""
    assert result.stderr == ""


def test_split_data_is_reproducible_and_stratified(tiny_reviews):
    texts, labels = tiny_reviews
    first = baseline.split_data(texts, labels)
    second = baseline.split_data(texts, labels)

    for first_part, second_part in zip(first, second, strict=True):
        pd.testing.assert_series_equal(first_part, second_part)

    _, _, y_train, y_val = first
    assert set(y_train) == set(EXPECTED_LABELS)
    assert set(y_val) == set(EXPECTED_LABELS)
    assert y_train.value_counts().to_dict() == {label: 8 for label in EXPECTED_LABELS}
    assert y_val.value_counts().to_dict() == {label: 2 for label in EXPECTED_LABELS}


def test_human_loader_excludes_other_without_modifying_source(tmp_path):
    path = tmp_path / "human.csv"
    source = pd.DataFrame(
        {
            "cleaned_review_text": ["works", "unknown issue", "broken"],
            "human_label": ["NO_COMPLAINT", "OTHER", "FUNCTIONALITY"],
        }
    )
    source.to_csv(path, index=False)
    before = path.read_bytes()

    texts, labels = baseline.load_human_evaluation_data(path)

    assert texts.tolist() == ["works", "broken"]
    assert labels.tolist() == ["NO_COMPLAINT", "FUNCTIONALITY"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("class_weight", [None, "balanced"])
def test_create_pipeline_configuration(class_weight):
    pipeline = baseline.create_pipeline(class_weight)

    assert list(pipeline.named_steps) == ["tfidf", "classifier"]
    assert pipeline.named_steps["classifier"].class_weight == class_weight
    assert pipeline.named_steps["classifier"].max_iter == 1000


def test_evaluate_classifier_returns_results_without_fitting_or_mutating(tiny_reviews):
    texts, labels = tiny_reviews
    fitted = baseline.create_pipeline().fit(texts, labels)
    coefficients_before = fitted.named_steps["classifier"].coef_.copy()

    results = baseline.evaluate_classifier(fitted, texts, labels)

    assert {"accuracy", "macro_f1", "predictions", "classification_report"} <= results.keys()
    assert len(results["predictions"]) == len(texts)
    assert isinstance(results["classification_report"], str)
    np.testing.assert_array_equal(fitted.named_steps["classifier"].coef_, coefficients_before)

    unfitted = clone(fitted)
    with pytest.raises(NotFittedError):
        baseline.evaluate_classifier(unfitted, texts, labels)


def test_pipeline_fits_and_predicts_raw_strings(tiny_reviews):
    texts, labels = tiny_reviews
    pipeline = baseline.create_pipeline().fit(texts, labels)

    predictions = pipeline.predict(["battery stopped charging"])

    assert predictions.tolist() == ["FUNCTIONALITY"]


def test_mlflow_round_trip_preserves_raw_text_pipeline(tmp_path, tiny_reviews):
    texts, labels = tiny_reviews
    pipeline = baseline.create_pipeline().fit(texts, labels)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    previous_tracking_uri = mlflow.get_tracking_uri()

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("baseline-smoke-test")
        with mlflow.start_run():
            model_info = mlflow.sklearn.log_model(pipeline, name="model")

        loaded = mlflow.sklearn.load_model(model_info.model_uri)
        predictions = loaded.predict(["battery stopped charging"])
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)

    assert list(loaded.named_steps) == ["tfidf", "classifier"]
    assert predictions.tolist() == ["FUNCTIONALITY"]


def test_run_experiment_logs_expected_metadata_and_reloadable_model(
    tmp_path, tiny_reviews
):
    texts, labels = tiny_reviews
    X_train, X_val, y_train, y_val = baseline.split_data(texts, labels)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    previous_tracking_uri = mlflow.get_tracking_uri()

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("baseline-run-test")
        unweighted_pipeline, unweighted_results = baseline.run_baseline_experiment(
            X_train,
            X_val,
            y_train,
            y_val,
            class_weight=None,
            run_name="tfidf-logreg-unweighted",
        )
        _, balanced_results = baseline.run_baseline_experiment(
            X_train,
            X_val,
            y_train,
            y_val,
            class_weight="balanced",
            run_name="tfidf-logreg-balanced",
        )
        human_path = tmp_path / "human.csv"
        pd.DataFrame(
            {
                "cleaned_review_text": texts.iloc[::10].tolist() + ["unsupported"],
                "human_label": list(EXPECTED_LABELS) + ["OTHER"],
            }
        ).to_csv(human_path, index=False)
        baseline.evaluate_selected_on_human_set(
            unweighted_pipeline,
            unweighted_results,
            human_path,
        )

        unweighted_run = mlflow.get_run(unweighted_results["mlflow_run_id"])
        balanced_run = mlflow.get_run(balanced_results["mlflow_run_id"])
        loaded_unweighted = mlflow.sklearn.load_model(
            unweighted_results["mlflow_model_uri"]
        )
        loaded_balanced = mlflow.sklearn.load_model(balanced_results["mlflow_model_uri"])
        artifact_names = {
            artifact.path
            for artifact in mlflow.MlflowClient().list_artifacts(
                unweighted_results["mlflow_run_id"], "evaluation"
            )
        }
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)

    expected_params = {
        "model_type": "logistic_regression",
        "vectorizer": "tfidf",
        "class_weight": "balanced",
        "max_iter": "1000",
        "validation_fraction": "0.2",
        "random_state": "42",
        "train_rows": str(len(X_train)),
        "validation_rows": str(len(X_val)),
    }
    assert unweighted_results["mlflow_run_id"] != balanced_results["mlflow_run_id"]
    assert unweighted_run.info.run_name == "tfidf-logreg-unweighted"
    assert balanced_run.info.run_name == "tfidf-logreg-balanced"
    assert balanced_run.data.params.items() >= expected_params.items()
    assert unweighted_run.data.params["class_weight"] == "none"
    assert int(balanced_run.data.params["vocabulary_size"]) > 0
    assert {"weak_val_accuracy", "weak_val_macro_f1"} <= balanced_run.data.metrics.keys()
    assert {"human_accuracy", "human_macro_f1"} <= unweighted_run.data.metrics.keys()
    assert artifact_names == {
        "evaluation/human_classification_report.json",
        "evaluation/human_classification_report.txt",
        "evaluation/human_confusion_matrix.json",
    }
    assert loaded_unweighted.predict(["battery stopped charging"]).tolist() == [
        "FUNCTIONALITY"
    ]
    assert loaded_balanced.predict(["battery stopped charging"]).tolist() == [
        "FUNCTIONALITY"
    ]
