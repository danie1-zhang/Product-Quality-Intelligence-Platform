"""Explicit post-training evaluation for the persisted final transformer model."""

import json
from dataclasses import asdict
from pathlib import Path

import mlflow
import mlflow.transformers as mlflow_transformers
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from quality_intelligence.models import baseline, train_transformer

DEFAULT_OUTPUT_DIR = Path("artifacts/evaluation/distilbert")


def load_persisted_transformer(model_uri):
    """Reload independently persisted Hugging Face model and tokenizer components."""
    return mlflow_transformers.load_model(model_uri, return_type="components")


def predict_with_persisted_transformer(model_uri, texts, batch_size=32, device=None):
    components = load_persisted_transformer(model_uri)
    model = components["model"]
    tokenizer = components["tokenizer"]
    device = train_transformer.get_device() if device is None else device
    model.to(device)
    model.eval()
    predictions = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = list(texts[start : start + batch_size])
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=train_transformer.FINAL_TRANSFORMER_CONFIG.max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            predicted_ids = model(**encoded).logits.argmax(dim=1).cpu().tolist()
            predictions.extend(
                train_transformer.ID_TO_LABEL[predicted_id]
                for predicted_id in predicted_ids
            )

    return predictions


def calculate_human_metrics(texts, labels, predictions):
    label_order = train_transformer.SUPPORTED_LABELS
    report = classification_report(
        labels,
        predictions,
        labels=label_order,
        output_dict=True,
        zero_division=0,
    )
    per_class_metrics = [
        {
            "label": label,
            "precision": report[label]["precision"],
            "recall": report[label]["recall"],
            "f1": report[label]["f1-score"],
            "support": report[label]["support"],
        }
        for label in label_order
    ]
    matrix = confusion_matrix(labels, predictions, labels=label_order)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels,
            predictions,
            labels=label_order,
            average="macro",
            zero_division=0,
        ),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": matrix.tolist(),
        "predictions": [
            {
                "review_text": text,
                "actual_label": actual,
                "predicted_label": predicted,
            }
            for text, actual, predicted in zip(texts, labels, predictions, strict=True)
        ],
    }


def write_evaluation_artifacts(results, output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "accuracy": results["accuracy"],
        "macro_f1": results["macro_f1"],
        "evaluated_rows": len(results["predictions"]),
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame(results["per_class_metrics"]).to_csv(
        output_dir / "per_class_metrics.csv", index=False
    )
    pd.DataFrame(
        results["confusion_matrix"],
        index=train_transformer.SUPPORTED_LABELS,
        columns=train_transformer.SUPPORTED_LABELS,
    ).rename_axis("actual_label").to_csv(output_dir / "confusion_matrix.csv")
    pd.DataFrame(results["predictions"]).to_csv(
        output_dir / "predictions.csv", index=False
    )
    return summary


def evaluate_persisted_transformer_on_human_set(
    model_uri,
    path=baseline.HUMAN_EVAL_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    batch_size=32,
    device=None,
):
    """Run the explicitly separate final evaluation on supported frozen labels."""
    texts, labels = baseline.load_human_evaluation_data(path)
    predictions = predict_with_persisted_transformer(
        model_uri, texts, batch_size=batch_size, device=device
    )
    results = calculate_human_metrics(
        texts.tolist(), labels.tolist(), predictions
    )
    write_evaluation_artifacts(results, output_dir)
    return results


def load_baseline_human_metrics(mlflow_run_id):
    metrics = mlflow.get_run(mlflow_run_id).data.metrics
    return {
        "accuracy": metrics["human_accuracy"],
        "macro_f1": metrics["human_macro_f1"],
    }


def compare_human_metrics(baseline_metrics, transformer_metrics):
    baseline_macro_f1 = baseline_metrics["macro_f1"]
    transformer_macro_f1 = transformer_metrics["macro_f1"]
    if transformer_macro_f1 > baseline_macro_f1:
        champion = "distilbert"
    elif baseline_macro_f1 > transformer_macro_f1:
        champion = "tfidf_logistic_regression"
    else:
        champion = "tie"
    return {
        "baseline_human_accuracy": baseline_metrics["accuracy"],
        "baseline_human_macro_f1": baseline_macro_f1,
        "distilbert_human_accuracy": transformer_metrics["accuracy"],
        "distilbert_human_macro_f1": transformer_macro_f1,
        "macro_f1_delta": transformer_macro_f1 - baseline_macro_f1,
        "champion_model": champion,
    }


def write_final_metadata(
    training_results,
    human_results,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    config = train_transformer.FINAL_TRANSFORMER_CONFIG
    metadata = {
        "model_name": config.model_name,
        "selected_optuna_trial": config.selected_trial_number,
        "optuna_tuning_macro_f1": config.optuna_best_validation_macro_f1,
        "selected_hyperparameters": asdict(config),
        "full_validation_best_epoch": training_results["best_epoch"],
        "full_validation_macro_f1": training_results[
            "best_validation_macro_f1"
        ],
        "mlflow_run_id": training_results["mlflow_run_id"],
        "mlflow_model_uri": training_results["mlflow_model_uri"],
        "human_test_metrics": {
            "accuracy": human_results["accuracy"],
            "macro_f1": human_results["macro_f1"],
        },
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "final_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata
