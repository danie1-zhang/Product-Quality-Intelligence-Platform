import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASELINE_LABELS = (
    "NO_COMPLAINT",
    "FUNCTIONALITY",
    "BUILD_QUALITY",
    "SHIPPING",
    "FIT_COMPATIBILITY",
    "USABILITY_SETUP",
)

MLFLOW_EXPERIMENT_NAME = "product-quality-complaint-classification"
TRAINING_DATA_PATH = "data/processed/reviews_weak_train.parquet"
HUMAN_EVAL_PATH = "data/annotation/human_labeled_reviews.csv"
VALIDATION_FRACTION = 0.2
RANDOM_STATE = 42
MAX_ITER = 1000


def create_pipeline(class_weight=None):
    """Create a deployable baseline pipeline that accepts raw review text."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer()),
            (
                "classifier",
                LogisticRegression(max_iter=MAX_ITER, class_weight=class_weight),
            ),
        ]
    )


def split_data(texts, labels):
    return train_test_split(
        texts,
        labels,
        test_size=VALIDATION_FRACTION,
        random_state=RANDOM_STATE,
        stratify=labels,
    )


def load_training_data(path=TRAINING_DATA_PATH):
    df = pd.read_parquet(path, columns=["cleaned_review_text", "weak_label"])
    return df["cleaned_review_text"], df["weak_label"]


def load_human_evaluation_data(path=HUMAN_EVAL_PATH):
    df = pd.read_csv(path, usecols=["cleaned_review_text", "human_label"])
    supported_df = df[df["human_label"].isin(BASELINE_LABELS)].copy()
    return supported_df["cleaned_review_text"], supported_df["human_label"]


def evaluate_classifier(classifier, X, y):
    """Evaluate a fitted classifier or pipeline without fitting it."""
    predictions = classifier.predict(X)
    accuracy = accuracy_score(y, predictions)
    macro_f1 = f1_score(
        y,
        predictions,
        labels=BASELINE_LABELS,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y,
        predictions,
        labels=BASELINE_LABELS,
        zero_division=0,
    )
    report_dict = classification_report(
        y,
        predictions,
        labels=BASELINE_LABELS,
        zero_division=0,
        output_dict=True,
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(report)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "predictions": predictions,
        "classification_report": report,
        "classification_report_dict": report_dict,
    }


def run_baseline_experiment(
    X_train,
    X_val,
    y_train,
    y_val,
    *,
    class_weight=None,
    run_name=None,
):
    """Fit and weak-validate one raw-text baseline pipeline in an MLflow run."""
    pipeline = create_pipeline(class_weight=class_weight)
    pipeline.fit(X_train, y_train)

    vectorizer = pipeline.named_steps["tfidf"]
    vocabulary_size = len(vectorizer.get_feature_names_out())
    print(f"TF-IDF train shape: ({len(X_train)}, {vocabulary_size})")
    print(f"TF-IDF validation shape: ({len(X_val)}, {vocabulary_size})")
    print(f"Vocabulary size: {vocabulary_size}")

    results = evaluate_classifier(pipeline, X_val, y_val)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "model_type": "logistic_regression",
                "vectorizer": "tfidf",
                "class_weight": "none" if class_weight is None else class_weight,
                "max_iter": MAX_ITER,
                "validation_fraction": VALIDATION_FRACTION,
                "random_state": RANDOM_STATE,
                "train_rows": len(X_train),
                "validation_rows": len(X_val),
                "vocabulary_size": vocabulary_size,
            }
        )
        mlflow.log_metrics(
            {
                "weak_val_accuracy": results["accuracy"],
                "weak_val_macro_f1": results["macro_f1"],
            }
        )
        model_info = mlflow.sklearn.log_model(pipeline, name="model")
        results["mlflow_run_id"] = run.info.run_id
        results["mlflow_model_uri"] = model_info.model_uri

    return pipeline, results


def evaluate_selected_on_human_set(pipeline, weak_results, path=HUMAN_EVAL_PATH):
    """Evaluate and log the selected pipeline on the frozen supported human set."""
    human_texts, human_labels = load_human_evaluation_data(path)
    print("\n=== Frozen Human Evaluation (selected unweighted model) ===")
    print(f"Supported human evaluation rows: {len(human_texts)}")

    human_results = evaluate_classifier(pipeline, human_texts, human_labels)
    matrix = confusion_matrix(
        human_labels,
        human_results["predictions"],
        labels=BASELINE_LABELS,
    )
    print("\nHuman confusion matrix:")
    print(matrix)

    confusion_matrix_artifact = {
        "labels": list(BASELINE_LABELS),
        "matrix": matrix.tolist(),
    }
    with mlflow.start_run(run_id=weak_results["mlflow_run_id"]):
        mlflow.log_metrics(
            {
                "human_accuracy": human_results["accuracy"],
                "human_macro_f1": human_results["macro_f1"],
            }
        )
        mlflow.log_text(
            human_results["classification_report"],
            "evaluation/human_classification_report.txt",
        )
        mlflow.log_dict(
            human_results["classification_report_dict"],
            "evaluation/human_classification_report.json",
        )
        mlflow.log_dict(
            confusion_matrix_artifact,
            "evaluation/human_confusion_matrix.json",
        )
    human_results["confusion_matrix"] = matrix
    return human_results


def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    texts, labels = load_training_data()
    X_train, X_val, y_train, y_val = split_data(texts, labels)

    print(f"Training rows: {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")
    print("\nTraining distribution:")
    print(y_train.value_counts())
    print("\nValidation distribution:")
    print(y_val.value_counts())

    print("\n=== Unweighted Logistic Regression ===")
    unweighted_pipeline, unweighted_results = run_baseline_experiment(
        X_train,
        X_val,
        y_train,
        y_val,
        class_weight=None,
        run_name="tfidf-logreg-unweighted",
    )

    print("\n=== Balanced Logistic Regression ===")
    run_baseline_experiment(
        X_train,
        X_val,
        y_train,
        y_val,
        class_weight="balanced",
        run_name="tfidf-logreg-balanced",
    )

    evaluate_selected_on_human_set(unweighted_pipeline, unweighted_results)


if __name__ == "__main__":
    main()
