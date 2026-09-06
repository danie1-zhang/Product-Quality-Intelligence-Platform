"""Real MLflow round-trip smoke test for a small, untrained DistilBERT model."""

import tempfile
from pathlib import Path

import mlflow
import mlflow.transformers
import torch
import transformers

from quality_intelligence.models import train_transformer


def main():
    tokenizer = train_transformer.build_tokenizer()
    model = train_transformer.build_model()

    with tempfile.TemporaryDirectory() as directory:
        temporary_path = Path(directory)
        mlflow.set_tracking_uri(f"sqlite:///{temporary_path / 'mlflow.db'}")
        experiment_id = mlflow.create_experiment(
            "distilbert-persistence-smoke",
            artifact_location=(temporary_path / "artifacts").as_uri(),
        )
        with mlflow.start_run(experiment_id=experiment_id):
            model_info = train_transformer.log_best_model(model, tokenizer)

        components = mlflow.transformers.load_model(
            model_info.model_uri,
            return_type="components",
        )
        inputs = components["tokenizer"](
            "The battery stopped working after one day.", return_tensors="pt"
        )
        with torch.no_grad():
            logits = components["model"](**inputs).logits

    assert logits.shape == (1, len(train_transformer.SUPPORTED_LABELS))
    predicted_id = logits.argmax(dim=1).item()
    print(f"MLflow version: {mlflow.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"Model URI: {model_info.model_uri}")
    print("Logging succeeded: yes")
    print("Reload succeeded: yes")
    print(f"Inference result type: {type(logits).__name__}")
    print(f"Inference logits shape: {tuple(logits.shape)}")
    print(f"Predicted class: {components['model'].config.id2label[predicted_id]}")


if __name__ == "__main__":
    main()
