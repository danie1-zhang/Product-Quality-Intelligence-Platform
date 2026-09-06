# Product Quality Intelligence Platform

An end-to-end machine learning project for extracting product-quality signals from large-scale Amazon review data. The implemented pipeline structures and preprocesses reviews, generates weak complaint labels, trains supervised classifiers, evaluates the selected models against a frozen manually labeled test set, and tracks experiments and model artifacts with MLflow.

The current project focuses on supervised complaint classification. Unsupervised defect discovery and production serving are planned extensions.

## Complaint taxonomy

The supervised benchmark uses six classes:

- `NO_COMPLAINT`
- `FUNCTIONALITY`
- `BUILD_QUALITY`
- `SHIPPING`
- `FIT_COMPATIBILITY`
- `USABILITY_SETUP`

`OTHER` is excluded from the supervised benchmark and its reported evaluation metrics.

## Implemented pipeline

```text
Raw Amazon reviews
    → PyArrow / PySpark preprocessing and structuring
    → Weak complaint labeling
    → Canonical stratified train/validation split
    → TF-IDF + Logistic Regression baseline
    → DistilBERT fine-tuning
    → Optuna hyperparameter search
    → MLflow experiment tracking and model persistence
    → Frozen human evaluation
```

This is a portfolio-scale ML workflow rather than a production deployment. The repository includes data ingestion and preprocessing utilities, weak-label generation, reproducible model training, hyperparameter tuning, model serialization, and evaluation artifact generation.

## Model selection methodology

Development uses weak-labeled data with a deterministic, stratified 80/20 train/validation split. Hyperparameters and training checkpoints are selected only from weak-label validation macro-F1.

A separate, frozen manually labeled set is reserved for final evaluation. It is not used for training, checkpoint selection, or Optuna tuning. This separation provides a more credible estimate of generalization beyond the weak-label rules.

Macro-F1 is the primary model-comparison metric because complaint classes are imbalanced and performance across all classes matters.

## Final supervised results

| Model | Human Accuracy | Human Macro-F1 |
| --- | ---: | ---: |
| TF-IDF + Logistic Regression | 0.7785 | 0.7643 |
| DistilBERT | 0.8354 | 0.8245 |

DistilBERT is the supervised champion, improving absolute human-test macro-F1 by **+0.0602** over the baseline.

The selected DistilBERT checkpoint reached a best **weak-label validation macro-F1 of 0.9800 at epoch 2**. This is development-set performance against weak labels—not performance on the frozen human test set. Its human-test macro-F1 is 0.8245, as reported above.

## Selected DistilBERT configuration

The final configuration was selected by Optuna before opening the frozen human evaluation results.

| Parameter | Value |
| --- | --- |
| Model | `distilbert-base-uncased` |
| Learning rate | `1.827226177606625e-05` |
| Weight decay | `0.09507143064099162` |
| Warmup ratio | `0.10979909127171077` |
| Batch size | `8` |
| Epochs | `2` |
| Maximum sequence length | `256` |
| Random seed | `42` |

Training uses dynamic batch padding, AdamW, linear warmup and decay, deterministic sampling where practical, device selection across MPS/CUDA/CPU, and best-epoch restoration based on validation macro-F1.

## Human-test error analysis

`SHIPPING` and `BUILD_QUALITY` were among the strongest classes on the frozen human set. `USABILITY_SETUP` achieved high recall but substantially lower precision and was the weakest class overall. These observations are post-evaluation findings; the model and hyperparameters were not subsequently tuned against them.

## Experiment tracking and evaluation

MLflow records model configuration, dataset sizes, device information, per-epoch loss and validation metrics, best-checkpoint metrics, and the restored DistilBERT model with its tokenizer. Final evaluation can reload the persisted model independently and produce summary metrics, per-class metrics, a confusion matrix, and row-level prediction files for local analysis. Human review contents are intentionally not documented here.

## Technology

- Python
- PyTorch
- Hugging Face Transformers
- scikit-learn
- PySpark and PyArrow
- pandas
- Optuna
- MLflow
- pytest
- Ruff

## Repository status

The supervised classification workflow is implemented and tested, including preprocessing, weak labeling, TF-IDF baseline training, DistilBERT fine-tuning, Optuna tuning, MLflow persistence, and frozen-set evaluation plumbing. Tests use lightweight fixtures and mocks to avoid expensive model training during routine development.

Large datasets, local experiment databases, generated evaluation artifacts, and model binaries are excluded from version control.

## Next steps

The following capabilities are planned and are not presented as complete:

- Sentence-transformer embeddings for semantic review representation
- HDBSCAN-based unsupervised defect discovery
- Production API and model serving
- Workflow orchestration, containerization, and deployment
