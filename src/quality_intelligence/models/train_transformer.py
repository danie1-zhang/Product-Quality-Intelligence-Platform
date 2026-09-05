import copy
import gc
import random
from collections.abc import Sized
from importlib.metadata import version
from pathlib import Path
from typing import cast

import mlflow
import mlflow.transformers as mlflow_transformers
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
TRAINING_DATA_PATH = "data/processed/reviews_weak_train.parquet"
VALIDATION_FRACTION = 0.2
RANDOM_STATE = 42
SMOKE_TRAIN_SIZE = 5_000
SMOKE_VALIDATION_SIZE = 1_000
MLFLOW_EXPERIMENT_NAME = "product-quality-complaint-classification"
TUNING_TRAIN_SIZE = 20_000
TUNING_VALIDATION_SIZE = 5_000
OPTUNA_STUDY_NAME = "distilbert-hyperparameter-tuning"
OPTUNA_STORAGE = "sqlite:///artifacts/optuna/distilbert_optuna.db"


def set_seed(seed):
    """Seed supported RNGs for practical experiment reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def load_training_data(path=TRAINING_DATA_PATH):
    df = pd.read_parquet(path, columns=["cleaned_review_text", "weak_label"])
    return df["cleaned_review_text"], df["weak_label"]


def split_data(texts, labels):
    return train_test_split(
        texts,
        labels,
        test_size=VALIDATION_FRACTION,
        random_state=RANDOM_STATE,
        stratify=labels,
    )


def stratified_subset(texts, labels, subset_size):
    subset_texts, _, subset_labels, _ = train_test_split(
        texts,
        labels,
        train_size=subset_size,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    return subset_texts, subset_labels


def build_training_dataloaders(tokenizer, batch_size, path=TRAINING_DATA_PATH):
    texts, labels = load_training_data(path)
    train_texts, validation_texts, train_labels, validation_labels = split_data(
        texts, labels
    )

    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    validation_dataset = ReviewDataset(
        validation_texts, validation_labels, tokenizer, MAX_LENGTH
    )

    train_dataloader = build_dataloader(
        train_dataset, tokenizer, batch_size=batch_size, shuffle=True
    )
    validation_dataloader = build_dataloader(
        validation_dataset, tokenizer, batch_size=batch_size, shuffle=False
    )
    return train_dataloader, validation_dataloader


def build_smoke_training_dataloaders(
    tokenizer,
    batch_size,
    path=TRAINING_DATA_PATH,
    train_subset_size=SMOKE_TRAIN_SIZE,
    validation_subset_size=SMOKE_VALIDATION_SIZE,
):
    texts, labels = load_training_data(path)
    train_texts, validation_texts, train_labels, validation_labels = split_data(
        texts, labels
    )
    train_texts, train_labels = stratified_subset(
        train_texts, train_labels, train_subset_size
    )
    validation_texts, validation_labels = stratified_subset(
        validation_texts, validation_labels, validation_subset_size
    )

    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    validation_dataset = ReviewDataset(
        validation_texts, validation_labels, tokenizer, MAX_LENGTH
    )
    train_dataloader = build_dataloader(
        train_dataset, tokenizer, batch_size=batch_size, shuffle=True
    )
    validation_dataloader = build_dataloader(
        validation_dataset, tokenizer, batch_size=batch_size, shuffle=False
    )
    return train_dataloader, validation_dataloader


def build_tuning_dataloaders(
    tokenizer,
    batch_size,
    path=TRAINING_DATA_PATH,
    train_subset_size=TUNING_TRAIN_SIZE,
    validation_subset_size=TUNING_VALIDATION_SIZE,
):
    texts, labels = load_training_data(path)
    train_texts, validation_texts, train_labels, validation_labels = split_data(
        texts, labels
    )
    train_texts, train_labels = stratified_subset(
        train_texts, train_labels, train_subset_size
    )
    validation_texts, validation_labels = stratified_subset(
        validation_texts, validation_labels, validation_subset_size
    )
    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    validation_dataset = ReviewDataset(
        validation_texts, validation_labels, tokenizer, MAX_LENGTH
    )
    return (
        build_dataloader(
            train_dataset, tokenizer, batch_size=batch_size, shuffle=True
        ),
        build_dataloader(
            validation_dataset, tokenizer, batch_size=batch_size, shuffle=False
        ),
    )


def build_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer


def build_dataloader(dataset, tokenizer, batch_size, shuffle, seed=RANDOM_STATE):
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    generator = torch.Generator().manual_seed(seed) if shuffle else None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        generator=generator,
    )

    return dataloader


def build_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(SUPPORTED_LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    return model


def build_optimizer(model, learning_rate, weight_decay):
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    return optimizer


def calculate_training_steps(train_dataloader, epochs):
    return len(train_dataloader) * epochs


def build_scheduler(optimizer, num_training_steps, warmup_ratio):
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


def train_one_epoch(model, dataloader, optimizer, device, scheduler=None):
    model.train()

    total_loss = 0.0

    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss/len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(all_labels, all_predictions, average="macro")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def train_model(
    model,
    train_dataloader,
    validation_dataloader,
    optimizer,
    epochs,
    device,
    scheduler=None,
    epoch_callback=None,
):

    history = []

    best_macro_f1 = -1.0
    best_epoch = None
    best_model_state = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_dataloader, optimizer, device, scheduler=scheduler
        )
        validation_metrics = evaluate(model, validation_dataloader, device)

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"]
        }

        current_macro_f1 = validation_metrics["macro_f1"]

        if current_macro_f1 > best_macro_f1:
            best_macro_f1 = current_macro_f1
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())

        history.append(epoch_metrics)
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"train_loss: {train_loss:.4f} - "
            f"validation_accuracy: {validation_metrics['accuracy']:.4f} - "
            f"validation_macro_f1: {validation_metrics['macro_f1']:.4f}"
        )
        if epoch_callback is not None:
            epoch_callback(epoch_metrics)

    model.load_state_dict(best_model_state)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1
    }


def log_experiment_parameters(
    *,
    learning_rate,
    batch_size,
    weight_decay,
    epochs,
    warmup_ratio,
    train_size,
    validation_size,
    device,
):
    mlflow.log_params(
        {
            "model_name": MODEL_NAME,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "max_length": MAX_LENGTH,
            "warmup_ratio": warmup_ratio,
            "random_seed": RANDOM_STATE,
            "train_size": train_size,
            "validation_size": validation_size,
            "device": str(device),
        }
    )


def log_training_metrics(training_results):
    for epoch_metrics in training_results["history"]:
        mlflow.log_metrics(
            {
                "train_loss": epoch_metrics["train_loss"],
                "validation_accuracy": epoch_metrics["validation_accuracy"],
                "validation_macro_f1": epoch_metrics["validation_macro_f1"],
            },
            step=epoch_metrics["epoch"],
        )
    mlflow.log_metrics(
        {
            "best_epoch": training_results["best_epoch"],
            "best_validation_macro_f1": training_results["best_macro_f1"],
        }
    )


def log_best_model(model, tokenizer):
    return mlflow_transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        name="model",
        task="text-classification",
        pip_requirements=[
            f"mlflow=={version('mlflow')}",
            f"torch=={version('torch')}",
            f"transformers=={version('transformers')}",
        ],
    )


def cleanup_trial_resources():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def optuna_objective(
    trial,
    *,
    train_subset_size=TUNING_TRAIN_SIZE,
    validation_subset_size=TUNING_VALIDATION_SIZE,
    epochs=2,
    path=TRAINING_DATA_PATH,
):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.15)
    batch_size = trial.suggest_categorical("batch_size", [8, 16])
    set_seed(RANDOM_STATE)
    device = get_device()
    model = tokenizer = optimizer = scheduler = None
    training_results = None

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    nested = mlflow.active_run() is not None
    try:
        tokenizer = build_tokenizer()
        train_dataloader, validation_dataloader = build_tuning_dataloaders(
            tokenizer,
            batch_size,
            path=path,
            train_subset_size=train_subset_size,
            validation_subset_size=validation_subset_size,
        )
        model = build_model().to(device)
        optimizer = build_optimizer(model, learning_rate, weight_decay)
        num_training_steps = calculate_training_steps(train_dataloader, epochs)
        scheduler = build_scheduler(optimizer, num_training_steps, warmup_ratio)

        with mlflow.start_run(
            run_name=f"optuna-trial-{trial.number}", nested=nested
        ):
            log_experiment_parameters(
                learning_rate=learning_rate,
                batch_size=batch_size,
                weight_decay=weight_decay,
                epochs=epochs,
                warmup_ratio=warmup_ratio,
                train_size=train_subset_size,
                validation_size=validation_subset_size,
                device=device,
            )
            mlflow.log_param("optuna_trial_number", trial.number)

            def report_epoch(epoch_metrics):
                mlflow.log_metrics(
                    {
                        "train_loss": epoch_metrics["train_loss"],
                        "validation_accuracy": epoch_metrics["validation_accuracy"],
                        "validation_macro_f1": epoch_metrics[
                            "validation_macro_f1"
                        ],
                    },
                    step=epoch_metrics["epoch"],
                )
                trial.report(
                    epoch_metrics["validation_macro_f1"],
                    step=epoch_metrics["epoch"],
                )
                if trial.should_prune():
                    raise optuna.TrialPruned()

            try:
                training_results = train_model(
                    model,
                    train_dataloader,
                    validation_dataloader,
                    optimizer,
                    epochs,
                    device,
                    scheduler=scheduler,
                    epoch_callback=report_epoch,
                )
            except optuna.TrialPruned:
                mlflow.set_tag("optuna_trial_state", "pruned")
                pruned = True
            else:
                mlflow.log_metric(
                    "best_validation_macro_f1", training_results["best_macro_f1"]
                )
                mlflow.set_tag("optuna_trial_state", "completed")
                pruned = False

        if pruned:
            raise optuna.TrialPruned()
        assert training_results is not None
        return training_results["best_macro_f1"]
    finally:
        model = tokenizer = optimizer = scheduler = None
        cleanup_trial_resources()


def run_optuna_study(
    n_trials=10,
    train_subset_size=TUNING_TRAIN_SIZE,
    validation_subset_size=TUNING_VALIDATION_SIZE,
    epochs=2,
    storage=OPTUNA_STORAGE,
    path=TRAINING_DATA_PATH,
):
    Path("artifacts/optuna").mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=OPTUNA_STUDY_NAME,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
    )
    study.optimize(
        lambda trial: optuna_objective(
            trial,
            train_subset_size=train_subset_size,
            validation_subset_size=validation_subset_size,
            epochs=epochs,
            path=path,
        ),
        n_trials=n_trials,
        n_jobs=1,
        gc_after_trial=True,
    )
    completed_trials = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    pruned_trials = sum(
        trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials
    )
    return {
        "best_trial_number": study.best_trial.number,
        "best_validation_macro_f1": study.best_value,
        "best_hyperparameters": study.best_params,
        "completed_trial_count": completed_trials,
        "pruned_trial_count": pruned_trials,
    }


def run_transformer_experiment(
    *,
    learning_rate,
    batch_size,
    weight_decay,
    epochs,
    warmup_ratio,
    smoke=False,
    path=TRAINING_DATA_PATH,
    run_name=None,
):
    set_seed(RANDOM_STATE)
    device = get_device()
    tokenizer = build_tokenizer()
    dataloader_builder = (
        build_smoke_training_dataloaders if smoke else build_training_dataloaders
    )
    train_dataloader, validation_dataloader = dataloader_builder(
        tokenizer, batch_size=batch_size, path=path
    )
    model = build_model().to(device)
    optimizer = build_optimizer(model, learning_rate, weight_decay)
    num_training_steps = calculate_training_steps(train_dataloader, epochs)
    scheduler = build_scheduler(optimizer, num_training_steps, warmup_ratio)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name) as run:
        log_experiment_parameters(
            learning_rate=learning_rate,
            batch_size=batch_size,
            weight_decay=weight_decay,
            epochs=epochs,
            warmup_ratio=warmup_ratio,
            train_size=len(cast(Sized, train_dataloader.dataset)),
            validation_size=len(cast(Sized, validation_dataloader.dataset)),
            device=device,
        )
        training_results = train_model(
            model,
            train_dataloader,
            validation_dataloader,
            optimizer,
            epochs,
            device,
            scheduler=scheduler,
        )
        log_training_metrics(training_results)
        training_results["mlflow_run_id"] = run.info.run_id
        try:
            model_info = log_best_model(model, tokenizer)
            training_results["mlflow_model_uri"] = model_info.model_uri
        except Exception as error:  # noqa: BLE001 - persistence must not discard training results
            training_results["mlflow_model_logging_error"] = str(error)
            print(f"Best-model artifact logging failed: {error}")

    return model, training_results


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class ReviewDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_token_type_ids=False,)
        encoding["labels"] = LABEL_TO_ID[label]
        return encoding


SUPPORTED_LABELS = [
     "NO_COMPLAINT",
    "FUNCTIONALITY",
    "BUILD_QUALITY",
    "SHIPPING",
    "FIT_COMPATIBILITY",
    "USABILITY_SETUP",
]

LABEL_TO_ID, ID_TO_LABEL = {}, {}
for idx, label in enumerate(SUPPORTED_LABELS):
    LABEL_TO_ID[label] = idx
    ID_TO_LABEL[idx] = label
