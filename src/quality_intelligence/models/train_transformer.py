from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, f1_score


MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256


def build_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer


def build_dataloader(dataset, tokenizer, batch_size, shuffle):
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator
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


def train_one_epoch(model, dataloader, optimizer):
    model.train()

    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss/len(dataloader)


def evaluate(model, dataloader):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=1)

            all_predictions.extend(predictions.tolist())
            all_labels.extend(batch["labels"].tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(all_labels, all_predictions, average="macro")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


class ReviewDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
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