# ============================================================
# Team: primeline
# Task: Tamil Abusive Comment Detection
# Model: MuRIL (google/muril-base-cased)
# ============================================================

import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load datasets
train_df = pd.read_csv("trainV2.csv")
test_df  = pd.read_csv("TestV2 - testV2.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Fix columns
train_df = train_df.iloc[:, :2]
train_df.columns = ["text", "label"]

test_df = test_df.iloc[:, :1]
test_df.columns = ["text"]

# Clean text
train_df["text"] = train_df["text"].astype(str).str.strip()
test_df["text"]  = test_df["text"].astype(str).str.strip()

# Label mapping
label_map = {
    "Non-Abusive": 0,
    "Abusive": 1,
    "சாதாரணம்": 0,
    "அவதூறு": 1,
    "non-abusive": 0,
    "abusive": 1,
    "0": 0,
    "1": 1
}

train_df["label"] = train_df["label"].astype(str).map(label_map)
train_df = train_df.dropna()
train_df["label"] = train_df["label"].astype(int)

print("\nLabel distribution:")
print(train_df["label"].value_counts())

# Train / Validation Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["text"],
    train_df["label"],
    test_size=0.1,
    stratify=train_df["label"],
    random_state=42
)

# Load MuRIL tokenizer
model_name = "google/muril-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(texts):
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=128
    )

train_encodings = tokenize(train_texts)
val_encodings   = tokenize(val_texts)

train_dataset = Dataset.from_dict({
    **train_encodings,
    "labels": train_labels.tolist()
})

val_dataset = Dataset.from_dict({
    **val_encodings,
    "labels": val_labels.tolist()
})

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

model.to(device)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    f1 = f1_score(labels, preds, average="macro")
    print(classification_report(labels, preds))
    return {"macro_f1": f1}

# Training settings
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_steps=50,
    save_steps=500,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train
print("\nTraining started...\n")
trainer.train()

print("\nValidation Results:\n")
trainer.evaluate()

# Predict on Test Set
print("\nGenerating test predictions...\n")

tokens = tokenizer(
    test_df["text"].tolist(),
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

tokens = {k: v.to(device) for k, v in tokens.items()}

model.eval()

with torch.no_grad():
    outputs = model(**tokens)
    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

# Convert predictions
reverse_map = {
    0: "Non-Abusive",
    1: "Abusive"
}

submission = pd.DataFrame({
    "label": [reverse_map[p] for p in preds]
})

# Save submission
submission.to_csv("primeline.csv", index=False)

print("\nSubmission file created: primeline.csv")
print("\nPipeline complete.")
