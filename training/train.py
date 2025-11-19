
"""
Training script for fine tuning model for classification.
Use with DeBERTa, DistilBERT, and any HF sequence classification model
"""

import argparse
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
import evaluate

# Config

@dataclass
class TrainingConfig:
    data_path: str
    model_checkpoint: str
    output_dir: str
    max_length: int = 256
    test_size: float = 0.1
    learning_rate: float = 2e-5
    num_epochs: int = 3
    weight_decay: float = 0.01
    train_batch: int = 8
    eval_batch: int = 8
def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./model_output")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--train_batch", type=int, default=8)
    parser.add_argument("--eval_batch", type=int, default=8)
    a = parser.parse_args()
    return TrainingConfig(**vars(a))

# Data loading + preprocessing

def load_dataset(cfg: TrainingConfig) -> Dataset:
    df = pd.read_csv(cfg.data_path)
    df = df.dropna(subset=["note", "label"])
    df["label"] = df["label"].astype("category").cat.codes
    return Dataset.from_pandas(df[["note", "label"]])
def tokenize_dataset(dataset: Dataset, tokenizer, cfg: TrainingConfig):
    def tok(ex):
        return tokenizer(
            ex["note"],
            truncation=True,
            padding="max_length",
            max_length=cfg.max_length
        )
    dataset = dataset.rename_columns({"note": "text"})
    dataset = dataset.class_encode_column("label")
    split = dataset.train_test_split(test_size=cfg.test_size)
    tokenized = split.map(tok, batched=True)
    return tokenized

# Metrics

metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1": metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

# Training

def main():
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)
    dataset = load_dataset(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_checkpoint)
    tokenized = tokenize_dataset(dataset, tokenizer, cfg)
    num_labels = len(set(dataset["label"]))
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_checkpoint,
        num_labels=num_labels
    )
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.train_batch,
        per_device_eval_batch_size=cfg.eval_batch,
        num_train_epochs=cfg.num_epochs,
        weight_decay=cfg.weight_decay,
        logging_dir=os.path.join(cfg.output_dir, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"Training complete. Model saved to: {cfg.output_dir}")
  
if __name__ == "__main__":
    main()
