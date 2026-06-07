"""
NewsIQ — DistilBERT News Classifier
Fine-tunes DistilBERT on AG News (4-class classification).
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score,
)
from src.config import (
    CLASSIFIER_MODEL_NAME, CLASSIFIER_MAX_LENGTH,
    CLASSIFIER_EPOCHS, CLASSIFIER_BATCH_SIZE,
    CLASSIFIER_EVAL_BATCH_SIZE, CLASSIFIER_LEARNING_RATE,
    CLASSIFIER_WEIGHT_DECAY, CLASSIFIER_WARMUP_RATIO,
    CLASSIFIER_SAVE_DIR, AG_NEWS_CLASSES,
    AG_NEWS_NUM_CLASSES, RANDOM_SEED,
)

logger = logging.getLogger(__name__)


def tokenize_dataset(dataset, tokenizer):
    """Tokenize AG News dataset for DistilBERT."""
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"], truncation=True,
            padding="max_length", max_length=CLASSIFIER_MAX_LENGTH,
        )
    tokenized = dataset.map(tokenize_fn, batched=True, batch_size=1000)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def compute_metrics(eval_pred):
    """Compute F1 (macro) and accuracy for HuggingFace Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "f1": f1_score(labels, predictions, average="macro"),
        "accuracy": accuracy_score(labels, predictions),
    }


def train_classifier(dataset: dict, save_dir: Optional[Path] = None):
    """
    Fine-tune DistilBERT on AG News. Designed for Colab T4.
    Training: ~15 minutes, 3 epochs, batch_size=32, fp16.
    Returns (trainer, tokenizer, model).
    """
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
        TrainingArguments, Trainer,
    )
    save_dir = Path(save_dir or CLASSIFIER_SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = DistilBertTokenizerFast.from_pretrained(CLASSIFIER_MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        CLASSIFIER_MODEL_NAME, num_labels=AG_NEWS_NUM_CLASSES,
    )

    # Sample dataset for fast demonstration training (prevents Colab timeouts)
    # 10,000 train / 1,500 test samples are enough for ~90%+ F1 score in < 45 seconds on T4 GPU.
    train_dataset = dataset["train"].select(range(min(10000, len(dataset["train"]))))
    test_dataset = dataset["test"].select(range(min(1500, len(dataset["test"]))))

    tokenized_train = tokenize_dataset(train_dataset, tokenizer)
    tokenized_test = tokenize_dataset(test_dataset, tokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"
    logger.info(f"Device: {device}, FP16: {use_fp16}")

    training_args = TrainingArguments(
        output_dir=str(save_dir),
        num_train_epochs=CLASSIFIER_EPOCHS,
        per_device_train_batch_size=CLASSIFIER_BATCH_SIZE,
        per_device_eval_batch_size=CLASSIFIER_EVAL_BATCH_SIZE,
        learning_rate=CLASSIFIER_LEARNING_RATE,
        weight_decay=CLASSIFIER_WEIGHT_DECAY,
        warmup_ratio=CLASSIFIER_WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=use_fp16,
        report_to="none",
        seed=RANDOM_SEED,
        logging_steps=100,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=tokenized_train, eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    logger.info(f"Model saved to {save_dir}")
    return trainer, tokenizer, model


def evaluate_classifier(trainer, tokenized_test) -> dict:
    """Run full evaluation. Returns predictions, F1, confusion matrix, report."""
    preds_output = trainer.predict(tokenized_test)
    y_pred = np.argmax(preds_output.predictions, axis=1)
    y_true = preds_output.label_ids
    return {
        "y_pred": y_pred, "y_true": y_true,
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(
            y_true, y_pred, target_names=AG_NEWS_CLASSES, digits=4,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def load_classifier(model_dir: Optional[Path] = None):
    """Load a trained classifier from disk. Returns (model, tokenizer, device)."""
    from transformers import (
        DistilBertForSequenceClassification, DistilBertTokenizerFast,
    )
    model_dir = model_dir or CLASSIFIER_SAVE_DIR
    if not Path(model_dir).exists():
        raise FileNotFoundError(f"Model not found at {model_dir}.")
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir))
    model = DistilBertForSequenceClassification.from_pretrained(
        str(model_dir),
        attn_implementation="eager"
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device


def predict(text: str, model, tokenizer, device: str) -> dict:
    """Classify a single article. Returns class, confidence, all probabilities."""
    inputs = tokenizer(
        text, truncation=True, padding="max_length",
        max_length=CLASSIFIER_MAX_LENGTH, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    pred_class = int(np.argmax(probs))
    return {
        "predicted_class": pred_class,
        "predicted_label": AG_NEWS_CLASSES[pred_class],
        "confidence": float(probs[pred_class]),
        "probabilities": {AG_NEWS_CLASSES[i]: float(p) for i, p in enumerate(probs)},
    }


def get_attention(text: str, model, tokenizer, device: str, layer: int = -1) -> dict:
    """
    Extract [CLS] token attention weights (averaged across heads) for visualization.
    Shows which words the model focuses on for classification.
    """
    inputs = tokenizer(
        text, truncation=True, padding=False,
        max_length=CLASSIFIER_MAX_LENGTH, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attn = outputs.attentions[layer].squeeze(0).mean(dim=0).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())
    return {
        "tokens": tokens, "attention_weights": attn[0],
        "layer_used": layer if layer >= 0 else len(outputs.attentions) + layer,
    }
