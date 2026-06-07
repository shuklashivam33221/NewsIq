"""
NewsIQ — Data Loading & Preprocessing
Handles AG News (classification) and CC-News (topic modeling / summarization).
"""
import re
import logging
from typing import Optional

import numpy as np
import pandas as pd
from datasets import load_dataset
from langdetect import detect, LangDetectException
from tqdm import tqdm

from src.config import (
    AG_NEWS_DATASET, AG_NEWS_CLASSES,
    CC_NEWS_DATASET, CC_NEWS_SAMPLE_SIZE,
    CC_NEWS_MIN_LENGTH, CC_NEWS_MAX_LENGTH,
    DATA_DIR, RANDOM_SEED
)

logger = logging.getLogger(__name__)


# ──────────────────────────── Text Cleaning ────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean a raw article text. Rules:
    - Remove HTML tags (common in CC-News)
    - Remove URLs
    - Remove email addresses
    - Normalize whitespace (collapse multiple spaces/newlines)
    - Strip leading/trailing whitespace
    - Do NOT lowercase — DistilBERT tokenizer handles casing
    - Do NOT remove punctuation — BERT needs it for tokenization
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # Normalize unicode whitespace + collapse
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_english(text: str) -> bool:
    """Detect if text is English. Returns False on detection failure."""
    try:
        return detect(text[:500]) == "en"
    except LangDetectException:
        return False


# ──────────────────────────── AG News (Classification) ────────────────────────────

def load_ag_news() -> dict:
    """
    Load AG News dataset from HuggingFace.
    Returns dict with 'train' and 'test' splits.
    AG News is already clean and balanced (30K per class).

    Returns:
        dict with keys 'train', 'test' — each a HF Dataset object
        with columns: 'text', 'label'
    """
    logger.info("Loading AG News dataset...")
    dataset = load_dataset(AG_NEWS_DATASET)

    # Verify integrity
    train_size = len(dataset["train"])
    test_size = len(dataset["test"])
    logger.info(f"AG News loaded: {train_size} train, {test_size} test")
    logger.info(f"Classes: {AG_NEWS_CLASSES}")

    # Quick stats
    for split_name in ["train", "test"]:
        labels = dataset[split_name]["label"]
        unique, counts = np.unique(labels, return_counts=True)
        for cls_id, count in zip(unique, counts):
            logger.info(f"  {split_name} — {AG_NEWS_CLASSES[cls_id]}: {count}")

    return dataset


# ──────────────────────────── CC-News (Topic Modeling / Summarization) ────────────────────────────

def load_cc_news(
    num_articles: int = CC_NEWS_SAMPLE_SIZE,
    cache: bool = True
) -> pd.DataFrame:
    """
    Load and clean CC-News articles for topic modeling and summarization.
    Uses streaming to avoid downloading the entire dataset.

    Filters applied:
    - English only (langdetect)
    - Minimum length: CC_NEWS_MIN_LENGTH chars
    - Maximum length: truncated to CC_NEWS_MAX_LENGTH chars
    - Deduplication by title

    Args:
        num_articles: target number of articles to collect
        cache: if True, save/load from parquet cache

    Returns:
        DataFrame with columns: 'title', 'text', 'text_clean', 'date', 'url'
    """
    cache_path = DATA_DIR / "processed" / "cc_news_clean.parquet"

    # Return cached version if available
    if cache and cache_path.exists():
        logger.info(f"Loading cached CC-News from {cache_path}")
        df = pd.read_parquet(cache_path)
        logger.info(f"Loaded {len(df)} cached articles")
        return df

    logger.info(f"Streaming CC-News, collecting {num_articles} articles...")
    dataset = load_dataset(CC_NEWS_DATASET, split="train", streaming=True)

    articles = []
    seen_titles = set()
    skipped = {"short": 0, "non_english": 0, "duplicate": 0, "empty": 0}

    for item in tqdm(dataset, desc="Collecting articles", total=num_articles * 2):
        if len(articles) >= num_articles:
            break

        text = item.get("text", "")
        title = item.get("title", "")

        # Skip empty
        if not text or len(text.strip()) < 10:
            skipped["empty"] += 1
            continue

        # Deduplicate by title
        title_key = title.strip().lower()
        if title_key in seen_titles:
            skipped["duplicate"] += 1
            continue

        # Clean text
        text_clean = clean_text(text)

        # Skip too short
        if len(text_clean) < CC_NEWS_MIN_LENGTH:
            skipped["short"] += 1
            continue

        # English only
        if not is_english(text_clean):
            skipped["non_english"] += 1
            continue

        # Truncate long articles
        if len(text_clean) > CC_NEWS_MAX_LENGTH:
            text_clean = text_clean[:CC_NEWS_MAX_LENGTH]

        seen_titles.add(title_key)
        articles.append({
            "title": title.strip(),
            "text": text[:CC_NEWS_MAX_LENGTH],    # Original (truncated)
            "text_clean": text_clean,               # Cleaned version
            "date": item.get("date", ""),
            "url": item.get("url", ""),
        })

    logger.info(f"Collected {len(articles)} articles")
    logger.info(f"Skipped: {skipped}")

    df = pd.DataFrame(articles)

    # Cache to disk
    if cache:
        df.to_parquet(cache_path, index=False)
        logger.info(f"Cached to {cache_path}")

    return df


# ──────────────────────────── Dataset Statistics ────────────────────────────

def get_ag_news_stats(dataset: dict) -> dict:
    """Compute summary statistics for AG News dataset."""
    stats = {}
    for split_name in ["train", "test"]:
        split = dataset[split_name]
        texts = split["text"]
        labels = split["label"]

        lengths = [len(t.split()) for t in texts]
        unique, counts = np.unique(labels, return_counts=True)

        stats[split_name] = {
            "total": len(texts),
            "class_distribution": {
                AG_NEWS_CLASSES[int(c)]: int(n) for c, n in zip(unique, counts)
            },
            "avg_word_count": float(np.mean(lengths)),
            "max_word_count": int(np.max(lengths)),
            "min_word_count": int(np.min(lengths)),
        }

    return stats


def get_cc_news_stats(df: pd.DataFrame) -> dict:
    """Compute summary statistics for the cleaned CC-News corpus."""
    lengths = df["text_clean"].str.split().str.len()
    return {
        "total_articles": len(df),
        "avg_word_count": float(lengths.mean()),
        "max_word_count": int(lengths.max()),
        "min_word_count": int(lengths.min()),
        "has_dates": int(df["date"].astype(bool).sum()),
        "unique_titles": df["title"].nunique(),
    }
