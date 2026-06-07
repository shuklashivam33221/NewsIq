"""
NewsIQ — BERTopic Topic Modeling
Unsupervised topic discovery from 50K+ articles using BERTopic.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import (
    UMAP_N_NEIGHBORS, UMAP_N_COMPONENTS, UMAP_MIN_DIST, UMAP_METRIC,
    HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES,
    TOPIC_MODEL_DIR, TOPIC_TOP_N_WORDS, RANDOM_SEED,
)

logger = logging.getLogger(__name__)


def fit_topic_model(
    documents: list[str],
    embeddings: np.ndarray,
    save_dir: Optional[Path] = None,
) -> tuple:
    """
    Fit BERTopic on documents using pre-computed embeddings.
    Pipeline: embeddings → UMAP(5D) → HDBSCAN → c-TF-IDF → topics.

    Args:
        documents: list of article texts
        embeddings: pre-computed sentence embeddings (from embeddings.py)
        save_dir: path to save fitted model

    Returns:
        (topic_model, topics, probs)
    """
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    save_dir = Path(save_dir or TOPIC_MODEL_DIR)

    umap_model = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_SEED,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        prediction_data=True,
    )

    # Add stop words and bigrams for better topic representations
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000,
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        embedding_model=None,  # We pass pre-computed embeddings
        top_n_words=TOPIC_TOP_N_WORDS,
        verbose=True,
    )

    logger.info(f"Fitting BERTopic on {len(documents)} documents...")
    topics, probs = topic_model.fit_transform(documents, embeddings=embeddings)

    # Log summary
    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info[topic_info["Topic"] != -1])
    n_outliers = int((np.array(topics) == -1).sum())
    outlier_pct = n_outliers / len(topics) * 100

    logger.info(f"Topics discovered: {n_topics}")
    logger.info(f"Outliers: {n_outliers} ({outlier_pct:.1f}%)")

    # Save model
    save_dir.mkdir(parents=True, exist_ok=True)
    topic_model.save(str(save_dir), serialization="safetensors",
                     save_ctfidf=True, save_embedding_model=False)
    logger.info(f"Topic model saved to {save_dir}")

    return topic_model, topics, probs


def load_topic_model(model_dir: Optional[Path] = None):
    """Load a saved BERTopic model."""
    from bertopic import BERTopic
    model_dir = Path(model_dir or TOPIC_MODEL_DIR)
    if not model_dir.exists():
        raise FileNotFoundError(f"Topic model not found at {model_dir}.")
    return BERTopic.load(str(model_dir))


def get_topic_summary(topic_model) -> dict:
    """Get a summary of all discovered topics with keywords and counts."""
    info = topic_model.get_topic_info()
    summary = {}
    for _, row in info.iterrows():
        topic_id = row["Topic"]
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id)
        summary[topic_id] = {
            "name": row.get("Name", f"Topic_{topic_id}"),
            "count": int(row["Count"]),
            "keywords": [w for w, _ in words[:10]],
            "keyword_scores": {w: float(s) for w, s in words[:10]},
        }
    return summary
