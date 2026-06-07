"""
NewsIQ — Sentence Embeddings
Compute, cache, and load sentence embeddings using SentenceTransformers.
Embeddings are reused across BERTopic, UMAP visualization, and similarity search.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from src.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
    EMBEDDINGS_FILE,
    RANDOM_SEED
)

logger = logging.getLogger(__name__)


def compute_embeddings(
    documents: list[str],
    model_name: str = EMBEDDING_MODEL_NAME,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    save_path: Optional[Path] = EMBEDDINGS_FILE,
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Compute sentence embeddings for a list of documents.
    Uses SentenceTransformers (all-MiniLM-L6-v2 by default — 384-dim, fast, good quality).

    Embedding computation is the most expensive step (~20 min for 50K on Colab T4).
    Results are cached to disk and reused everywhere (BERTopic, UMAP, similarity search).

    Args:
        documents: list of text strings to embed
        model_name: SentenceTransformer model name
        batch_size: encoding batch size (256 works well on T4)
        save_path: path to save .npy file (None to skip saving)
        force_recompute: if True, recompute even if cache exists

    Returns:
        numpy array of shape (len(documents), embedding_dim)
    """
    # Return cached if available
    if save_path and save_path.exists() and not force_recompute:
        logger.info(f"Loading cached embeddings from {save_path}")
        embeddings = np.load(save_path)
        logger.info(f"Loaded embeddings: shape={embeddings.shape}")
        return embeddings

    logger.info(f"Computing embeddings with {model_name}...")
    logger.info(f"  Documents: {len(documents)}, Batch size: {batch_size}")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    # Encode in batches to manage memory
    # SentenceTransformer handles batching internally, but we log progress
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize — better for cosine similarity
    )

    logger.info(f"Embeddings computed: shape={embeddings.shape}")

    # Save to disk
    if save_path:
        np.save(save_path, embeddings)
        logger.info(f"Embeddings saved to {save_path}")

    return embeddings


def load_embeddings(path: Path = EMBEDDINGS_FILE) -> np.ndarray:
    """Load pre-computed embeddings from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {path}. "
            "Run compute_embeddings() first, or use the Colab notebook."
        )
    embeddings = np.load(path)
    logger.info(f"Loaded embeddings: shape={embeddings.shape}")
    return embeddings
