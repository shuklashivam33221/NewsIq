"""
NewsIQ — Visualization Utilities
Confusion matrix, attention heatmap, UMAP scatter, and topic charts.
"""
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.config import AG_NEWS_CLASSES, UMAP_VIZ_FILE, RANDOM_SEED

logger = logging.getLogger(__name__)


# ──────────────────────────── Classification Visuals ────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, labels=AG_NEWS_CLASSES) -> plt.Figure:
    """Create a styled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
        linewidths=0.5, linecolor="white",
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_class_distribution(labels: np.ndarray, class_names=AG_NEWS_CLASSES) -> plt.Figure:
    """Bar chart of class distribution."""
    unique, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#667eea", "#764ba2", "#f093fb", "#4facfe"]
    bars = ax.bar([class_names[i] for i in unique], counts, color=colors[:len(unique)])
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                str(count), ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    return fig


def plot_confidence_bars(probabilities: dict) -> go.Figure:
    """Plotly horizontal bar chart showing prediction confidence per class."""
    classes = list(probabilities.keys())
    values = list(probabilities.values())
    colors = ["#667eea", "#764ba2", "#f093fb", "#4facfe"]

    fig = go.Figure(go.Bar(
        x=values, y=classes, orientation="h",
        marker_color=colors[:len(classes)],
        text=[f"{v:.1%}" for v in values], textposition="auto",
    ))
    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Probability", yaxis_title="",
        height=300, margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(range=[0, 1]),
    )
    return fig


# ──────────────────────────── Attention Visualization ────────────────────────────

def plot_attention_heatmap(tokens: list, weights: np.ndarray, top_k: int = 30) -> plt.Figure:
    """
    Bar chart showing [CLS] attention to top-k tokens.
    Filters out [CLS], [SEP], [PAD] for cleaner visualization.
    """
    # Filter special tokens
    filtered = [
        (t, w) for t, w in zip(tokens, weights)
        if t not in ["[CLS]", "[SEP]", "[PAD]"]
    ]
    if not filtered:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No tokens to display", ha="center")
        return fig

    tokens_f, weights_f = zip(*filtered)
    # Take top-k by attention weight
    indices = np.argsort(weights_f)[-top_k:]
    top_tokens = [tokens_f[i] for i in indices]
    top_weights = [weights_f[i] for i in indices]

    fig, ax = plt.subplots(figsize=(10, max(6, len(top_tokens) * 0.3)))
    colors = plt.cm.YlOrRd(np.array(top_weights) / max(top_weights))
    ax.barh(range(len(top_tokens)), top_weights, color=colors)
    ax.set_yticks(range(len(top_tokens)))
    ax.set_yticklabels(top_tokens, fontsize=10)
    ax.set_xlabel("Attention Weight", fontsize=12)
    ax.set_title("[CLS] Token Attention (Last Layer)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ──────────────────────────── UMAP Visualization ────────────────────────────

def compute_umap_2d(embeddings: np.ndarray, save_path: Path = UMAP_VIZ_FILE) -> np.ndarray:
    """Compute UMAP 2D projection for visualization (separate from BERTopic's 5D)."""
    from umap import UMAP

    if save_path and save_path.exists():
        logger.info(f"Loading cached UMAP from {save_path}")
        return np.load(save_path)

    logger.info("Computing UMAP 2D projection...")
    reducer = UMAP(
        n_neighbors=30, min_dist=0.1, n_components=2,
        metric="cosine", random_state=RANDOM_SEED,
    )
    projection = reducer.fit_transform(embeddings)

    if save_path:
        np.save(save_path, projection)
        logger.info(f"UMAP projection saved to {save_path}")

    return projection


def plot_umap_scatter(
    projection: np.ndarray,
    labels: list,
    titles: list = None,
    color_label: str = "Topic",
    max_points: int = 15000,
) -> go.Figure:
    """Interactive Plotly UMAP scatter plot with hover info."""
    n = len(projection)
    if n > max_points:
        # Subsample for rendering performance
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(n, max_points, replace=False)
        projection = projection[idx]
        labels = [labels[i] for i in idx]
        titles = [titles[i] for i in idx] if titles else None

    df = pd.DataFrame({
        "x": projection[:, 0], "y": projection[:, 1],
        color_label: labels,
    })
    if titles:
        df["Title"] = [t[:80] + "..." if len(t) > 80 else t for t in titles]

    hover_data = ["Title"] if titles else None
    fig = px.scatter(
        df, x="x", y="y", color=color_label,
        hover_data=hover_data, opacity=0.5,
        title=f"UMAP Projection ({len(df)} articles)",
    )
    fig.update_layout(
        width=900, height=700,
        xaxis_title="UMAP-1", yaxis_title="UMAP-2",
        legend_title=color_label,
    )
    fig.update_traces(marker_size=3)
    return fig
