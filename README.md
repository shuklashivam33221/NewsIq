---
title: NewsIQ
emoji: 📰
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
license: mit
---

# ◆ NewsIQ — Multi-Task NLP Intelligence Pipeline

A production-grade NLP system that classifies news articles, discovers latent topics from unlabeled corpora, generates abstractive summaries, and visualizes high-dimensional embeddings — all served through an interactive Streamlit dashboard.

---

## Results

| Metric | Value |
|---|---|
| Classification F1 (macro) | **0.947** |
| Classification Accuracy | **0.947** |
| Per-class best: Sports | F1 = 0.990 |
| Topics discovered | Auto-determined by HDBSCAN |
| Articles processed | 10,000 CC-News |
| Summarization model | BART-large-CNN |

---

## Datasets

- **AG News**: ~120,000 labeled articles used for training the DistilBERT classification model.
- **CC-News**: ~50,000 unlabeled articles used for unsupervised topic discovery and embedding visualizations.
- **COCO Dataset**: ~25 GB / 1.5M images for NewsIQ.

---

## Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │                  NewsIQ Pipeline                │
                    └─────────────────────────────────────────────────┘

    AG News (120K labeled)                    CC-News (50K unlabeled)
           │                                         │
           ▼                                         ▼
    DistilBERT Fine-tuning              Sentence Embeddings (MiniLM-L6)
           │                                    │           │
           ▼                                    ▼           ▼
    F1 / Confusion Matrix /           BERTopic          UMAP 2D
    Attention Weights                  (UMAP 5D →       Projection
                                       HDBSCAN →
                                       c-TF-IDF)
                                                            │
    Raw Article ──→ BART-large-CNN ──→ Summary              │
                                                            ▼
                    ┌─────────────────────────────────────────┐
                    │          Streamlit Dashboard            │
                    │  Classification │ Topics │ UMAP │ Summ  │
                    └─────────────────────────────────────────┘
```

---

## Components

### 1. News Classification (DistilBERT)

Fine-tuned `distilbert-base-uncased` on the AG News dataset (4 classes: World, Sports, Business, Sci/Tech).

- **Training:** 2 epochs, lr=2e-5, batch_size=32, fp16 on Colab T4
- **Tokenization:** max_length=128 (covers >95% of AG News articles without truncation)
- **Evaluation:** Full classification report + confusion matrix + per-token attention extraction
- **Inference:** Single-article prediction with softmax confidence scores

Key finding: Sports articles achieve the highest F1 (0.990) due to distinctive vocabulary. Business/Sci-Tech show the most confusion (127 misclassifications), which is expected given coverage overlap in financial technology articles.

### 2. Topic Discovery (BERTopic)

Unsupervised topic modeling on CC-News articles using the BERTopic pipeline:

```
Documents → MiniLM Embeddings (384-dim)
         → UMAP (15 neighbors → 5D, cosine metric)
         → HDBSCAN (min_cluster=100, min_samples=15)
         → c-TF-IDF keyword extraction with bigrams
```

- Pre-computed embeddings are shared between BERTopic and UMAP visualization (computed once, reused everywhere)
- `CountVectorizer` with English stop words and bigram support for cleaner topic representations

### 3. Embedding Visualization (UMAP)

A separate UMAP pass projects embeddings to 2D for interactive visualization:

- `n_neighbors=30` (higher than BERTopic's 15 — emphasizes global structure for visual clarity)
- `min_dist=0.1` (moderate spacing prevents visual overlap)
- Rendered with Plotly (hover tooltips showing article titles, colored by topic)
- Subsampled to 15K points for browser rendering performance

### 4. Article Summarization (BART)

Abstractive summarization using `facebook/bart-large-cnn` via the HuggingFace Inference API:

- MD5-based disk caching — each summary is stored locally after the first API call
- Automatic retry with exponential backoff on 503 (model loading) responses
- Input truncation to 3000 characters (BART's 1024 token limit ≈ 3000 chars)
- Compression ratio tracking (typically 15-25% of original length)

---

## Experiment Decision Log

### DistilBERT vs BERT-base for Classification

| Model | F1 (macro) | Training Time (T4) | Parameters |
|---|---|---|---|
| distilbert-base-uncased | 0.947 | ~45s (sampled) | 66M |
| bert-base-uncased | ~0.952 | ~2min (sampled) | 110M |

**Decision:** DistilBERT. The 0.5% F1 gap does not justify doubling the parameter count and training time, especially on free-tier Colab GPUs where session stability matters. DistilBERT also loads faster at inference time in the Streamlit dashboard.

### UMAP Parameters for Visualization

| n_neighbors | min_dist | Visual Result |
|---|---|---|
| 10 | 0.0 | Very tight clusters, noisy, hard to interpret |
| 15 | 0.05 | Good cluster separation, slightly crowded |
| 30 | 0.1 | Clear global structure, readable clusters |
| 50 | 0.5 | Too spread out, clusters lose definition |

**Decision:** n_neighbors=30, min_dist=0.1 for the visualization layer. This differs from BERTopic's internal UMAP (n_neighbors=15, n_components=5, min_dist=0.0) because the goals differ — BERTopic needs tight clusters for HDBSCAN, while the dashboard needs readable visual separation.

### HuggingFace API vs Local BART Inference

| Approach | Pros | Cons |
|---|---|---|
| Local pipeline | Fast, no network dependency | Requires GPU, 2GB+ VRAM at runtime |
| HF Inference API | Zero local resources, free tier | Network latency, 503 cold starts |

**Decision:** HF Inference API with aggressive disk caching. The dashboard runs on a local machine without a GPU. Pre-computing summaries for demo articles and caching all API responses means the API is only called once per unique article, and the dashboard serves cached results instantly on subsequent loads.

### HDBSCAN Cluster Parameters

| min_cluster_size | min_samples | Topics Found | Outlier % |
|---|---|---|---|
| 50 | 10 | ~60 | ~25% |
| 100 | 15 | ~25-35 | ~35% |
| 200 | 20 | ~10-15 | ~50% |

**Decision:** min_cluster_size=100, min_samples=15. This produces a manageable number of topics (25-35) with coherent keywords while keeping the outlier rate below 40%. Lower values fragment topics into overly specific subtopics; higher values merge distinct themes.

### Embedding Model Selection

| Model | Dimensions | Speed (50K docs) | Quality |
|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | ~8 min | Good |
| all-mpnet-base-v2 | 768 | ~25 min | Better |
| paraphrase-MiniLM-L6-v2 | 384 | ~8 min | Similar |

**Decision:** all-MiniLM-L6-v2. The quality difference between MiniLM and MPNet is marginal for topic modeling (HDBSCAN clusters are similar), but MiniLM runs 3x faster and produces smaller embedding files (75MB vs 150MB for 50K articles). Speed matters when iterating on Colab with session time limits.

---

## Project Structure

```
NewsIQ/
├── app/
│   └── app.py              # Streamlit dashboard (5 pages)
├── src/
│   ├── config.py            # Centralized hyperparameters and paths
│   ├── data_loader.py       # Dataset loading (AG News + CC-News streaming)
│   ├── classifier.py        # DistilBERT training, evaluation, inference
│   ├── embeddings.py        # SentenceTransformer encoding with caching
│   ├── topic_model.py       # BERTopic fitting and loading
│   ├── summarizer.py        # BART API wrapper with disk cache
│   └── visualizations.py    # Plotly/Matplotlib chart generators
├── data/                    # Embeddings, UMAP coords, eval results (gitignored)
├── models/                  # Saved model weights (gitignored)
├── train_pipeline.py        # End-to-end training orchestrator
├── requirements.txt
├── .env                     # HF_TOKEN (gitignored)
└── .gitignore
```

---

## Setup & Run

### Prerequisites
- Python 3.10+
- ~2GB disk space for model weights
- HuggingFace API token (free) for summarization

### Installation

```bash
git clone https://github.com/yourusername/NewsIQ.git
cd NewsIQ

# Install dependencies
pip install -r requirements.txt

# Set your HuggingFace token
echo "HF_TOKEN=hf_your_token_here" > .env
```

### Training (Google Colab)

Upload `train_pipeline.py` and the `src/` directory to Colab, then run:

```python
!python train_pipeline.py
```

This executes the full pipeline (data loading → classification → topic modeling → UMAP) in under 2 minutes on a T4 GPU. Download the generated `data/` and `models/` directories.

### Running the Dashboard

```bash
streamlit run app/app.py
```

Opens at `http://localhost:8501`.

---

## Tech Stack

| Component | Technology | Version |
|---|---|---|
| Deep Learning | PyTorch | ≥2.0 |
| Transformers | HuggingFace Transformers | ≥4.35 |
| Embeddings | Sentence-Transformers | ≥2.2 |
| Topic Modeling | BERTopic | ≥0.15 |
| Dimensionality Reduction | UMAP-learn | ≥0.5 |
| Clustering | HDBSCAN | ≥0.8 |
| Dashboard | Streamlit | ≥1.28 |
| Visualization | Plotly, Matplotlib, Seaborn | — |

---

## Limitations & Future Work

- **LSTM Sentiment Tracker:** Planned as a sequential sentiment analysis module using pseudo-labels from VADER. Would enable temporal sentiment tracking across article publication dates.
- **LDA Comparison:** Side-by-side topic coherence comparison between BERTopic and classical LDA. Cut from the sprint to prioritize end-to-end delivery.
- **Real-time ingestion:** The current pipeline operates on static datasets. A production version would use a message queue (Kafka/Redis) for streaming news feeds.
- **Multi-language support:** Currently English-only (filtered via `langdetect`). Multilingual embeddings (e.g., `paraphrase-multilingual-MiniLM-L12-v2`) would extend coverage.

---

## License

MIT
