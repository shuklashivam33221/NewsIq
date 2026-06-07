"""
NewsIQ — Central Configuration
All hyperparameters, paths, and constants in one place.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────── Paths ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = DATA_DIR / "cache"

# Create directories if they don't exist
for d in [DATA_DIR, MODELS_DIR, CACHE_DIR, DATA_DIR / "raw", DATA_DIR / "processed"]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────── API Keys ────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")

# ──────────────────────────── Dataset ────────────────────────────
# AG News — for classification (4 classes, 120K articles, pre-labeled)
AG_NEWS_DATASET = "fancyzhx/ag_news"
AG_NEWS_CLASSES = ["World", "Sports", "Business", "Sci/Tech"]
AG_NEWS_NUM_CLASSES = 4

# CC-News — for topic modeling + summarization (unlabeled, large corpus)
CC_NEWS_DATASET = "vblagoje/cc_news"
CC_NEWS_SAMPLE_SIZE = 10_000       # Number of articles to use (optimized for speed)
CC_NEWS_MIN_LENGTH = 100           # Minimum characters per article
CC_NEWS_MAX_LENGTH = 5000          # Maximum characters per article (truncate beyond)

# ──────────────────────────── Classification (DistilBERT) ────────────────────────────
CLASSIFIER_MODEL_NAME = "distilbert-base-uncased"
CLASSIFIER_MAX_LENGTH = 128        # AG News is short text; 128 covers >95%
CLASSIFIER_EPOCHS = 2
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_EVAL_BATCH_SIZE = 64
CLASSIFIER_LEARNING_RATE = 2e-5
CLASSIFIER_WEIGHT_DECAY = 0.01
CLASSIFIER_WARMUP_RATIO = 0.1
CLASSIFIER_SAVE_DIR = MODELS_DIR / "distilbert-news"

# ──────────────────────────── Embeddings ────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality
EMBEDDING_BATCH_SIZE = 256
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"

# ──────────────────────────── Topic Modeling (BERTopic) ────────────────────────────
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 5              # For HDBSCAN input (not visualization)
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"

HDBSCAN_MIN_CLUSTER_SIZE = 100
HDBSCAN_MIN_SAMPLES = 15

TOPIC_MODEL_DIR = MODELS_DIR / "bertopic"
TOPIC_TOP_N_WORDS = 10

# ──────────────────────────── UMAP Visualization ────────────────────────────
UMAP_VIZ_N_NEIGHBORS = 30         # Higher for visualization (more global structure)
UMAP_VIZ_MIN_DIST = 0.1           # Moderate spacing for readability
UMAP_VIZ_FILE = DATA_DIR / "umap_2d.npy"

# ──────────────────────────── Summarization ────────────────────────────
SUMMARIZER_MODEL = "facebook/bart-large-cnn"
SUMMARIZER_API_URL = f"https://api-inference.huggingface.co/models/{SUMMARIZER_MODEL}"
SUMMARY_MAX_LENGTH = 130
SUMMARY_MIN_LENGTH = 30
SUMMARY_MAX_INPUT_CHARS = 3000     # BART max ≈ 1024 tokens ≈ 3000 chars
SUMMARY_CACHE_DIR = CACHE_DIR / "summaries"
SUMMARY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
NUM_PRECOMPUTED_SUMMARIES = 100    # Pre-compute for demo speed

# ──────────────────────────── Random State ────────────────────────────
RANDOM_SEED = 42
