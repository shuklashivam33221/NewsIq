"""
NewsIQ — Complete Training Pipeline
Run this script on Google Colab (T4 GPU) to train all models and generate artifacts.
Upload the entire NewsIQ/ folder to Colab, then run: !python train_pipeline.py

This script executes the full pipeline:
1. Load & preprocess AG News + CC-News datasets
2. Compute sentence embeddings (all-MiniLM-L6-v2)
3. Fine-tune DistilBERT classifier on AG News
4. Evaluate classifier (F1, confusion matrix)
5. Fit BERTopic on CC-News corpus
6. Compute UMAP 2D projection for visualization
7. Save all artifacts to data/ and models/ directories

After running, download data/ and models/ folders back to your local machine
for the Streamlit dashboard.
"""
import sys
import os
import json
import logging

import numpy as np

# ──────────────────────────── Monkey Patch JSON for NumPy Types ────────────────────────────
_original_json_dump = json.dump
_original_json_dumps = json.dumps

def custom_json_default(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def patched_json_dump(*args, **kwargs):
    if 'default' not in kwargs:
        kwargs['default'] = custom_json_default
    return _original_json_dump(*args, **kwargs)

def patched_json_dumps(*args, **kwargs):
    if 'default' not in kwargs:
        kwargs['default'] = custom_json_default
    return _original_json_dumps(*args, **kwargs)

json.dump = patched_json_dump
json.dumps = patched_json_dumps

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("NewsIQ")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import DATA_DIR, MODELS_DIR, EMBEDDINGS_FILE, UMAP_VIZ_FILE


def step_1_load_data():
    """Load AG News and CC-News datasets."""
    logger.info("=" * 60)
    logger.info("STEP 1: Loading Datasets")
    logger.info("=" * 60)

    from src.data_loader import load_ag_news, load_cc_news, get_ag_news_stats

    # AG News (classification — clean, labeled, fast)
    ag_news = load_ag_news()
    stats = get_ag_news_stats(ag_news)
    logger.info(f"AG News stats: {json.dumps(stats, indent=2)}")

    # CC-News (topic modeling — streaming, needs filtering)
    cc_news_df = load_cc_news(num_articles=50000)
    logger.info(f"CC-News loaded: {len(cc_news_df)} articles")

    return ag_news, cc_news_df


def step_2_compute_embeddings(cc_news_df):
    """Compute sentence embeddings for CC-News (reused by BERTopic + UMAP)."""
    logger.info("=" * 60)
    logger.info("STEP 2: Computing Sentence Embeddings")
    logger.info("=" * 60)

    from src.embeddings import compute_embeddings

    documents = cc_news_df["text_clean"].tolist()
    embeddings = compute_embeddings(documents)
    logger.info(f"Embeddings shape: {embeddings.shape}")
    return documents, embeddings


def step_3_train_classifier(ag_news):
    """Fine-tune DistilBERT on AG News."""
    logger.info("=" * 60)
    logger.info("STEP 3: Training DistilBERT Classifier")
    logger.info("=" * 60)

    from src.config import CLASSIFIER_SAVE_DIR

    eval_path = DATA_DIR / "eval_results.json"
    if eval_path.exists() and (CLASSIFIER_SAVE_DIR / "model.safetensors").exists():
        logger.info("Trained classifier and evaluation results already exist. Loading cached results...")
        with open(eval_path) as f:
            return json.load(f)

    from src.classifier import train_classifier, evaluate_classifier, tokenize_dataset
    from transformers import DistilBertTokenizerFast
    from src.config import CLASSIFIER_MODEL_NAME

    trainer, tokenizer, model = train_classifier(ag_news)

    # Evaluate
    tokenized_test = tokenize_dataset(ag_news["test"], tokenizer)
    results = evaluate_classifier(trainer, tokenized_test)

    # Save evaluation results for Streamlit
    eval_save = {
        "f1_macro": float(results["f1_macro"]),
        "accuracy": float(results["accuracy"]),
        "classification_report": results["classification_report"],
        "confusion_matrix": results["confusion_matrix"].tolist(),
    }
    with open(eval_path, "w") as f:
        json.dump(eval_save, f, indent=2)
    logger.info(f"Eval results saved to {eval_path}")

    return eval_save


def step_4_topic_modeling(documents, embeddings):
    """Fit BERTopic on CC-News corpus."""
    logger.info("=" * 60)
    logger.info("STEP 4: BERTopic Topic Modeling")
    logger.info("=" * 60)

    from src.topic_model import fit_topic_model, get_topic_summary

    topic_model, topics, probs = fit_topic_model(documents, embeddings)

    # Save topics array for Streamlit
    np.save(DATA_DIR / "topics.npy", np.array(topics))

    # Save topic summary
    summary = get_topic_summary(topic_model)
    with open(DATA_DIR / "topic_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Topics saved. Discovered {len(summary)} topics.")
    return topic_model, topics


def step_5_umap_visualization(embeddings):
    """Compute UMAP 2D projection."""
    logger.info("=" * 60)
    logger.info("STEP 5: UMAP 2D Projection")
    logger.info("=" * 60)

    from src.visualizations import compute_umap_2d

    projection = compute_umap_2d(embeddings)
    logger.info(f"UMAP projection shape: {projection.shape}")
    return projection


def main():
    logger.info("🧠 NewsIQ Training Pipeline")
    logger.info("=" * 60)

    # Step 1: Load data
    ag_news, cc_news_df = step_1_load_data()

    # Step 2: Compute embeddings
    documents, embeddings = step_2_compute_embeddings(cc_news_df)

    # Step 3: Train classifier
    eval_results = step_3_train_classifier(ag_news)

    # Step 4: Topic modeling
    topic_model, topics = step_4_topic_modeling(documents, embeddings)

    # Step 5: UMAP visualization
    projection = step_5_umap_visualization(embeddings)

    # Summary
    logger.info("=" * 60)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"F1 Score: {eval_results['f1_macro']:.4f}")
    logger.info(f"Topics:   {len(set(topics)) - 1}")
    logger.info(f"Articles: {len(documents)}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Download data/ and models/ folders")
    logger.info("  2. Run: streamlit run app/app.py")


if __name__ == "__main__":
    main()
