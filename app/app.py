"""
NewsIQ — Interactive NLP Intelligence Dashboard
Built as a multi-task pipeline exploration tool for news understanding.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import json
import time

from src.config import (
    AG_NEWS_CLASSES, CLASSIFIER_SAVE_DIR, TOPIC_MODEL_DIR,
    DATA_DIR, MODELS_DIR, EMBEDDINGS_FILE, UMAP_VIZ_FILE,
)

# ──────────────────────────── Page Config ────────────────────────────

st.set_page_config(
    page_title="NewsIQ · NLP Pipeline",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────── Design System ────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --accent-cyan: #0891b2;
        --accent-violet: #7c3aed;
        --accent-amber: #d97706;
        --accent-emerald: #059669;
        --accent-rose: #e11d48;
    }

    * { font-family: 'Space Grotesk', sans-serif; }
    code, pre, .mono { font-family: 'JetBrains Mono', monospace; }

    .main .block-container { padding-top: 1.5rem; max-width: 1280px; }

    div[data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem;
        padding: 6px 0;
    }

    /* Header block */
    .niq-header { padding: 1.5rem 0 1rem 0; }
    .niq-logo {
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .niq-logo span { color: var(--accent-cyan); }
    .niq-sub {
        opacity: 0.65;
        font-size: 0.95rem;
        margin-top: 2px;
        font-weight: 400;
    }

    /* Stat cards — theme adaptive */
    .stat-row { display: flex; gap: 16px; margin: 1.5rem 0; flex-wrap: wrap; }
    .stat-card {
        flex: 1;
        min-width: 160px;
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 12px;
        padding: 20px 24px;
    }
    .stat-val {
        font-size: 1.9rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.2;
    }
    .stat-val.cyan { color: var(--accent-cyan); }
    .stat-val.violet { color: var(--accent-violet); }
    .stat-val.amber { color: var(--accent-amber); }
    .stat-val.emerald { color: var(--accent-emerald); }
    .stat-label {
        opacity: 0.5;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-top: 6px;
    }

    /* Section headers */
    .sec-head {
        font-size: 1.15rem;
        font-weight: 600;
        margin: 2rem 0 0.8rem 0;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
    }

    /* Pipeline diagram */
    .pipe-box {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    .pipe-node {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .pipe-node.active {
        border-color: var(--accent-cyan);
        box-shadow: 0 0 12px rgba(8, 145, 178, 0.2);
    }
    .pipe-arrow { opacity: 0.4; font-size: 1.1rem; }

    /* Chip tags */
    .chip {
        display: inline-block;
        background: rgba(128, 128, 128, 0.08);
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        margin: 2px 3px;
        border: 1px solid rgba(128, 128, 128, 0.15);
        opacity: 0.7;
    }

    /* Result banner */
    .result-banner {
        background: rgba(8, 145, 178, 0.06);
        border: 1px solid rgba(8, 145, 178, 0.2);
        border-radius: 12px;
        padding: 20px 28px;
        margin: 1rem 0;
    }
    .result-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.5;
    }
    .result-value {
        font-size: 1.6rem;
        font-weight: 600;
        color: var(--accent-cyan);
        margin-top: 4px;
    }

    /* Footnote */
    .footnote {
        opacity: 0.45;
        font-size: 0.72rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(128, 128, 128, 0.15);
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────── Sidebar ────────────────────────────

st.sidebar.markdown(
    '<div class="niq-header">'
    '<div class="niq-logo">◆ News<span>IQ</span></div>'
    '<div class="niq-sub">NLP Pipeline Explorer</div>'
    '</div>',
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Section",
    ["Pipeline Overview", "Classification", "Topic Discovery",
     "Embedding Map", "Summarizer"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="padding:8px 0">'
    '<span class="chip">DistilBERT</span><span class="chip">BERTopic</span>'
    '<span class="chip">UMAP</span><span class="chip">BART</span>'
    '<span class="chip">PyTorch</span><span class="chip">Streamlit</span>'
    '</div>',
    unsafe_allow_html=True,
)
st.sidebar.caption("v1.0 · Shivam Shukla")


# ──────────────────────────── Cached Loaders ────────────────────────────

@st.cache_resource
def load_classifier_model():
    from src.classifier import load_classifier
    return load_classifier()

@st.cache_resource
def load_bertopic_model():
    from src.topic_model import load_topic_model
    return load_topic_model()

@st.cache_data
def load_umap_projection():
    if UMAP_VIZ_FILE.exists():
        return np.load(UMAP_VIZ_FILE)
    return None

@st.cache_data
def load_cc_news_data():
    path = DATA_DIR / "processed" / "cc_news_clean.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None

@st.cache_data
def load_eval_results():
    path = DATA_DIR / "eval_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data
def load_topics_data():
    path = DATA_DIR / "topics.npy"
    if path.exists():
        return np.load(path)
    return None


# ──────────────────────────── Helpers ────────────────────────────

def stat_card(value, label, color="cyan"):
    return f'<div class="stat-card"><div class="stat-val {color}">{value}</div><div class="stat-label">{label}</div></div>'


# ──────────────────────────── Page: Overview ────────────────────────────

def render_overview():
    st.markdown(
        '<div class="niq-header">'
        '<div class="niq-logo" style="font-size:2.2rem">◆ News<span>IQ</span></div>'
        '<div class="niq-sub">End-to-end NLP pipeline for news classification, '
        'topic discovery, embedding analysis, and abstractive summarization.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Metrics row
    eval_results = load_eval_results()
    cc_data = load_cc_news_data()
    topics_data = load_topics_data()

    f1 = f"{eval_results['f1_macro']:.3f}" if eval_results else "—"
    acc = f"{eval_results['accuracy']:.3f}" if eval_results else "—"
    n_articles = f"{len(cc_data):,}" if cc_data is not None else "—"
    n_topics = str(len(set(topics_data)) - 1) if topics_data is not None else "—"

    st.markdown(
        '<div class="stat-row">'
        + stat_card(f1, "F1 Score · Macro", "cyan")
        + stat_card(acc, "Accuracy", "violet")
        + stat_card(n_articles, "Articles Processed", "amber")
        + stat_card(n_topics, "Topics Found", "emerald")
        + '</div>',
        unsafe_allow_html=True,
    )

    # Pipeline flow
    st.markdown('<div class="sec-head">Pipeline Architecture</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pipe-box">'
        '<div class="pipe-node">AG News (120K)</div>'
        '<span class="pipe-arrow">→</span>'
        '<div class="pipe-node active">DistilBERT Classifier</div>'
        '<span class="pipe-arrow">→</span>'
        '<div class="pipe-node">F1 / Confusion / Attention</div>'
        '</div>'
        '<div class="pipe-box">'
        '<div class="pipe-node">CC-News (50K)</div>'
        '<span class="pipe-arrow">→</span>'
        '<div class="pipe-node">MiniLM Embeddings</div>'
        '<span class="pipe-arrow">→</span>'
        '<div class="pipe-node active">BERTopic</div>'
        '<span class="pipe-arrow">→</span>'
        '<div class="pipe-node">UMAP 2D Scatter</div>'
        '</div>'
        '<div class="pipe-box">'
        '<div class="pipe-node">Raw Article</div>'
        '<span class="pipe-arrow">→</span>'
        '<div class="pipe-node active">BART-large-CNN</div>'
        '<span class="pipe-arrow">→</span>'
        '<div class="pipe-node">Abstractive Summary</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Two-column architecture breakdown
    st.markdown('<div class="sec-head">Component Details</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**Classification Module**
- Fine-tuned `distilbert-base-uncased` on AG News (4 classes)
- 2 epochs, lr=2e-5, batch=32, fp16 on T4 GPU
- Includes per-token attention extraction for explainability

**Topic Discovery Module**
- Sentence embeddings via `all-MiniLM-L6-v2` (384-dim)
- UMAP reduction (15 neighbors → 5D) → HDBSCAN clustering
- c-TF-IDF keyword extraction with bigram support
        """)
    with col_b:
        st.markdown("""
**Embedding Visualization**
- Separate UMAP pass (30 neighbors → 2D) for interactive scatter
- Plotly-based with hover tooltips and topic coloring
- Subsampled to 15K points for rendering performance

**Summarization Engine**
- BART-large-CNN via HuggingFace Inference API
- MD5-based disk caching to avoid redundant API calls
- Auto-retry with exponential backoff on 503 responses
        """)

    # Footer
    st.markdown(
        '<div class="footnote">'
        'Built with PyTorch, HuggingFace Transformers, BERTopic, UMAP, and Streamlit. '
        'Training executed on Google Colab T4 GPU. '
        'Artifacts transferred and served locally for inference.'
        '</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────── Page: Classification ────────────────────────────

def render_classifier():
    st.markdown(
        '<div class="niq-header">'
        '<div class="niq-logo" style="font-size:1.5rem">Classification</div>'
        '<div class="niq-sub">DistilBERT fine-tuned on AG News — '
        'classify any article into World, Sports, Business, or Sci/Tech</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    tab_predict, tab_eval = st.tabs(["Predict", "Evaluation Metrics"])

    with tab_predict:
        # Example articles for quick testing
        examples = {
            "(paste your own)": "",
            "Sports example": "The Golden State Warriors defeated the Boston Celtics 112-104 in a thrilling overtime game. Stephen Curry scored 38 points including six three-pointers in the fourth quarter.",
            "Tech example": "Researchers at MIT developed a new quantum computing chip that can perform certain calculations 100 times faster than existing processors, marking a breakthrough in the field.",
            "Business example": "Wall Street stocks fell sharply on Friday after the Federal Reserve signaled it may raise interest rates again in September, sending the Dow Jones down 450 points.",
            "World example": "The United Nations Security Council voted unanimously to extend peacekeeping operations in the disputed border region, following weeks of diplomatic negotiations between the two nations.",
        }

        selected_example = st.selectbox("Quick examples:", list(examples.keys()))
        text_input = st.text_area(
            "Article text:",
            value=examples[selected_example],
            height=160,
            placeholder="Paste a news article here to classify it...",
        )

        if st.button("Run Classification", type="primary"):
            if not text_input.strip():
                st.warning("Enter some text first.")
            elif len(text_input.split()) < 5:
                st.warning("Text too short — need at least 5 words.")
            else:
                try:
                    model, tokenizer, device = load_classifier_model()
                    from src.classifier import predict, get_attention
                    from src.visualizations import plot_confidence_bars, plot_attention_heatmap

                    with st.spinner("Running inference..."):
                        result = predict(text_input, model, tokenizer, device)

                    # Result banner
                    conf = result["confidence"]
                    st.markdown(
                        f'<div class="result-banner">'
                        f'<div class="result-label">Predicted Category</div>'
                        f'<div class="result-value">{result["predicted_label"]} — {conf:.1%} confidence</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if conf < 0.6:
                        st.warning("Low confidence — the model is uncertain about this prediction.")

                    # Confidence distribution
                    fig = plot_confidence_bars(result["probabilities"])
                    st.plotly_chart(fig, width="stretch")

                    # Attention
                    with st.expander("Token Attention Weights (what the model focused on)"):
                        attn = get_attention(text_input, model, tokenizer, device)
                        fig_attn = plot_attention_heatmap(attn["tokens"], attn["attention_weights"])
                        st.pyplot(fig_attn)
                        st.caption(f"[CLS] attention from layer {attn['layer_used']} — averaged across all heads")

                except FileNotFoundError:
                    st.error("Model weights not found. Run the training pipeline first.")

    with tab_eval:
        eval_results = load_eval_results()
        if eval_results is None:
            st.info("Evaluation metrics appear here after training.")
            return

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                '<div class="stat-row">'
                + stat_card(f"{eval_results['f1_macro']:.4f}", "F1 Macro", "cyan")
                + stat_card(f"{eval_results['accuracy']:.4f}", "Accuracy", "violet")
                + '</div>',
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown('<div class="sec-head">Classification Report</div>', unsafe_allow_html=True)
            st.code(eval_results["classification_report"], language=None)

        st.markdown('<div class="sec-head">Confusion Matrix</div>', unsafe_allow_html=True)
        from src.visualizations import plot_confusion_matrix
        cm = np.array(eval_results["confusion_matrix"])
        fig = plot_confusion_matrix(cm)
        st.pyplot(fig)

        with st.expander("Interpretation notes"):
            st.markdown("""
- **Sports** has the highest per-class F1 (0.9897) — sports articles use distinctive vocabulary
- **Business vs Sci/Tech** shows the most confusion (127 misclassifications) — expected overlap in financial tech coverage
- **World** news occasionally misclassifies as Business when articles cover trade/economic policy
            """)


# ──────────────────────────── Page: Topics ────────────────────────────

def render_topics():
    st.markdown(
        '<div class="niq-header">'
        '<div class="niq-logo" style="font-size:1.5rem">Topic Discovery</div>'
        '<div class="niq-sub">Unsupervised topic extraction from CC-News using BERTopic '
        '(embeddings → UMAP → HDBSCAN → c-TF-IDF)</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    try:
        topic_model = load_bertopic_model()
    except FileNotFoundError:
        st.info("Topic model not found. Run the training notebook first.")
        return

    cc_data = load_cc_news_data()
    topics_data = load_topics_data()

    # Topic table
    topic_info = topic_model.get_topic_info()
    valid_topics = topic_info[topic_info["Topic"] != -1]

    n_outliers = int(topic_info[topic_info["Topic"] == -1]["Count"].sum()) if -1 in topic_info["Topic"].values else 0
    total = int(topic_info["Count"].sum())

    st.markdown(
        '<div class="stat-row">'
        + stat_card(str(len(valid_topics)), "Topics Found", "emerald")
        + stat_card(f"{n_outliers:,}", "Outlier Articles", "amber")
        + stat_card(f"{n_outliers/max(total,1)*100:.1f}%", "Outlier Rate", "rose")
        + '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec-head">Topic Table</div>', unsafe_allow_html=True)
    st.dataframe(
        valid_topics[["Topic", "Count", "Name"]].head(25),
        width="stretch", hide_index=True,
    )

    # Topic drill-down
    st.markdown('<div class="sec-head">Topic Drill-Down</div>', unsafe_allow_html=True)
    topic_ids = sorted([t for t in topic_info["Topic"] if t != -1])
    selected = st.selectbox("Pick a topic:", topic_ids,
                            format_func=lambda x: f"Topic {x}")

    if selected is not None:
        words = topic_model.get_topic(selected)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top Keywords**")
            for word, score in words[:10]:
                bar_len = int(score / max(w for _, w in words[:10]) * 20)
                bar = "█" * bar_len
                st.markdown(f"`{word}` {bar} {score:.4f}")
        with col2:
            if cc_data is not None and topics_data is not None:
                mask = topics_data == selected
                sample = cc_data[mask].head(5)
                st.markdown("**Sample Articles**")
                for _, row in sample.iterrows():
                    st.markdown(f"- {row['title'][:120]}")

    # BERTopic built-in charts
    st.markdown('<div class="sec-head">Topic Visualizations</div>', unsafe_allow_html=True)
    try:
        fig_bars = topic_model.visualize_barchart(top_n_topics=10)
        st.plotly_chart(fig_bars, width="stretch")
    except Exception as e:
        st.warning(f"Barchart unavailable: {e}")

    try:
        fig_topics = topic_model.visualize_topics()
        st.plotly_chart(fig_topics, width="stretch")
    except Exception as e:
        st.warning(f"Distance map unavailable: {e}")


# ──────────────────────────── Page: Embeddings ────────────────────────────

def render_umap():
    st.markdown(
        '<div class="niq-header">'
        '<div class="niq-logo" style="font-size:1.5rem">Embedding Map</div>'
        '<div class="niq-sub">UMAP 2D projection of sentence embeddings — '
        'each point is an article, colored by discovered topic</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    projection = load_umap_projection()
    if projection is None:
        st.info("UMAP projection data not found.")
        return

    topics_data = load_topics_data()
    cc_data = load_cc_news_data()

    labels = [f"Topic {t}" for t in topics_data] if topics_data is not None else ["—"] * len(projection)
    titles = cc_data["title"].tolist() if cc_data is not None else None

    from src.visualizations import plot_umap_scatter
    fig = plot_umap_scatter(projection, labels, titles, max_points=15000)
    st.plotly_chart(fig, width="stretch")

    # Parameter reference
    with st.expander("UMAP configuration used"):
        st.markdown("""
| Parameter | Value | Rationale |
|---|---|---|
| `n_neighbors` | 30 | Higher values preserve more global structure — better for overview visualization |
| `min_dist` | 0.1 | Moderate spacing prevents cluster overlap while keeping structure |
| `metric` | cosine | Standard for normalized text embeddings from SentenceTransformers |
| `n_components` | 2 | For 2D scatter rendering (BERTopic uses 5D internally for HDBSCAN) |

UMAP was chosen over t-SNE because it preserves global topology, runs faster on large datasets,
and supports transform on new data points without re-fitting.
        """)


# ──────────────────────────── Page: Summarizer ────────────────────────────

def render_summarizer():
    st.markdown(
        '<div class="niq-header">'
        '<div class="niq-logo" style="font-size:1.5rem">Summarizer</div>'
        '<div class="niq-sub">Abstractive summarization using BART-large-CNN '
        'via HuggingFace Inference API — cached to disk</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    cc_data = load_cc_news_data()
    use_example = False

    if cc_data is not None and len(cc_data) > 0:
        use_example = st.checkbox("Load a random article from the dataset")

    if use_example and cc_data is not None:
        sample = cc_data.sample(5, random_state=42)
        selected_title = st.selectbox("Select article:", sample["title"].tolist())
        text_input = sample[sample["title"] == selected_title]["text_clean"].iloc[0]
        st.text_area("Article text:", text_input, height=180, disabled=True)
    else:
        text_input = st.text_area(
            "Paste an article:",
            height=180,
            placeholder="Enter at least 100 characters of article text...",
        )

    if st.button("Generate Summary", type="primary"):
        if not text_input or len(text_input.strip()) < 100:
            st.warning("Need at least 100 characters for meaningful summarization.")
        else:
            from src.summarizer import summarize

            with st.spinner("Calling BART-large-CNN..."):
                result = summarize(text_input)

            if result.get("error"):
                st.error(f"Error: {result['error']}")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="sec-head">Original</div>', unsafe_allow_html=True)
                    st.write(text_input[:2000])
                with col2:
                    st.markdown('<div class="sec-head">Summary</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="result-banner">{result["summary"]}</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    '<div class="stat-row">'
                    + stat_card(str(result["source_length"]), "Source Words", "cyan")
                    + stat_card(str(result["summary_length"]), "Summary Words", "violet")
                    + stat_card(f"{result['compression_ratio']:.0%}", "Compression", "emerald")
                    + '</div>',
                    unsafe_allow_html=True,
                )
                if result.get("cached"):
                    st.caption("Served from disk cache")


# ──────────────────────────── Router ────────────────────────────

if page == "Pipeline Overview":
    render_overview()
elif page == "Classification":
    render_classifier()
elif page == "Topic Discovery":
    render_topics()
elif page == "Embedding Map":
    render_umap()
elif page == "Summarizer":
    render_summarizer()
