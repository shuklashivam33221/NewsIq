"""
NewsIQ — Article Summarization
Uses HuggingFace Inference API (BART-large-CNN) with local caching.
"""
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests

from src.config import (
    HF_TOKEN, SUMMARIZER_API_URL,
    SUMMARY_MAX_LENGTH, SUMMARY_MIN_LENGTH,
    SUMMARY_MAX_INPUT_CHARS, SUMMARY_CACHE_DIR,
)

logger = logging.getLogger(__name__)


def _cache_key(text: str) -> str:
    return hashlib.md5(text[:500].encode()).hexdigest()


def summarize(
    text: str,
    max_length: int = SUMMARY_MAX_LENGTH,
    min_length: int = SUMMARY_MIN_LENGTH,
    max_retries: int = 3,
) -> dict:
    """
    Summarize an article using HuggingFace Inference API.
    Results are cached to disk to avoid redundant API calls.

    Args:
        text: article text to summarize
        max_length: max summary length in tokens
        min_length: min summary length in tokens
        max_retries: retries on 503 (model loading)

    Returns:
        dict with keys: summary, source_length, summary_length, compression_ratio, cached
    """
    if not text or len(text.strip()) < 100:
        return {
            "summary": text.strip(),
            "source_length": len(text.split()),
            "summary_length": len(text.split()),
            "compression_ratio": 1.0,
            "cached": False,
            "error": "Text too short for summarization",
        }

    # Check cache
    key = _cache_key(text)
    cache_path = SUMMARY_CACHE_DIR / f"{key}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        cached["cached"] = True
        return cached

    # Truncate to BART's max input
    truncated = text[:SUMMARY_MAX_INPUT_CHARS]
    source_words = len(truncated.split())

    # Call API
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    payload = {
        "inputs": truncated,
        "parameters": {"max_length": max_length, "min_length": min_length},
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                SUMMARIZER_API_URL, headers=headers, json=payload, timeout=60,
            )

            if response.status_code == 503:
                wait = min(20 * (attempt + 1), 60)
                logger.warning(f"Model loading, retrying in {wait}s...")
                time.sleep(wait)
                continue

            if response.status_code != 200:
                return {
                    "summary": "", "error": f"API error {response.status_code}",
                    "source_length": source_words, "summary_length": 0,
                    "compression_ratio": 0, "cached": False,
                }

            result = response.json()
            summary_text = result[0]["summary_text"]
            summary_words = len(summary_text.split())

            output = {
                "summary": summary_text,
                "source_length": source_words,
                "summary_length": summary_words,
                "compression_ratio": round(summary_words / max(source_words, 1), 3),
                "cached": False,
            }

            # Cache result
            cache_path.write_text(json.dumps(output, indent=2))
            return output

        except requests.exceptions.Timeout:
            logger.warning(f"API timeout, attempt {attempt + 1}/{max_retries}")
        except requests.exceptions.ConnectionError as e:
            error_msg = "Network Error: Your ISP or router is blocking access to HuggingFace's Inference API (api-inference.huggingface.co). Try connecting via a VPN or mobile hotspot."
            logger.error(error_msg)
            return {
                "summary": "", "error": error_msg,
                "source_length": source_words, "summary_length": 0,
                "compression_ratio": 0, "cached": False,
            }
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return {
                "summary": "", "error": str(e),
                "source_length": source_words, "summary_length": 0,
                "compression_ratio": 0, "cached": False,
            }

    return {
        "summary": "", "error": "Max retries exceeded",
        "source_length": source_words, "summary_length": 0,
        "compression_ratio": 0, "cached": False,
    }


def precompute_summaries(documents: list[str], n: int = 100, seed: int = 42):
    """Pre-compute summaries for N random articles (for fast Streamlit demo)."""
    import random
    random.seed(seed)
    indices = random.sample(range(len(documents)), min(n, len(documents)))
    results = {}
    for i, idx in enumerate(indices):
        logger.info(f"Summarizing [{i+1}/{len(indices)}]...")
        results[idx] = summarize(documents[idx])
    return results
