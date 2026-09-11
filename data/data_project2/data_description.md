# Data Description: Health-Fact-Check-India

This document outlines the dataset architecture, preprocessing pipeline, and subset configurations used for training the deep-learning-based fact-checking system for health misinformation.

## 1. Dataset Overview
The primary dataset is a curated collection of verified and unverified health claims circulating on Indian social media platforms, specifically targeting the nuances of Hinglish (Hindi-English code-switching).

* **Source:** Aggregated from Indian fact-checking portals (e.g., AltNews Health, Boom Live) and anonymized public health-related WhatsApp groups.
* **Total Samples:** ~12,500 claims.
* **Categories:** Home remedies, COVID-19, Vaccines, Chronic diseases, and Nutrition.
* **Languages:** English, Hindi, and Hinglish.
* **Labels:** `Supported`, `Refuted`, `Not Enough Evidence`.

## 2. Train / Val / Test Split
To ensure the model generalizes across evolving misinformation trends, a stratified split was used to maintain the label distribution across all sets.

| Split | Percentage | Sample Count | Purpose |
| :--- | :--- | :--- | :--- |
| **Train** | 70% | 8,750 | Model fine-tuning and weight updates. |
| **Validation** | 15% | 1,875 | Hyperparameter optimization and early stopping. |
| **Test** | 15% | 1,875 | Final benchmark against unseen real-world claims. |

## 3. Preprocessing
To handle the high level of noise in social media data, the following pipeline is executed:

1.  **Text Cleaning:** Removal of platform-specific metadata (e.g., "Forwarded many times") and excessive punctuation.
2.  **Emoji Normalization:** Emojis are converted into text tokens (e.g., 💊 becomes "pill") to retain context.
3.  **Transliteration:** Romanized Hindi (Hinglish) is handled through a custom tokenizer to map phonetically similar words to a unified embedding space.
4.  **Claim Normalization:** Using an LLM-based pass to convert conversational/slang text into a formal declarative statement for better retrieval performance.

## 4. Reduced Setup (Development Subset)
For local development and debugging—specifically designed for environments with VRAM constraints (e.g., 8GB-12GB GPUs)—a **Lite Setup** is available:

* **Subset Size:** 1,200 samples.
* **Selection Logic:** A balanced selection (400 per class) focusing on short-form claims (<50 words) to minimize sequence padding.
* **Retrieval Cache:** Pre-computed FAISS embeddings are provided for this subset to bypass the need for a full-scale vector database initialization during quick iterations.

## 5. Class Imbalance Handling
The raw data is naturally skewed toward `Refuted` claims. To mitigate this:
* **Training:** Weighted Cross-Entropy Loss is used to penalize misclassifications of the minority `Supported` and `NEI` classes.
* **Evaluation:** Focus is placed on Macro-F1 scores rather than raw accuracy.
