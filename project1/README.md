# 🔍 Project 1: Multilingual Fact-Checker (MuRIL + NLI Fusion)

A multilingual claim verification system that classifies health and general claims in Hindi, English, and code-mixed text. Uses a **fine-tuned MuRIL classifier** fused with **retrieval-augmented NLI voting** for robust fact-checking across languages.

---

## 📐 Architecture Overview

```
Input Claim (Hindi / English / Code-mixed)
        │
        ▼
 ┌──────────────────┐
 │  1. Exact Match  │ ← character-level lookup in dataset
 └──────────────────┘
        │ (no match)
        ▼
 ┌──────────────────────────────────────┐
 │  2. Top-K Retrieval                  │
 │     paraphrase-multilingual-mpnet    │
 │     + FAISS cosine similarity        │
 └──────────────────────────────────────┘
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
 ┌─────────────────┐               ┌───────────────────────┐
 │  3. MuRIL       │               │  4. NLI Vote          │
 │  Classifier     │               │  xlm-roberta-large    │
 │  [claim][SEP]   │               │  over top-K matches   │
 │  [evidence]     │               │  ENTAIL/CONTRA/NEUTRAL│
 └─────────────────┘               └───────────────────────┘
        │                                      │
        └──────────────┬───────────────────────┘
                       ▼
            ┌─────────────────────┐
            │  5. Confidence-     │
            │  Weighted Fusion    │
            │  NLI × sim_score    │
            │  MuRIL × clf_conf   │
            └─────────────────────┘
                       │
                       ▼
             Verdict: true / false / misleading / other
```

---

## 🖥️ Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GPU ≥8 GB VRAM (tested on A100 40 GB) |
| CPU | Any modern x86-64 |
| RAM | ≥16 GB recommended |
| Storage | ≥20 GB (model + dataset + FAISS index) |
| OS | Linux (Ubuntu 20.04+ / Fedora 38+) |

> MuRIL inference runs comfortably in **~3 GB VRAM**. The NLI verifier (`xlm-roberta-large-xnli`) needs ~2.5 GB VRAM additionally if running both simultaneously.

---

## ⚙️ Environment Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv_p1
source venv_p1/bin/activate

# 2. Install dependencies
pip install -r project1/requirements.txt
```

---

## 📦 Dependency Versions

```
torch==2.2.2
transformers==4.44.0
datasets==2.20.0
sentence-transformers==3.0.1
faiss-gpu==1.7.4
scikit-learn==1.5.1
pandas==2.2.2
numpy==1.26.4
evaluate==0.4.2
```

---

## 📁 Project Structure

```
project1/
├── README.md
├── requirements.txt
│
├── configs/
│   ├── train_muril.yaml       # Training hyperparameters
│   └── inference.yaml         # Runtime thresholds and model paths
│
├── scripts/
│   ├── preprocess.sh          # Merge + clean raw CSVs
│   ├── build_index.sh         # Build FAISS retriever
│   ├── train.sh               # Train MuRIL classifier
│   ├── eval.sh                # Evaluate on held-out test set
│   └── infer.sh               # Interactive / single-claim inference
│
└── src/
    ├── preprocess.py          # Data merging and label normalisation
    ├── build_retriever.py     # FAISS index builder
    ├── train_classifier.py    # MuRIL fine-tuning with weighted loss
    ├── nli_verifier.py        # NLI pipeline wrapper (xlm-roberta)
    ├── inference.py           # Full prediction pipeline
    └── utils.py               # Shared utilities
```

---

## 🗂️ Data Format

`data/final_dataset.csv` — output of `preprocess.py`:

| Column | Type | Description |
|--------|------|-------------|
| `claim` | str | The health or general claim |
| `evidence` | str | Supporting/contradicting evidence text |
| `label` | int | 0=true, 1=false, 2=misleading, 3=other |
| `source` | str | (optional) source URL or name |

Raw input files expected in `data/`:
- `facts_merged_final.csv`
- `factcheck_dataset_modified.csv`

---

## 🏋️ Training Commands

### Step 1 — Preprocess data

```bash
bash project1/scripts/preprocess.sh
# or:
python project1/src/preprocess.py
```

Merges and deduplicates both raw CSVs, maps string labels → integers, saves to `data/final_dataset.csv`.

---

### Step 2 — Build FAISS retriever index

```bash
bash project1/scripts/build_index.sh
# or:
python project1/src/build_retriever.py
```

Encodes all claims with `paraphrase-multilingual-MiniLM-L12-v2` and saves:
- `models/fact_index.faiss`
- `models/claims.pkl`

---

### Step 3 — Train MuRIL classifier

```bash
bash project1/scripts/train.sh
```

Or with explicit config:

```bash
python project1/src/train_classifier.py \
    --config project1/configs/train_muril.yaml
```

Key design choices (see `configs/train_muril.yaml` for all values):

| Setting | Value | Reason |
|---------|-------|--------|
| Input format | `[claim][SEP][evidence]` | Evidence context at train time |
| Loss | Weighted cross-entropy | Handles label imbalance |
| Label smoothing | 0.1 | Prevents overconfident predictions |
| LR scheduler | Cosine with warmup | Better generalisation |
| Best checkpoint | Max `f1_macro` on val | Robust to class imbalance |

Saved to: `models/muril/`

---

## 📊 Evaluation Commands

```bash
bash project1/scripts/eval.sh
```

Or directly:

```bash
# Evaluate MuRIL on validation split (auto-runs after training)
python project1/src/train_classifier.py   # prints final val metrics

# Standalone evaluation on a custom test JSON:
python - <<'EOF'
from sklearn.metrics import classification_report
# load your test data and call predict_claim() from inference.py
EOF
```

Expected training output (per epoch):

```
              precision    recall  f1-score   support
        true       0.81      0.79      0.80       312
       false       0.78      0.83      0.80       289
  misleading       0.65      0.61      0.63        98
       other       0.55      0.52      0.53        44

    accuracy                           0.77       743
   macro avg       0.70      0.69      0.69       743
```

---

## 🔍 Inference Commands

### Interactive REPL

```bash
bash project1/scripts/infer.sh
```

```
Enter claim: Haldi mein anti-cancer properties hoti hain

── Result ──────────────────────────────
  verdict              : true
  muril_pred           : true
  muril_conf           : 0.8341
  nli_pred             : true
  similarity           : 0.8741
  matched_claim        : Curcumin in turmeric has demonstrated anti-tumor properties...
  method               : nli+muril_agree
────────────────────────────────────────
```

### Single-claim JSON output

```bash
bash project1/scripts/infer.sh --claim "Vitamin C boosts immunity"
```

---

## 🏷️ Label Schema

| Int | String | Meaning |
|-----|--------|---------|
| 0 | `true` | Claim supported by evidence |
| 1 | `false` | Claim contradicted by evidence |
| 2 | `misleading` | Partially true / out of context |
| 3 | `other` | Insufficient evidence |

---

## 🔧 Fusion Logic

The final verdict is a **confidence-weighted vote** between two signals:

```
score[verdict] += similarity_score    # NLI signal
score[verdict] += muril_softmax_conf  # MuRIL signal
final = argmax(score)
```

- `method = nli+muril_agree` → both agree
- `method = nli_wins_fusion` → NLI outweighed MuRIL
- `method = muril_wins_fusion` → MuRIL outweighed NLI
- `method = classifier_only` → similarity < 0.45, NLI skipped
- `method = exact_match` → claim found verbatim in dataset

---

## ⚠️ Known Issues

- **FAISS on CPU**: Replace `faiss-gpu` with `faiss-cpu` if no GPU; retrieval will be ~10× slower.
- **NLI verifier cold start**: `nli_verifier.py` loads `xlm-roberta-large-xnli` at import time — expect ~15s startup on first run.
- **Empty evidence column**: If `evidence` is missing for some rows, the classifier still runs correctly with an empty sentence B.
