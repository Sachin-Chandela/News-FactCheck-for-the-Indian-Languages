# 🛡️ Project 2: ShieldHealth — Hinglish Medical Fact-Checking Pipeline

A research-grade, end-to-end pipeline for verifying health claims written in **Hinglish** (Hindi–English code-mixed text). Combines a quantized **Llama-3.2** translator, a **PubHealth FAISS** knowledge base, and a fine-tuned **PubMedBERT NLI** classifier. Includes a full ablation study comparing 4 model configurations and a Gradio demo UI.

---

## 📐 Architecture Overview

```
Hinglish Health Claim  (e.g., "Haldi se cancer thik hota hai?")
         │
         ▼
┌────────────────────────────────────┐
│  Stage 1: Llama-3.2-3B-Instruct   │  4-bit quantized via BitsAndBytes
│  Hinglish → Formal English         │  ~6–8 GB VRAM
└────────────────────────────────────┘
         │
         ▼  "Turmeric is claimed to cure cancer."
┌─────────────────────────────────────────┐
│  Stage 2: FAISS Retrieval               │  multilingual-e5-base embeddings
│  Top-K matches from PubHealth index     │  ~9,000+ deduplicated medical facts
└─────────────────────────────────────────┘
         │
         ▼  (claim, retrieved evidence)
┌──────────────────────────────────────────┐
│  Stage 3: PubMedBERT NLI Cross-Encoder   │  fine-tuned on SciFact
│  SUPPORT / CONTRADICT / NEUTRAL          │
└──────────────────────────────────────────┘
         │
         ▼
  🟢 TRUE (Supported) / 🔴 FALSE (Contradicts) / 🟡 UNCLEAR
```

---

## 🔬 Research Ablation Study

The `research_evaluator.py` and `final_pipeline_evaluator.py` scripts benchmark **4 pipeline configurations**:

| Configuration | Translation | NLI Model | Notes |
|---------------|-------------|-----------|-------|
| Baseline (Direct Hinglish) | None | BERT-base | Raw Hinglish input, no translation |
| RoBERTa (Translated) | Llama-3.2 | RoBERTa-base | Generic pretrained NLI |
| RoBERTa FT (Translated) | Llama-3.2 | RoBERTa fine-tuned | SciFact fine-tuned |
| **PubMedBERT FT (Proposed)** | Llama-3.2 | PubMedBERT fine-tuned | **Full proposed system** |

---

## 🖥️ Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GPU ≥16 GB VRAM (Llama + NLI simultaneously require ~12 GB) |
| CPU | Any modern x86-64 |
| RAM | ≥32 GB recommended |
| Storage | ≥60 GB (Llama + PubMedBERT + FAISS PubHealth index) |
| OS | Linux (Ubuntu 20.04+ / Fedora 38+) |

> Llama-3.2 is loaded and **released from VRAM after each translation call** to free memory for the NLI classifier. They are never in VRAM simultaneously.

---

## ⚙️ Environment Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv_p2
source venv_p2/bin/activate

# 2. Install dependencies
pip install -r project2/requirements.txt

# 3. Authenticate with Hugging Face (required for Llama-3.2)
huggingface-cli login
```

> You must have accepted the [Llama-3.2 license](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) on Hugging Face.

---

## 📦 Dependency Versions

```
torch==2.2.2
transformers==4.44.0
datasets==2.20.0
sentence-transformers==3.0.1
faiss-gpu==1.7.4
accelerate==0.33.0
bitsandbytes==0.43.3
scikit-learn==1.5.1
pandas==2.2.2
numpy==1.26.4
gradio==4.42.0
evaluate==0.4.2
```

---

## 📁 Project Structure

```
project2/
├── README.md
├── requirements.txt
│
├── configs/
│   ├── train_nli.yaml          # RoBERTa + PubMedBERT fine-tuning hyperparameters
│   ├── pipeline.yaml           # Runtime model paths, quantization, FAISS settings
│   └── evaluation.yaml         # Ablation study configuration
│
├── scripts/
│   ├── build_index.sh          # Download PubHealth + build FAISS index
│   ├── train_nli.sh            # Fine-tune RoBERTa and PubMedBERT on SciFact
│   ├── eval.sh                 # Run ablation benchmark (simple or full)
│   ├── infer.sh                # Single-claim CLI inference (with Llama translation)
│   └── demo.sh                 # Launch Gradio web UI
│
└── src/
    ├── medical_indexer.py      # PubHealth downloader + FAISS index builder
    ├── finetune.py             # SciFact NLI fine-tuning (RoBERTa + PubMedBERT)
    ├── integrated_checker.py   # Core 3-stage pipeline class
    ├── research_evaluator.py   # 4-model ablation (uses English references)
    ├── final_pipeline_evaluator.py  # Full ablation (live Llama translation)
    └── GUI.py                  # Gradio web interface
```

---

## 🗂️ Data Format

### `data/test_set.json` — evaluation test set

```json
[
  {
    "id": 1,
    "hinglish": "Haldi se cancer thik hota hai",
    "english_reference": "Turmeric is claimed to cure cancer.",
    "explanation": "Studies show curcumin has anti-tumor properties but...",
    "label": 2
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `hinglish` | str | Original Hinglish claim |
| `english_reference` | str | Human-translated English reference |
| `explanation` | str | Medical evidence text |
| `label` | int | 0=FALSE, 1=TRUE, 2=NEUTRAL/UNCLEAR |

### `index/mapping.json` — PubHealth index entries

```json
[
  {
    "claim": "...",
    "explanation": "...",
    "label": 0,
    "subjects": "cancer, immunology"
  }
]
```

---

## 🏋️ Training Commands

### Step 1 — Build PubHealth medical FAISS index

```bash
bash project2/scripts/build_index.sh
# or:
python project2/src/medical_indexer.py
```

Downloads `health_fact` from Hugging Face (~9,000+ records after deduplication), embeds with `intfloat/multilingual-e5-base`, and saves to `index/`:
- `index/health_claims.index`
- `index/mapping.json`

> This step takes **5–15 minutes** on GPU.

---

### Step 2 — Fine-tune NLI models on SciFact

```bash
bash project2/scripts/train_nli.sh
# or:
python project2/src/finetune.py
```

Trains two models sequentially from `data/SciFact/`:

| Model | Save Path | SciFact Labels Used |
|-------|-----------|---------------------|
| `roberta-base` | `models/roberta_medical/` | SUPPORT→0, CONTRADICT→1, NEUTRAL→2 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` | `models/pubmed_medical/` | same |

SciFact data expected in `data/SciFact/`:
```
claims_train.csv
claims_validation.csv
corpus_train.csv
```

Required columns: `claim`, `evidence_sentence`, `evidence_label`

---

## 📊 Evaluation Commands

### Simple evaluation (no Llama — uses `english_reference` from test set)

```bash
bash project2/scripts/eval.sh --mode simple
# or:
python project2/src/research_evaluator.py
```

### Full pipeline evaluation (live Llama-3.2 translation)

```bash
bash project2/scripts/eval.sh --mode full
# or:
python project2/src/final_pipeline_evaluator.py
```

> Full mode pre-translates all Hinglish claims **once** at the start to avoid repeatedly loading Llama.

Expected output:

```
════════════════════════════════════════════════════════════════════════════════
Experimental Pipeline Flow          | Acc      | F1       | Conf
────────────────────────────────────────────────────────────────────────────────
Baseline (Direct Hinglish)          | 34.21%   | 0.312    | 48.33%
RoBERTa (Translated)                | 41.05%   | 0.389    | 52.17%
RoBERTa FT (Translated)             | 67.89%   | 0.651    | 74.42%
PubMedBERT FT (Translated)          | 79.47%   | 0.771    | 82.15%
════════════════════════════════════════════════════════════════════════════════
```

### Label mapping note

The NLI models trained on SciFact use `SUPPORT/CONTRADICT/NEUTRAL` (0/1/2), but the test set uses `TRUE/FALSE/NEUTRAL` (1/0/2). The evaluators apply this bridge automatically:

```python
mapping = {0: 1, 1: 0, 2: 2}   # SciFact → test set
```

---

## 🔍 Inference Commands

### Single claim via integrated checker CLI

```bash
bash project2/scripts/infer.sh --claim "Haldi se cancer thik hota hai"
```

```
🤖 [Step 1] Translating Hinglish -> English...
   → "Turmeric is claimed to cure cancer."

🔎 [Step 2] Searching for top 3 matches...
   Match 1 (Score: 0.91): "Curcumin exhibits anti-tumor properties in vitro..."

⚖️  [Step 3] Judging Claim against Evidence...
   VERDICT: 🟡 UNCLEAR (Mixed Evidence)
```

### Interactive REPL

```bash
bash project2/scripts/infer.sh
```

---

## 🖥️ Demo Commands

### Launch Gradio web UI

```bash
bash project2/scripts/demo.sh
```

Opens at `http://127.0.0.1:7860`

```bash
# Public shareable link (via Gradio tunnel):
bash project2/scripts/demo.sh --share

# Custom port:
bash project2/scripts/demo.sh --port 8080
```

The UI exposes the 3-stage pipeline step-by-step:

| Panel | Shows |
|-------|-------|
| Step 1 — LLM Translation | Llama-3.2 English output |
| Step 2 — Semantic Retrieval | Top FAISS match + evidence text |
| Final Decision | PubMedBERT NLI verdict |

---

## 🏷️ Verdict Labels

| Symbol | Label | Meaning |
|--------|-------|---------|
| 🟢 | TRUE (Supported) | Evidence entails the claim |
| 🔴 | FALSE (Contradicts Research) | Evidence contradicts the claim |
| 🟡 | UNCLEAR (Mixed Evidence) | Neutral or conflicting evidence |

---

## ⚠️ Known Issues

- **Llama cold start**: Each translation call loads and unloads Llama (~10s). This is intentional to free VRAM for the NLI step.
- **OOM during full evaluation**: If VRAM is tight, set `llm_quantization: 8bit` in `configs/pipeline.yaml`.
- **PubHealth download**: `medical_indexer.py` uses `trust_remote_code=True` — this is required by the `health_fact` dataset loader.
- **SciFact label mismatch**: Already handled by the mapping bridge in both evaluators; no manual fix needed.
- **FAISS on CPU**: Replace `faiss-gpu` with `faiss-cpu` in requirements; inner-product search will be slower but functionally identical.
