# Fact-Checking Health & News Claims in Hinglish
 
Two fact-checking pipelines built as part of an undergraduate deep learning project, run entirely on a single laptop GPU (8 GB VRAM, 16 GB RAM):
 
1. **Task 1 — Health claims**: Hinglish claim → translate to English (Llama-3.2, 4-bit) → retrieve evidence (FAISS + multilingual-E5) → NLI verdict (DeBERTa cross-encoder / fine-tuned RoBERTa & PubMedBERT).
2. **Task 2 — News claims**: look up the claim in a verified database first (dense retrieval); if no close match exists, fall back to a fine-tuned MuRIL classifier.
Full write-up, methodology, and results analysis are in [`main.pdf`](./main.pdf).
 
---
 
## Fine-Tuned Models
 
Three models were actually fine-tuned in this project (everything else — Llama-3.2, the E5/MiniLM/mpnet embedders, the DeBERTa and XLM-R NLI models — is used off-the-shelf/zero-shot, not fine-tuned):
 
| # | Base model | Fine-tuned for | Task | Trained on | Classes |
|---|---|---|---|---|---|
| 1 | `google/muril-base-cased` | News claim classification | Task 2 | `final_dataset.csv` (merged `facts_merged_final.csv` + `factcheck_dataset_modified.csv`) | true / false / misleading / other (4-way) |
| 2 | `roberta-base` | Health claim NLI | Task 1 | SciFact (`claims_train.csv`) | SUPPORT / CONTRADICT / NEI (3-way) |
| 3 | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` | Health claim NLI | Task 1 | SciFact (`claims_train.csv`) | SUPPORT / CONTRADICT / NEI (3-way) |
 
**MuRIL training config:** 5 epochs, batch size 16 (grad. accumulation 2), lr 2e-5, cosine LR schedule, fp16, class-weighted cross-entropy + label smoothing (0.1), input = `claim [SEP] evidence` pairs (max length 256), best checkpoint selected by macro-F1. Config: `project1-news-factcheck/configs/train_muril.yaml`, script: `train_classifier.py`.
 
**RoBERTa / PubMedBERT training config:** 3 epochs, batch size 8, lr 2e-5, fp16, input = `claim [SEP] evidence` pairs (max length 512). Script: `finetune.py`.
 
---
## Task 2: News Fact-Checking Pipeline
 
**Flow:** claim → embed with MiniLM → nearest-neighbour search in verified-claims index → if L2 distance < 0.5, return the database label directly; otherwise run the fine-tuned MuRIL classifier.
 
| Component | Choice |
|---|---|
| Classifier | `google/muril-base-cased`, fine-tuned, 4-way (`true / false / misleading / other`) |
| Retrieval embedder | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` + FAISS (`IndexFlatL2`) |
| NLI (interpretability only, not decision-making) | `joeddav/xlm-roberta-large-xnli` (zero-shot, not fine-tuned) |
| Training data | `facts_merged_final.csv` + `factcheck_dataset_modified.csv`, merged and deduplicated by `preprocess.py` → `final_dataset.csv` |
| Training setup | class-weighted cross-entropy + label smoothing (0.1), cosine LR schedule, 5 epochs, batch size 16, lr 2e-5, fp16, best checkpoint by macro-F1 |
| Input format | cross-encoder pair — `claim [SEP] evidence`, max length 256 |
 
**Scripts**
- `preprocess.py` — merge/dedupe raw CSVs, map string labels → ints, save `final_dataset.csv`
- `build_retriever.py` — embed all claims and build the FAISS index + claims pickle
- `train_classifier.py` — fine-tune MuRIL on claim–evidence pairs
- `inference.py` — full runtime pipeline: exact match → retrieval → classifier → NLI vote → confidence-weighted fusion
- `nli_verifier.py` — thin wrapper around the zero-shot XLM-R-XNLI pipeline
---
 
## Task 1: Health Fact-Checking Pipeline
 
**Flow:** Hinglish claim → Llama-3.2-3B-Instruct (4-bit, prompted) translates to English → embed with multilingual-E5-Base → retrieve top-3 evidence docs from a FAISS index built over PubHealth → run each (claim, evidence) pair through a DeBERTa NLI cross-encoder → aggregate (prefer any definitive TRUE/FALSE over UNCLEAR).
 
| Component | Choice |
|---|---|
| Translation | `meta-llama/Llama-3.2-3B-Instruct`, 4-bit NF4 quantized (QLoRA/BitsAndBytes), prompted — not fine-tuned |
| Retrieval embedder | `intfloat/multilingual-e5-base` (`query:` / `passage:` prefixes) + FAISS (`IndexFlatIP`) |
| Retrieval corpus | PubHealth (`health_fact` dataset), filtered to entries with an explanation > 150 chars, deduped by claim |
| Live-pipeline NLI verdict | `cross-encoder/nli-deberta-v3-small` (zero-shot, not fine-tuned) |
| Fine-tuned NLI models | `roberta-base` → `roberta_medical`, `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` → `pubmed_medical`, both fine-tuned on **SciFact** (claim, evidence → SUPPORT/CONTRADICT/NEI) |
| Training setup | batch size 8, lr 2e-5, 3 epochs, fp16, max length 512 |
| Baseline (for comparison) | raw Hinglish → TF-IDF search → zero-shot BART-large-MNLI |
 
**Scripts**
- `medical_indexer.py` — build the PubHealth FAISS index + mapping.json
- `finetune.py` — fine-tune RoBERTa and PubMedBERT on SciFact
- `integrated_checker.py` — full pipeline class (translate → retrieve → NLI verdict), also runnable as a CLI
- `GUI.py` — Gradio front-end over `integrated_checker.py`
- `research_evaluator.py` / `final_pipeline_evaluator.py` — benchmark baseline vs. fine-tuned models on a labeled test set
---
 
## Setup
 
```bash
pip install torch transformers sentence-transformers faiss-cpu \
            datasets evaluate scikit-learn pandas numpy \
            gradio bitsandbytes accelerate
```
 
GPU with ≥8 GB VRAM recommended for the health pipeline (4-bit Llama-3.2 + sequential model loading). The news pipeline is lighter and can run on CPU, though training MuRIL is much faster on GPU.
 

 
## Known Limitations / Issues
 
- **Tiny data test set** 

 
---
 
## Models & Datasets Used
 
- **MuRIL** (`google/muril-base-cased`) — Indic-language pretrained encoder, fine-tuned for news classification
- **SciFact** — scientific claim/evidence dataset, used to fine-tune RoBERTa & PubMedBERT
- **PubHealth** (`health_fact`) — public health claims + journalist explanations, used as the retrieval corpus
- **Llama-3.2-3B-Instruct** — 4-bit quantized translator (Hinglish → English)
- **multilingual-E5-Base**, **paraphrase-multilingual-MiniLM-L12-v2** — dense retrieval embedders
- **cross-encoder/nli-deberta-v3-small**, **xlm-roberta-large-xnli** — zero-shot NLI verifiers
- **FAISS** — vector similarity search
- **QLoRA / BitsAndBytes** — 4-bit quantization for running Llama-3.2 on an 8 GB GPU
---
 
