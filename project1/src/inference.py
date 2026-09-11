import torch
import faiss
import numpy as np
import pandas as pd
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nli_verifier import verify

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data
df = pd.read_csv("../data/final_dataset.csv")
df["claim"] = df["claim"].astype(str)

# Load classifier
tokenizer = AutoTokenizer.from_pretrained("../models/muril")
clf_model = AutoModelForSequenceClassification.from_pretrained("../models/muril")
clf_model.to(device)
clf_model.eval()

# Embedding model
embed_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    device=device
)

# Load FAISS
index = faiss.read_index("../models/fact_index.faiss")

# Load labels from saved model config (set during training)
# Falls back to hardcoded dict if model was saved without id2label
labels = clf_model.config.id2label if clf_model.config.id2label else {
    0: "true",
    1: "false",
    2: "misleading",
    3: "other"
}

# ──────────────────────────────────────────────────────────────
# Thresholds
# ──────────────────────────────────────────────────────────────
EXACT_MATCH_THRESHOLD  = 1.00   # identical string
HIGH_SIM_THRESHOLD     = 0.88   # was used to skip NLI — REMOVED
LOW_SIM_THRESHOLD      = 0.45   # below this → pure classifier, no retrieval trust
TOP_K                  = 5      # retrieve top-K neighbours for NLI voting
NLI_CONF_MIN           = 0.60   # min NLI confidence to trust its label


def classify_claim(claim: str, evidence: str = "") -> tuple:
    """
    Fine-tuned MuRIL classifier — returns (label_str, confidence).

    Takes [claim] [SEP] [evidence] pair exactly as trained.
    If no evidence is available, passes empty string as sentence B —
    token_type_ids are still correctly structured, just no evidence context.
    """
    inputs = tokenizer(
        claim,                  # sentence A
        evidence,               # sentence B — matches training tokenization
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,         # matches MAX_LENGTH in training
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = clf_model(**inputs).logits
        probs  = F.softmax(logits, dim=1)
        pred   = torch.argmax(probs, dim=1).item()
        conf   = probs[0][pred].item()

    return labels[pred], conf


def get_top_k_matches(claim_text, k=TOP_K):
    """Return top-K similar claims from the dataset with their scores."""
    q_emb = embed_model.encode([claim_text], convert_to_tensor=True)
    corpus_emb = embed_model.encode(
        df["claim"].tolist(),
        convert_to_tensor=True,
        batch_size=128
    )
    scores   = util.cos_sim(q_emb, corpus_emb)[0]
    top_idxs = torch.topk(scores, k=min(k, len(df))).indices.tolist()
    return [(df.iloc[i], float(scores[i])) for i in top_idxs]


def nli_vote(matches, claim_text):
    """
    Run NLI between the input claim and each retrieved match.
    Each match votes: ENTAILMENT → use dataset label, CONTRADICTION → flip,
    NEUTRAL → abstain.

    Returns the winning verdict with a confidence-weighted vote count,
    or None if no match clears NLI_CONF_MIN.
    """
    vote_scores = {}   # verdict → accumulated NLI score
    best_evidence = {}  # verdict → best supporting row

    for row, sim_score in matches:
        # Use evidence text if available, otherwise use the matched claim
        premise = str(row.get("evidence", "") or row["claim"]).strip()
        if not premise:
            premise = str(row["claim"])

        nli_result = verify(premise, claim_text)
        nli_label  = nli_result["label"].lower()
        nli_conf   = float(nli_result.get("score", nli_result.get("confidence", 0.5)))

        if nli_conf < NLI_CONF_MIN:
            continue

        dataset_verdict = labels[int(row["label"])]

        if "entail" in nli_label:
            # The evidence supports the claim as-is → trust dataset label
            verdict = dataset_verdict
        elif "contradict" in nli_label:
            # The claim contradicts the known fact → flip true↔false
            if int(row["label"]) == 0:
                verdict = "false"
            elif int(row["label"]) == 1:
                verdict = "true"
            else:
                # misleading/other contradicted → still call it false
                verdict = "false"
        else:
            # neutral → abstain
            continue

        weight = sim_score * nli_conf          # similarity × NLI confidence
        vote_scores[verdict] = vote_scores.get(verdict, 0.0) + weight

        if verdict not in best_evidence or weight > best_evidence[verdict]["weight"]:
            best_evidence[verdict] = {
                "weight": weight,
                "row": row,
                "sim": sim_score,
                "nli": nli_result,
            }

    if not vote_scores:
        return None

    winner = max(vote_scores, key=vote_scores.get)
    meta   = best_evidence[winner]
    return {
        "verdict":       winner,
        "matched_claim": meta["row"]["claim"],
        "similarity":    round(meta["sim"], 4),
        "source":        meta["row"].get("source", ""),
        "evidence":      meta["row"].get("evidence", ""),
        "nli":           meta["nli"],
        "vote_scores":   {k: round(v, 4) for k, v in vote_scores.items()},
        "method":        "nli_vote",
    }


def combine_nli_and_classifier(nli_out, clf_pred, clf_conf, top_sim):
    """
    Confidence-weighted fusion of NLI vote and fine-tuned MuRIL.

    Neither signal is thrown away. Instead each casts a weighted vote:
      - NLI weight  = top_sim        (how close the retrieved fact was)
      - MuRIL weight = clf_conf      (how confident the fine-tuned model is)

    The verdict with the higher total weight wins.
    This means MuRIL genuinely influences every prediction, not just fallbacks.
    """
    nli_verdict = nli_out["verdict"]

    # Build a score table across all possible verdicts
    scores = {}

    # NLI vote — weighted by retrieval similarity
    nli_weight = top_sim
    scores[nli_verdict] = scores.get(nli_verdict, 0.0) + nli_weight

    # MuRIL vote — weighted by its own softmax confidence
    clf_weight = clf_conf
    scores[clf_pred] = scores.get(clf_pred, 0.0) + clf_weight

    # Pick the verdict with the highest combined score
    final_verdict = max(scores, key=scores.get)

    if nli_verdict == clf_pred:
        method = "nli+muril_agree"
    elif final_verdict == nli_verdict:
        method = "nli_wins_fusion"
    else:
        method = "muril_wins_fusion"

    return {
        "verdict":       final_verdict,
        "muril_pred":    clf_pred,
        "muril_conf":    round(clf_conf, 4),
        "nli_pred":      nli_verdict,
        "nli_weight":    round(nli_weight, 4),
        "fusion_scores": {k: round(v, 4) for k, v in scores.items()},
        "matched_claim": nli_out["matched_claim"],
        "similarity":    nli_out["similarity"],
        "source":        nli_out.get("source", ""),
        "evidence":      nli_out.get("evidence", ""),
        "method":        method,
    }


def predict_claim(claim_text: str) -> dict:
    claim_text = claim_text.strip()

    # ──────────────────────────────────────────────────────────
    # 1. EXACT MATCH (character-level identical)
    # ──────────────────────────────────────────────────────────
    exact = df[df["claim"].str.lower() == claim_text.lower()]
    if len(exact) > 0:
        row = exact.iloc[0]
        return {
            "verdict":       labels[int(row["label"])],
            "matched_claim": row["claim"],
            "source":        row.get("source", ""),
            "evidence":      row.get("evidence", ""),
            "method":        "exact_match",
        }

    # ──────────────────────────────────────────────────────────
    # 2. RETRIEVE TOP-K NEIGHBOURS
    # ──────────────────────────────────────────────────────────
    matches  = get_top_k_matches(claim_text, k=TOP_K)
    top_sim  = matches[0][1] if matches else 0.0

    # ──────────────────────────────────────────────────────────
    # 3. RUN CLASSIFIER (always, so we can use it for fusion)
    #    Pass the top retrieved evidence as sentence B —
    #    this mirrors the [claim][SEP][evidence] format used in training.
    # ──────────────────────────────────────────────────────────
    top_evidence = str(matches[0][0].get("evidence", "") or "") if matches else ""
    clf_pred, clf_conf = classify_claim(claim_text, top_evidence)

    # ──────────────────────────────────────────────────────────
    # 4. NLI VOTE over retrieved matches
    #    (runs regardless of similarity — fixes the core bug)
    # ──────────────────────────────────────────────────────────
    if top_sim >= LOW_SIM_THRESHOLD:
        nli_out = nli_vote(matches, claim_text)
    else:
        nli_out = None   # nothing close enough to retrieve against

    # ──────────────────────────────────────────────────────────
    # 5. FUSION: combine NLI + classifier
    # ──────────────────────────────────────────────────────────
    if nli_out is not None:
        return combine_nli_and_classifier(nli_out, clf_pred, clf_conf, top_sim)

    # No NLI signal → fall back to classifier alone
    top_row = matches[0][0] if matches else None
    return {
        "verdict":       clf_pred,
        "confidence":    round(clf_conf, 4),
        "matched_claim": top_row["claim"] if top_row is not None else "",
        "similarity":    round(top_sim, 4),
        "method":        "classifier_only",
    }


if __name__ == "__main__":
    while True:
        q = input("\nEnter claim: ").strip()
        if not q:
            continue
        result = predict_claim(q)
        print("\n── Result ──────────────────────────────")
        for k, v in result.items():
            print(f"  {k:20s}: {v}")
        print("────────────────────────────────────────")
