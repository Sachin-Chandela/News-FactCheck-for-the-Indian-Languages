import numpy as np
import faiss
import json
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def build_comprehensive_index():
    print("🏥 Downloading Full PubHealth Dataset...")
    # Loading the full training split
    dataset = load_dataset("health_fact", split="train", trust_remote_code=True)
    
    print("🧹 Cleaning and deduplicating data...")
    documents = []
    seen_claims = set()
    
    for item in dataset:
        # Quality filter: Ensure there is an explanation and a claim
        if item['explanation'] and len(item['explanation']) > 150:
            clean_claim = item['claim'].strip().lower()
            if clean_claim not in seen_claims:
                doc = {
                    "claim": item['claim'],
                    "explanation": item['explanation'],
                    "label": item['label'],
                    "subjects": item['subjects']
                }
                documents.append(doc)
                seen_claims.add(clean_claim)

    print(f"🧬 Total unique medical facts: {len(documents)}")
    
    print("🧠 Vectorizing (this may take 5-10 mins on GPU)...")
    model = SentenceTransformer('intfloat/multilingual-e5-base')
    
    # We combine claim and a snippet of explanation for the best search context
    texts_to_embed = [f"passage: {d['claim']} {d['explanation'][:400]}" for d in documents]
    
    embeddings = model.encode(texts_to_embed, show_progress_bar=True, normalize_embeddings=True)
    
    # Save FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    
    faiss.write_index(index, "index/health_claims.index")
    with open("index/mapping.json", "w", encoding='utf-8') as f:
        json.dump(documents, f)
        
    print(f"✅ COMPREHENSIVE INDEX SAVED: {len(documents)} medical records indexed.")

if __name__ == "__main__":
    build_comprehensive_index()