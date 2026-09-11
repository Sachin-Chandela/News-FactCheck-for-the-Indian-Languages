import pickle
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Embedding device:", device)

df = pd.read_csv("../data/final_dataset.csv")
claims = df["claim"].astype(str).tolist()

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device=device
)

embeddings = model.encode(
    claims,
    batch_size=128,
    show_progress_bar=True,
    convert_to_numpy=True
)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, "../models/fact_index.faiss")

with open("../models/claims.pkl", "wb") as f:
    pickle.dump(claims, f)

print("Saved retriever files.")