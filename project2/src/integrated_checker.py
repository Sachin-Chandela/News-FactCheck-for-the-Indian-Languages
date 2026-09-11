import torch
import gc
import json
import faiss
import numpy as np
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    AutoModelForSequenceClassification
)
from sentence_transformers import SentenceTransformer

def clear_vram():
    """Aggressive VRAM cleanup for Fedora/Linux."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    with torch.no_grad():
        torch.cuda.empty_cache()

class MedicalFactChecker:
    def __init__(self, index_dir="index"):
        print("⚡ Initializing Fact-Check Engine...")
        self.index = faiss.read_index(f"{index_dir}/health_claims.index")
        with open(f"{index_dir}/mapping.json", "r", encoding='utf-8') as f:
            self.mapping = json.load(f)

    def translate_hinglish(self, text):
        print("\n🤖 [Step 1] Translating Hinglish -> English...")
        model_id = "meta-llama/Llama-3.2-3B-Instruct"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        tok = AutoTokenizer.from_pretrained(model_id)
        mod = AutoModelForCausalLM.from_pretrained(
            model_id, 
            quantization_config=bnb_config, 
            device_map={"": 0}
        )
        
        messages = [
            {"role": "system", "content": "Translate the user's Hinglish health claim to one formal English sentence. Output ONLY the translation."},
            {"role": "user", "content": text}
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(mod.device)
        
        with torch.no_grad():
            out = mod.generate(**inputs, max_new_tokens=40, eos_token_id=tok.eos_token_id)
        
        # Extract only the new assistant tokens
        decoded = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        english = decoded.strip().split('\n')[0].replace('"', '')
        
        del mod, tok
        clear_vram()
        return english

    def retrieve_evidence(self, query, k=3):
        print(f"🔎 [Step 2] Searching for top {k} matches...")
        model = SentenceTransformer('intfloat/multilingual-e5-base')
        
        q_emb = model.encode([f"query: {query}"], normalize_embeddings=True)
        dist, idx = self.index.search(np.array(q_emb).astype('float32'), k=k)
        
        # Return lists of matches and their scores
        matches = [self.mapping[i] for i in idx[0]]
        scores = [float(s) for s in dist[0]]
        
        del model
        clear_vram()
        return matches, scores

    def get_verdict(self, claim, evidence_text):
        print("⚖️ [Step 3] Judging Claim against Evidence...")
        model_id = "cross-encoder/nli-deberta-v3-small"
        
        tok = AutoTokenizer.from_pretrained(model_id)
        mod = AutoModelForSequenceClassification.from_pretrained(model_id).to("cuda")
        
        inputs = tok(claim, evidence_text, truncation=True, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            logits = mod(**inputs).logits
            # Label 0: Contradiction, 1: Entailment, 2: Neutral
            prediction = torch.softmax(logits, dim=1).argmax().item()
        
        labels = ["🔴 FALSE (Contradicts Research)", "🟢 TRUE (Supported)", "🟡 UNCLEAR (Mixed Evidence)"]
        
        del mod, tok
        clear_vram()
        return labels[prediction]

if __name__ == "__main__":
    checker = MedicalFactChecker()
    
    while True:
        user_input = input("\nEnter Health Claim (or 'exit' to quit): ")
        if user_input.lower() == 'exit': break
        
        try:
            # 1. TRANSLATE
            eng_claim = checker.translate_hinglish(user_input)
            print(f"✨ English: {eng_claim}")
            
            # 2. RETRIEVE TOP 3
            matches, scores = checker.retrieve_evidence(eng_claim, k=3)
            
            print("\n📚 --- TOP 3 DATABASE MATCHES ---")
            results_to_judge = []
            
            for i in range(len(matches)):
                # We judge every one of the top 3 to show you the full logic
                verdict = checker.get_verdict(eng_claim, matches[i]['explanation'])
                
                print(f"Match {i+1} (Score: {scores[i]:.2f}):")
                print(f"   Claim: {matches[i]['claim']}")
                print(f"   Verdict: {verdict}")
                print("-" * 30)
                
                results_to_judge.append({
                    "verdict": verdict,
                    "evidence": matches[i],
                    "score": scores[i]
                })

            # 3. FINAL AGGREGATED DECISION
            # Logic: Prioritize TRUE or FALSE over UNCLEAR. 
            # If all are UNCLEAR, take the highest score.
            final_report = results_to_judge[0] # Default to top score
            for res in results_to_judge:
                if "TRUE" in res['verdict'] or "FALSE" in res['verdict']:
                    final_report = res
                    break

            print(f"\n📢 --- FINAL SYSTEM REPORT ---")
            print(f"Input: {user_input}")
            print(f"Decision Based On Match: {final_report['evidence']['claim']}")
            print(f"VERDICT: {final_report['verdict']}")
            print(f"Context: {final_report['evidence']['explanation'][:250]}...")

        except Exception as e:
            print(f"⚠️ Error: {e}")
            clear_vram()