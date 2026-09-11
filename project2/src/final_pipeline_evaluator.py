import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    AutoModelForSequenceClassification
)

# --- REUSING YOUR ROBUST CLEANUP ---
def clear_vram():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

class ResearchEvaluator:
    def __init__(self, test_set_path="../data/test_set.json"):
        with open(test_set_path, "r") as f:
            self.test_data = json.load(f)
        
        # Defining the 4 Research Scenarios
        self.scenarios = {
            "Baseline (Direct Hinglish)": "bert-base-uncased",
            "RoBERTa (Translated)": "roberta-base",
            "RoBERTa FT (Translated)": "../models/roberta_medical",
            "PubMedBERT FT (Translated)": "../models/pubmed_medical"
        }

    def translate_via_llama(self, hinglish_text):
        """Uses your Llama-3.2 logic to translate claims."""
        model_id = "meta-llama/Llama-3.2-3B-Instruct"
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        
        tok = AutoTokenizer.from_pretrained(model_id)
        mod = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0})
        
        messages = [{"role": "system", "content": "Translate Hinglish health claim to one formal English sentence. Output ONLY translation."},
                    {"role": "user", "content": hinglish_text}]
        
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(mod.device)
        
        with torch.no_grad():
            out = mod.generate(**inputs, max_new_tokens=40, eos_token_id=tok.eos_token_id)
        
        decoded = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        english = decoded.strip().split('\n')[0].replace('"', '')
        
        del mod, tok
        clear_vram()
        return english

    def get_nli_verdict(self, model_path, claim, evidence):
        """Standard NLI judging logic with SciFact label mapping."""
        tok = AutoTokenizer.from_pretrained(model_path if "./" in model_path else model_path)
        mod = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3).to("cuda")
        
        # Cross-Encoder Input
        inputs = tok(claim, evidence, truncation=True, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            logits = mod(**inputs).logits
            raw_pred = torch.argmax(logits, dim=1).item()
            conf = torch.softmax(logits, dim=1).max().item()

        # RESEARCH MAPPING:
        # SciFact/NLI: 0=SUPPORT, 1=CONTRADICT, 2=NEUTRAL
        # Our TestSet: 1=TRUE, 0=FALSE, 2=NEUTRAL
        mapping = {0: 1, 1: 0, 2: 2}
        final_label = mapping.get(raw_pred, 2)
        
        del mod, tok
        clear_vram()
        return final_label, conf

    def run_experiment(self):
        final_results = {}

        # 1. First, pre-translate all Hinglish claims once to save Llama loading time
        print("🌍 [Phase 1] Pre-translating Hinglish claims using Llama-3.2...")
        translated_claims = []
        for entry in self.test_data:
            translated_claims.append(self.translate_via_llama(entry['hinglish']))

        # 2. Run each model scenario
        for name, path in self.scenarios.items():
            print(f"🔬 [Phase 2] Evaluating Flow: {name}")
            y_true, y_pred, confs = [], [], []

            for i, entry in enumerate(self.test_data):
                evidence = entry['explanation']
                true_label = entry['label']
                
                # Logic: Baseline gets raw Hinglish, others get Llama's English
                current_claim = entry['hinglish'] if "Direct" in name else translated_claims[i]
                
                pred, conf = self.get_nli_verdict(path, current_claim, evidence)
                
                y_true.append(true_label)
                y_pred.append(pred)
                confs.append(conf)

            final_results[name] = {
                "Accuracy": accuracy_score(y_true, y_pred),
                "F1": f1_score(y_true, y_pred, average='macro'),
                "Confidence": np.mean(confs)
            }

        self.display_table(final_results)

    def display_table(self, res):
        print("\n" + "="*80)
        print(f"{'Experimental Pipeline Flow':<35} | {'Acc':<8} | {'F1':<8} | {'Conf'}")
        print("-" * 80)
        for name, m in res.items():
            print(f"{name:<35} | {m['Accuracy']:<8.2%} | {m['F1']:<8.3f} | {m['Confidence']:.2%}")
        print("="*80)

if __name__ == "__main__":
    evaluator = ResearchEvaluator()
    evaluator.run_experiment()