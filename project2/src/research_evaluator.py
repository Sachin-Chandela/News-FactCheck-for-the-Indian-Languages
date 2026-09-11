import json
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class ResearchEvaluator:
    def __init__(self, test_set_path="test_set.json"):
        with open(test_set_path, "r") as f:
            self.test_data = json.load(f)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Paths to your trained models
        self.model_configs = {
            "Baseline (BERT)": "bert-base-uncased",
            "RoBERTa (Generic)": "roberta-base",
            "RoBERTa (Fine-tuned)": "../models/roberta_medical",
            "PubMedBERT (Proposed)": "../models/pubmed_medical"
        }

    def get_prediction(self, model_name, model_path, claim, evidence):
        tokenizer = AutoTokenizer.from_pretrained(model_path if "./" in model_path else model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        
        # Combine Claim and Evidence for Cross-Encoding logic
        inputs = tokenizer(claim, evidence, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            logits = model(**inputs).logits
            raw_pred = torch.argmax(logits, dim=1).item()
            probs = torch.softmax(logits, dim=1)
            confidence = torch.max(probs).item()
            
        # --- CRITICAL: LABEL MAPPING BRIDGE ---
        # Your test_set uses: 0=FALSE, 1=TRUE, 2=NEUTRAL
        # Your SciFact Trainer uses: 0=SUPPORT, 1=CONTRADICT, 2=NEUTRAL
        
        final_verdict = 2 # Default to Neutral
        
        if "Baseline" in model_name or "Generic" in model_name:
            # Vanilla models are random/unaligned, so we take raw output % 3
            final_verdict = raw_pred % 3
        else:
            # For your Fine-tuned models:
            if raw_pred == 0:   # Model said SUPPORT
                final_verdict = 1 # Map to TRUE
            elif raw_pred == 1: # Model said CONTRADICT
                final_verdict = 0 # Map to FALSE
            elif raw_pred == 2: # Model said NEUTRAL/NOINFO
                final_verdict = 2 # Map to NEUTRAL

        del model, tokenizer
        torch.cuda.empty_cache()
        return final_verdict, confidence

    def run_benchmark(self):
        results_table = {}

        for name, path in self.model_configs.items():
            print(f"🧪 Evaluating {name}...")
            y_true = []
            y_pred = []
            confidences = []

            for entry in self.test_data:
                # We use English Reference to test NLI logic specifically
                claim = entry['english_reference']
                evidence = entry['explanation']
                true_label = entry['label']

                try:
                    pred, conf = self.get_prediction(name, path, claim, evidence)
                    y_true.append(true_label)
                    y_pred.append(pred)
                    confidences.append(conf)
                except Exception as e:
                    print(f"Error testing {name} on ID {entry['id']}: {e}")

            # Metric Calculation
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            avg_conf = np.mean(confidences)

            results_table[name] = {
                "Accuracy": acc,
                "Macro-F1": f1,
                "Avg Confidence": avg_conf
            }

        self.display_results(results_table)

    def display_results(self, results):
        print("\n" + "="*70)
        print(f"{'Model Architecture':<25} | {'Accuracy':<10} | {'Macro-F1':<10} | {'Confidence'}")
        print("-" * 70)
        for name, metrics in results.items():
            print(f"{name:<25} | {metrics['Accuracy']:<10.2%} | {metrics['Macro-F1']:<10.3f} | {metrics['Avg Confidence']:.2%}")
        print("="*70)

if __name__ == "__main__":
    evaluator = ResearchEvaluator()
    evaluator.run_benchmark()