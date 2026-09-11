import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,    
)

def load_local_scifact(claims_path, corpus_path):
    # Mapping SciFact labels to NLI logic
    LABEL_MAP = {"SUPPORT": 0, "CONTRADICT": 1}
    
    # 1. Load the files
    claims_df = pd.read_csv(claims_path)
    corpus_df = pd.read_csv(corpus_path)
    
    # 2. Join Claims with Corpus to get the actual Evidence Text
    # We assume claims_csv has 'claim' and 'corpus_id' (or similar) to link them
    # If your CSVs are already paired, this step simplifies to just reading.
    # Adjusting for a standard paired format:
    df = claims_df.copy()
    
    def process_row(row):
        claim = str(row["claim"])
        # If your CSV has an 'evidence' column, use it. 
        # If not, we'd merge with corpus_df here.
        evidence = str(row.get("evidence_sentence", "")) 
        label = LABEL_MAP.get(str(row["evidence_label"]).strip().upper(), 2)
        return claim, evidence, label

    processed_data = df.apply(process_row, axis=1, result_type='expand')
    processed_data.columns = ["claim", "evidence", "label"]
    
    print(f"Loaded {len(processed_data)} samples from {claims_path}")
    print("Label distribution:", processed_data["label"].value_counts().to_dict())
    
    return Dataset.from_pandas(processed_data, preserve_index=False)

def train_research_models():
    # Your target architectures
    model_configs = [
        {"name": "roberta-base", "save_path": "../models/roberta_medical"},
        {"name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", "save_path": "../models/pubmed_medical"},
    ]

    # Load local data
    train_ds = load_local_scifact("../data/SciFact/claims_train.csv", "../data/SciFact/corpus_train.csv")
    val_ds = load_local_scifact("../data/SciFact/claims_validation.csv", "../data/SciFact/corpus_train.csv")

    for config in model_configs:
        print(f"\n🚀 Training {config['name']}...")

        tokenizer = AutoTokenizer.from_pretrained(config["name"])
        model = AutoModelForSequenceClassification.from_pretrained(
            config["name"], num_labels=3, use_safetensors=True,
        )

        def tokenize_function(examples):
            # CRITICAL: This passes BOTH claim and evidence to the model
            return tokenizer(
                examples["claim"],
                examples["evidence"],
                padding="max_length",
                truncation=True,
                max_length=512,
            )

        def prepare(ds):
            # Tokenize and format for Trainer
            tok = ds.map(tokenize_function, batched=True, remove_columns=["claim", "evidence"])
            tok = tok.rename_column("label", "labels")
            return tok.to_list()

        train_tok = prepare(train_ds)
        val_tok = prepare(val_ds)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        args = TrainingArguments(
            output_dir=config["save_path"],
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            fp16=torch.cuda.is_available(),
            save_strategy="no",
            eval_strategy="epoch",
            logging_steps=50,
            report_to="none",
            learning_rate=2e-5, # Standard for NLI fine-tuning
            weight_decay=0.01
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            data_collator=data_collator,
        )
        
        trainer.train()
        model.save_pretrained(config["save_path"])
        tokenizer.save_pretrained(config["save_path"])
        print(f"✅ Saved to {config['save_path']}")

if __name__ == "__main__":
    train_research_models()