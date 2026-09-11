

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME  = "google/muril-base-cased"
OUTPUT_DIR  = "../models/muril"
MAX_LENGTH  = 256          # longer to fit claim + evidence
BATCH_SIZE  = 16           # reduce if OOM; gradient accumulation below compensates
EPOCHS      = 5
LR          = 2e-5
WARMUP_RATIO = 0.1

LABEL2ID = {"true": 0, "false": 1, "misleading": 2, "other": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Load & validate data
# ──────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("../data/final_dataset.csv")
df["claim"]    = df["claim"].astype(str).str.strip()
df["evidence"] = df["evidence"].fillna("").astype(str).str.strip()

# Remap string labels if needed
if df["label"].dtype == object:
    df["label"] = df["label"].str.lower().map(LABEL2ID)

df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

print("\nLabel distribution:")
print(df["label"].value_counts().sort_index().rename(ID2LABEL))

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Class weights  (handles imbalanced datasets)
# ──────────────────────────────────────────────────────────────────────────────
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_LABELS),
    y=df["label"].values,
)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print("\nClass weights:", class_weights)

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Train / val split
# ──────────────────────────────────────────────────────────────────────────────
train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    stratify=df["label"],
    random_state=42,
)
print(f"\nTrain: {len(train_df)}  |  Val: {len(val_df)}")

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Tokenise   [claim] [SEP] [evidence]
#     This is the single biggest quality improvement over the original.
#     MuRIL now sees the supporting/contradicting evidence while learning.
# ──────────────────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["claim"],          # sentence A
        batch["evidence"],       # sentence B  ← KEY CHANGE
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True))

train_ds = train_ds.map(tokenize, batched=True)
val_ds   = val_ds.map(tokenize,   batched=True)

keep_cols = ["input_ids", "attention_mask", "token_type_ids", "label"]
train_ds  = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
val_ds    = val_ds.remove_columns([c for c in val_ds.column_names   if c not in keep_cols])

train_ds.set_format("torch")
val_ds.set_format("torch")

# ──────────────────────────────────────────────────────────────────────────────
# 5.  Model
# ──────────────────────────────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    label2id=LABEL2ID,
    id2label=ID2LABEL,
    ignore_mismatched_sizes=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# 6.  Custom Trainer with weighted cross-entropy loss
# ──────────────────────────────────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        loss_fn = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(logits.device),
            label_smoothing=0.1,       # softens over-confident predictions
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Metrics
# ──────────────────────────────────────────────────────────────────────────────
accuracy_metric = evaluate.load("accuracy")
f1_metric       = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1  = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]

    # per-class report printed to console (not returned to Trainer)
    print("\n" + classification_report(
        labels, preds,
        target_names=[ID2LABEL[i] for i in range(NUM_LABELS)],
        zero_division=0,
    ))
    return {"accuracy": acc, "f1_macro": f1}

# ──────────────────────────────────────────────────────────────────────────────
# 8.  Training arguments
# ──────────────────────────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    eval_strategy="epoch",          # ← was evaluation_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    report_to="none",
)

# ──────────────────────────────────────────────────────────────────────────────
# 9.  Train
# ──────────────────────────────────────────────────────────────────────────────
trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

# ──────────────────────────────────────────────────────────────────────────────
# 10. Save
# ──────────────────────────────────────────────────────────────────────────────
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nSaved model to {OUTPUT_DIR}")

# Final val report
print("\n── Final validation metrics ──")
metrics = trainer.evaluate()
for k, v in metrics.items():
    print(f"  {k}: {round(v, 4)}")
