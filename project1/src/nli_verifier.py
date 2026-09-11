import torch
from transformers import pipeline

device_id = 0 if torch.cuda.is_available() else -1

nli = pipeline(
    "text-classification",
    model="joeddav/xlm-roberta-large-xnli",
    device=device_id
)

def verify(premise, hypothesis):
    text = f"{premise} </s></s> {hypothesis}"
    result = nli(text)[0]
    return result