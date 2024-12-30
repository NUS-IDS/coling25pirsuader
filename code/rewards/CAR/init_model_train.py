import sys
import random

import numpy as np
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification

from utils import CLASS_MAP

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("TRAINING on ", device)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def preprocess(item):
    #print (item.keys())
    item["label"] = torch.tensor(CLASS_MAP[str(item["reward"])]).unsqueeze(0)

    return item

model_path="roberta-large"
# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)


def tokenize(examples):
    toks = tokenizer.batch_encode_plus(examples["source"], padding="max_length", max_length=512, truncation=True,
                                       return_tensors="pt")
    toks["labels"] = examples["reward"]

    return toks



dataset = load_dataset('csv', data_files={'train': "../../../data/forrl/train_car_data.csv"}).map(preprocess) \
    .shuffle(seed=seed) \
    .map(tokenize, batched=True)

dataset = dataset["train"].train_test_split(test_size=0.1)
print (type(dataset))
print (dataset.keys())
# model
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(CLASS_MAP))

# fine-tuning
training_args = TrainingArguments(
    output_dir='scratch/results',
    num_train_epochs=20,
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='scratch/logs',
    logging_steps=10,
    evaluation_strategy='steps',
    save_total_limit=5,
    load_best_model_at_end=True,
    greater_is_better=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)

# actual training
trainer.train()

# loading for prediction
best_path = 'scratch/results/best_roberta'
trainer.save_model(best_path)
print ("Best model saved to "+best_path)
print("Done")
