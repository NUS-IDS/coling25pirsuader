# https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/flan-t5-samsum-summarization.ipynb
#####
# This code is basically borrowed from the above URL and edited for our purpose

from datasets import load_dataset
from random import randrange
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch

import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
nltk.download("punkt")

import FlanTrainConfig as config
######GSDAS SET MODEL NAME AND PARAMETERS HERE
##TRAINING PARAMETERS AT TOWARDS THE END OF THE FILE
model_id=config.model_id
max_source_length=config.max_source_length
max_target_length=config.max_target_length
outdir=config.model_out_path
##########


tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.add_tokens(['[SEP]']) #
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)


# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels


########## Load dataset from the hub
dataset = load_dataset('csv', data_files={'train': config.train_csv, 'test':config.test_csv})

print(f"Train dataset size: {len(dataset['train'])}")
dataset["validation"] = dataset["test"]




sample = dataset['train']



def preprocess_function(sample,padding="max_length"):
    # add prefix to the input for t5
    #inputs = [prompt+ " "+ item for item in sample["source"]]
    inputs = [item for item in sample["source"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["target"], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["source", "target"])
#print(f"Keys of tokenized dataset: {list(tokenized_dataset.features)}")


# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)


###GSDAS EDIT TRAINING PARAMETERS HERE, IF NEEDED
# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir="/tmp/scratch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    fp16=False, # Overflows with fp16
    learning_rate=5e-5,
    num_train_epochs=3,
    # logging & evaluation strategies
    logging_dir="/tmp/scratch",
    logging_strategy="steps",
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=None, #compute_metrics,
)

# Start training
trainer.train()
model.save_pretrained(outdir) 
tokenizer.save_pretrained(outdir) 
print ("Model written to "+outdir)
