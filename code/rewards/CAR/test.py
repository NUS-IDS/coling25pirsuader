import torch
from transformers import pipeline
from utils import iCLASS_MAP

model_path="scratch/results/best_roberta/"

source="Counselor: With insulin, you can have more flexibility in your meal timing and portions as it helps maintain stable blood sugar levels throughout the day. Client: I am willing to give insulin a try considering its benefits. Counselor: Thank you for considering insulin; it could greatly improve your diabetes management. Client: Goodbye, and thank you for your guidance."

classifier = pipeline('text-classification',model_path)
result = classifier(source)
print (result)

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import math

MAXREWARD=5
model2=AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer2=AutoTokenizer.from_pretrained(model_path)

def RoBERTa_reward(text):

    toks = tokenizer2.encode_plus(text, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    output = model2(input_ids = toks["input_ids"],
                attention_mask= toks["attention_mask"])

 #   print (type(output["logits"]))
#    print (output["logits"])   
    prediction = torch.argmax(output.logits, dim=1)
 #   print (prediction)
    return float(iCLASS_MAP[prediction.item()])/MAXREWARD

print(RoBERTa_reward(source))
