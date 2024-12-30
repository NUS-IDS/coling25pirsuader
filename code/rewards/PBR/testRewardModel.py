
import csv
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import math


model2path="./reward_model"
model2=AutoModelForSequenceClassification.from_pretrained(model2path)
tokenizer2=AutoTokenizer.from_pretrained(model2path)

def RoBERTa_reward(text):

    toks = tokenizer2.encode_plus(text, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    output = model2(input_ids = toks["input_ids"],
                attention_mask= toks["attention_mask"])

 #   print (type(output["logits"]))
 #   print (output["logits"])   
#    print (output["logits"][0].item())
    return 1 / (1 + math.exp(-output["logits"][0].item()))



if __name__=="__main__":

    inpfile="./forrl/AMT_dact_rewards.csv"

    fin = open (inpfile, "r")
    csvr = csv.reader(fin)

    header = []
    cr=0
    count=0
    for row in csvr:
        if len(header)==0:
            header = row

        cr += 1
        if cr > 10:
            break
        
        chosen = row[1]
        rejected = row[3]
        
        print (chosen)
        print ()
        print (rejected)
        print ()
        print ("crew="+str(RoBERTa_reward(chosen)))
        print ("rrew="+str(RoBERTa_reward(rejected)))

        if RoBERTa_reward(chosen) > RoBERTa_reward(rejected):
            count+=1

    print (count)
    print (100*(count/cr))
