from transformers import AutoModelForSequenceClassification, AutoTokenizer
import math
import csv
import torch
import CARConfig as config

CLASS_MAP = {
    cn: i for i, cn in enumerate(["1","2","3","4","5"])
}
iCLASS_MAP = {
    v: k for k, v in CLASS_MAP.items()
}

model_path=config.reward_model_path

CONV_DELIMITER='Conversation:'
STRINGMARKER="The last few acts"
ACTSOFINTEREST=["deny_to_try","express_interest","neutral_to_information","counter_information"]

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


def compute_reward(source):
    
    if CONV_DELIMITER not in source:
 #       print ("Could not find delimiter in source, using the whole thing "+CONV_DELIMITER)
 #       print (source+"\n\n")
        conversation = source.replace("Answer:","").strip()
    else:   
        conversation = source.split(CONV_DELIMITER)[1].replace("Answer:","").strip()

    pact=""
    
    if STRINGMARKER in source:
        pacts = source.split(STRINGMARKER)[1]
        pacts = pacts.split("{")[1]
        pacts = pacts.split("}")[0].strip()
        if ";" not in pacts:
            pact = pacts.strip()
        else:
            pact = pacts.split(";")[-1].strip()


 #   print ("DEBUG pacts="+pacts+", selected="+pact+", of interest="+str(pact in ACTSOFINTEREST))
    if True: #pact!="" and pact in ACTSOFINTEREST and conversation!="":
        reward = RoBERTa_reward(conversation)
 #       print ("DEBUG pacts="+pacts+", selected="+pact+", reward="+str(reward))
        return reward
    else:
        return 0.0


