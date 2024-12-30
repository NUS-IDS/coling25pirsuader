from transformers import AutoModelForSequenceClassification, AutoTokenizer
import math
import PBRConfig as config

STRINGMARKER="The last few acts"
#   ACTSOFINTEREST=["deny_to_try","express_interest","neutral_to_information","counter_information"]
CONV_DELIMITER='Conversation:'
model2path=config.reward_model_path #
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


 
    if True: #pact!="" and pact in ACTSOFINTEREST and conversation!="":
        reward = RoBERTa_reward(conversation)
        return reward
    else:
        return 0.0

