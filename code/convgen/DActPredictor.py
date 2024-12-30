#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:33:21 2024

@author: sdas
"""

from DialogActs import DBCAct_Counselor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import RunConfig as config

model_path=config.dact_model_path
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
max_source_length=512
max_target_length=16
padding="max_length"


def getAsOptString(dacts):

    top=""
    for key in dacts:
        #top+=key+": "+dacts[key]+"\n"
        top+=key+"\n"

    return top.strip().replace("\n","; ")



def getSourceForPrediction(pu, pacts):

    history=[]
    cwindowsz=4

    if len(pu)>cwindowsz:
        history = pu[len(pu)-cwindowsz:len(pu)]

    else:
        history = pu


    pconv = '\n'.join(history)
    if len(pacts)==0:
        pact="[CONVSTART]"
    else:
        if len(pacts)>cwindowsz:
            pacts=pacts[len(pacts)-cwindowsz:]

        pact='; '.join(pacts)

    optstring = getAsOptString(DBCAct_Counselor)
  
    source = ("Question: What dialog act must the Counselor use from the list of acts and their definitions given below? The last few acts were {"+pact+"}\nList of acts are \""+optstring+"\"?")

    source += "\nConversation: "+pconv.strip()
    source +="\nAnswer: "

    return source



def get_counselor_dact(u, uacts):
    input_str = getSourceForPrediction(u, uacts)
    input_ids = tokenizer.encode(input_str, return_tensors="pt")
    res = model.generate(input_ids, max_length=512) #k_max_tgt_len, **generator_args)
    op = tokenizer.batch_decode(res, skip_special_tokens=True)[0]

    return op