#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:06:10 2024

@author: sdas
"""

import json

##########3
#Configurable params here

OPENAI_KEY="TBD"
OPENAI_MODEL="gpt-4o-mini"

dact_model_path="/tmp/car_dacts_model"
data_dir="../../data/test_data"
output_dir="/tmp/convgen_car_op"


#############Some utility functions

def parseJSON(opstring):
    opstring = opstring.replace("```json","").replace("```","")
    try:
        pl = json.loads(opstring)
        return pl["ResponseType"], pl["Utterance"]
    except ValueError:
        print ("JSON appears invalid, breaking by new line ")
        return opstring.split("\n")

opening_greetings=[
    "Hello", "hi", "Hi there!", "Hi there! Good afternoon!",
    "Hello, nice to see you here"," Hello, How are you?",
    " Hello, How is it going? ", " Hello, How have you been?",
    "Good morning!", "Good evening!", "Hello, thank you for seeing me today."
    ]


def dumpRunData(inpfile, TEMPLATES):
    
    fout = open (inpfile, "w")
    fout.write ("PROMPTS \n"+str(TEMPLATES)+"\n")

    fout.flush()
    fout.close()
