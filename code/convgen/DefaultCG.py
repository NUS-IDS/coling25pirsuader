#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 17:21:32 2024

@author: sdas
"""
import openai
import random
import json
import os
from DialogActs import DBCAct_Counselor, DBCAct_Client
from Common import TEMPLATES, getClientResponse, getCounselorResponse
import RunConfig as config



if __name__=="__main__":
    
    
    data_q_p_dir=config.data_dir
    opdir=config.output_dir
    
    flist = os.listdir(data_q_p_dir) 
    
    
    config.dumpRunData(opdir+"/expt_settings.txt", TEMPLATES)
    
    for fx, fname in enumerate(flist):
        if ".json" not in fname or ("_A" not in fname and "_B" not in fname):
            continue
    
        
        print ("\n\nProcessing "+str(fx)+" "+fname)
        
        
        
        outfile=opdir+"/"+fname.replace("_data.json","_conv.json")
        
        #persona, problems, questions = extractPersonaAndProblem(inpfile)
        #treatment = getTreatments(questions)
        dinpfile=data_q_p_dir+"/"+fname
        data2 = json.load(open(dinpfile, "r", encoding="utf-8"))
        persona = data2["cpersona"]
        problems = data2["problems"]
        problems = data2["questions"]
        treatment = data2["treatment"]
      
    
    
        ch = random.randint(0, 1)
        if ch==0:
            srole="Client"
        else:
            srole="Counselor"
            
        if srole=="Client":
            other="Counselor"
        else:
            other="Client"
            
        print (srole)
        print (other)
        ch = random.randint(0, len(config.opening_greetings)-1)
        og= config.opening_greetings[ch]
        
        u=[srole+": "+og]
        data=[]
        rt="greet"
        data.append({"turn":1, "rtype":rt, "role": srole, "utterance": og})
        
        nturns = 10
        uturn=1
        for turn in range (0, nturns):
            print ("\n Turn="+str(turn))
            dialog=' '.join(u)
            
            if srole=="Client":
                uturn += 1
                temp = (getCounselorResponse(treatment, dialog))
                print ("ChatGPT Response="+ temp)
                rt, utterance = config.parseJSON(temp)
                data.append({"turn":uturn, "rtype":rt, "role": "Counselor", "utterance": utterance})
                if "Counselor:" not in utterance:
                    utterance="Counselor: "+utterance
                
                u.append(utterance)
                print (utterance)
                
                dialog=' '.join(u)
                uturn+=1
                temp = (config.getClientResponse(persona, str(problems), dialog))
                print ("ChatGPT Response="+ temp)
                rt, utterance = config.parseJSON(temp)
                data.append({"turn":uturn, "rtype":rt, "role": "Client", "utterance": utterance})
                if "Client:" not in utterance:
                    utterance="Client: "+utterance
                
                u.append(utterance)
                print (utterance)
            else: # srole=="Counselor":
                uturn+=1
                temp = (getClientResponse(persona, str(problems), dialog))
                print ("ChatGPT Response="+ temp)
                rt, utterance = config.parseJSON(temp)
                data.append({"turn":uturn, "rtype":rt, "role": "Client", "utterance": utterance})
                if "Client:" not in utterance:
                    utterance="Client: "+utterance
                
                u.append(utterance)
                print (utterance)
    
                dialog=' '.join(u)
                uturn+=1
                temp = (getCounselorResponse(treatment, dialog))
                print ("ChatGPT Response="+ temp)
                rt, utterance = config.parseJSON(temp)
                data.append({"turn":uturn, "rtype":rt, "role": "Counselor", "utterance": utterance})
                if "Counselor:" not in utterance:
                    utterance="Counselor: "+utterance
                
                u.append(utterance)
                print (utterance)
                
            if rt=="closing":
                print ("Breaking earlier at uturn="+str(uturn)+" nturns="+str(turn))
                break
                
        json.dump(data, open(outfile, "w"))















































