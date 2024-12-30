#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 17:21:32 2024

"""
import random
import json
import os

from DialogActs import DBCAct_Counselor, DBCAct_Client
from Common import getClientResponse, getChatGPTResponse, TEMPLATES
from DActPredictor import get_counselor_dact
import RunConfig as config


TEMPLATES2={
    "counselor_sys":("In the following conversation, \
    you will play a counselor who wants to persuade a\
        diabetic patient to try insulin for better control \
    of their health. The available response types to use are defined as %s").replace("[ ]+"," ").strip(),
    
    "counselor_usr": ("You are a counselor who  \
        tries to persuades a client about trying insulin. \
            Please incorporate the information  \
        from %s while convincing the client\nPlease reply with \
    only one short and succinct sentence using the response type %s. \
    Now start the chat. The output is a JSON tuple with {\"ResponseType\":, \"Utterance\":\"} \
            Take the previous conversation into account %s").replace("[ ]+"," ").strip()
    }

ENDACTS=['acknowledge','thank','end_conversation','closing']
ENDACTSEARLY=['chitchat','acknowledge','support','motivate','compliment','sympathize']

def getCounselorResponse(issue_solution, dbcact, pconv):
    
    COUNSELOR_SYS= TEMPLATES2["counselor_sys"]%(str(DBCAct_Counselor))
    COUNSELOR_USR= TEMPLATES2["counselor_usr"]% (issue_solution, dbcact, pconv)
            
    return getChatGPTResponse(COUNSELOR_USR, COUNSELOR_SYS)
    



if __name__=="__main__":
    
    data_q_p_dir = config.data_dir
    opdir = config.output_dir
    flist = os.listdir(data_q_p_dir) 
    
    
    config.dumpRunData(opdir+"/expt_settings.txt", TEMPLATES2)
    
    for fx, fname in enumerate(flist):
        if ("_A" not in fname and "_B" not in fname):
            continue
    
        
        print ("\n\nProcessing "+str(fx)+" "+fname)
        
        
        outfile=opdir+"/"+fname.replace("_data.json","_conv.json")
        
  
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
        og=config.opening_greetings[ch]
        
        u=[srole+": "+og]
        data=[]
        rt="greet"
        uacts=[rt]
        data.append({"turn":1, "rtype":rt, "role": srole, "utterance": og})
        deny_count=0
        nturns = 10
        uturn=1
        for turn in range (0, nturns):
            print ("\n Turn="+str(turn))
            dialog=' '.join(u)
            
            if srole=="Client":
                uturn += 1
                if deny_count>1 and turn>5:
                    ch = random.randint(0, len(ENDACTS)-1)
                    rt = ENDACTS[ch]
                    print ("Repeated denial, Choosing random act="+rt)
                elif deny_count>1 and turn<5:
                    ch = random.randint(0, len(ENDACTSEARLY)-1)
                    rt = ENDACTSEARLY[ch]
                    print ("Repeated denial, Choosing random act="+rt)
                else:
                    rt = get_counselor_dact(u, uacts)
                    

                temp = (getCounselorResponse(treatment, rt, dialog))
                print ("ChatGPT Response="+ temp)
                _, utterance = config.parseJSON(temp)
                
                data.append({"turn":uturn, "rtype":rt, "role": "Counselor", "utterance": utterance})
                if "Counselor:" not in utterance:
                    utterance="Counselor: "+utterance
                
                u.append(utterance)
                uacts.append(rt)

                print (utterance)
                
                dialog=' '.join(u)
                uturn+=1
                temp = (getClientResponse(persona, str(problems), dialog))
                print ("ChatGPT Response="+ temp)
                rt, utterance = config.parseJSON(temp)
                data.append({"turn":uturn, "rtype":rt, "role": "Client", "utterance": utterance})
                if "Client:" not in utterance:
                    utterance="Client: "+utterance
                
                u.append(utterance)
                uacts.append(rt)
                print (utterance)
                if rt.startswith("deny"):
                    deny_count+=1
            else: # srole=="Counselor":
                uturn+=1
                temp = (getClientResponse(persona, str(problems), dialog))
                print ("ChatGPT Response="+ temp)
                rt, utterance = config.parseJSON(temp)
                data.append({"turn":uturn, "rtype":rt, "role": "Client", "utterance": utterance})
                if "Client:" not in utterance:
                    utterance="Client: "+utterance
                
                u.append(utterance)
                uacts.append(rt)
                print (utterance)
                if rt.startswith("deny"):
                    deny_count+=1
    
                dialog=' '.join(u)
                uturn+=1

                if deny_count>1 and turn>5:
                    ch = random.randint(0, len(ENDACTS)-1)
                    rt = ENDACTS[ch]
                    print ("Repeated denial, Choosing random act="+rt)
                elif deny_count>1 and turn<5:
                    ch = random.randint(0, len(ENDACTSEARLY)-1)
                    rt = ENDACTSEARLY[ch]
                    print ("Repeated denial, Choosing random act="+rt)
                else:
                    rt = get_counselor_dact(u, uacts)

                temp = (getCounselorResponse(treatment, rt, dialog))

                print ("ChatGPT Response="+ temp)
                _, utterance = config.parseJSON(temp)
                data.append({"turn":uturn, "rtype":rt, "role": "Counselor", "utterance": utterance})
                if "Counselor:" not in utterance:
                    utterance="Counselor: "+utterance
                
                u.append(utterance)
                print (utterance)
                uacts.append(rt)

            if rt=="closing":
                print ("Breaking earlier at uturn="+str(uturn)+" nturns="+str(turn))
                break
                
        json.dump(data, open(outfile, "w"))















































