#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 17:21:32 2024

@author: sdas
"""

import json
import sys
import os

import csv
import CGWithPredictedActs as cgen
import RunConfig as config
import Common as common



SYSPROMPT="As a conversation analysis model, given a conversation "+\
    "between a Counselor and a Client, decide if the Client sounds open to "+\
        "trying insulin for their diabetes control."


cacts={
    'deny_to_try':{  'ask_concerns',
    'logical_appeal', 'emotion_appeal',
    'credibility_appeal', 'ask_about_consequence',
    'ask_about_antecedent', 'suggest_a_reason',
    'motivate'},

    'express_interest': {'amplify_excitement', 'motivate','compliment','provide_insulin_information','support'},
    'neutral_to_information': {'amplify_excitement', 'motivate','compliment','provide_insulin_information'},
    'counter_information': {'amplify_excitement', 'motivate','compliment','provide_insulin_information'},
    }


def compute_reward2(conversation):
    
    if conversation=="":
        return 0.0
    response = common.getChatGPTResponse("Given the following conversation, rate how "+\
            "likely is the Client may be willing to try insulin after the Counselor's last statement "+\
            "on a scale of 1 to 5, where 1 is most unlikely and 5 is most likely. "+\
            "Only print your score. \nConversation: "+conversation, SYSPROMPT)
    return response

def getRewardsForFile(inpfile, outfile):

    
    fin = open (inpfile, "r")
    csvr = csv.reader(fin)

    fout = open (outfile, "w")
    csvw = csv.writer(fout, delimiter=",", quoting=csv.QUOTE_ALL)
    header=[]
    nr=0
    for row in csvr:
        if len(header)==0:
            header=row
            header.append("Reward")
            csvw.writerow(header)
            continue

        conv_extra2 = row[-1]
        reward = compute_reward2(conv_extra2)
        row.append(reward)
        csvw.writerow(row)
        fout.flush()
        if nr%50==0:
            print()
            print (conv_extra2)
            print (reward)
        nr+=1

    fout.close()


if __name__=="__main__":
    
    
    if len(sys.argv) < 3:
        
        print ("Expected args1: conv-dir, args2: out-csv-path")
        sys.exit(1)
    
    
    data_q_p_dir= config.data_dir
    convdir = sys.argv[1]
    outfile = sys.argv[2]
        
    flist = os.listdir(data_q_p_dir) 
    ctxtwinsz=4
    

    fout = open (outfile, "w", encoding='utf-8')
    csvw = csv.writer(fout, delimiter=",", quoting=csv.QUOTE_ALL)
    csvw.writerow(["fname","clrtype","cortype","csofar", "conversation"])
    noprows=0
    for fx, fname in enumerate(flist):
        if ("_A" not in fname and "_B" not in fname):
            continue
    
        
        print ("\n\nProcessing "+str(fx)+" "+fname)
    
        dinpfile=data_q_p_dir+"/"+fname
        data2 = json.load(open(dinpfile, "r", encoding="utf-8"))
        persona = data2["cpersona"]
        problems = data2["problems"]
        problems = data2["questions"]
        treatment = data2["treatment"]

        cinpfile=convdir+"/"+fname.replace("_data.json","_conv.json")
        conv = json.load(open(cinpfile, "r", encoding="utf-8"))
        for tx, turn in enumerate(conv):
            turnid = turn["turn"]
            rtype = turn["rtype"]
            role = turn["role"]
            utterance = turn["utterance"]

            if role=="Client":
                if rtype in cacts:

                    for crt in cacts[rtype]:
                        newconv=[]
                        for ptx in range(tx-ctxtwinsz, tx+1):
                            if ptx>0:
                                newconv.append(conv[ptx]["role"]+": "+conv[ptx]["utterance"])
                        
                        csofar=' '.join(newconv)

                        temp = (cgen.getCounselorResponse(treatment, crt, ' '.join(newconv)))
                        _, cutterance = config.parseJSON(temp)
                        
                        newconv.append("Counselor: "+cutterance)
                        temp = (cgen.getClientResponse(persona, str(problems), ' '.join(newconv)))
                        _, clutterance = config.parseJSON(temp)
                        newconv.append("Client: "+clutterance)
                        csvw.writerow([fname, rtype, crt, csofar, ' '.join(newconv)])
                        fout.flush()
                        noprows += 1

                        
                print ("Written "+str(noprows))

    fout.close()

    outfile2 = outfile.replace(".csv","_rewards.csv")
    getRewardsForFile(outfile, outfile2)












































