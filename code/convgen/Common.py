#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:41:48 2024

@author: sdas
"""

from DialogActs import DBCAct_Client, DBCAct_Counselor
from RunConfig import OPENAI_KEY, OPENAI_MODEL
import time
import openai


CHATGPT_MODEL = OPENAI_MODEL
openai.api_key = OPENAI_KEY


TEMPLATES={
    
    "client_sys":("In the following conversation, \
    you will play as a client who is a diabetic patient \
        chatting with a counselor regarding your health. ").replace("[ ]+"," ").strip(),
    #with the \
#following persona "+
    
    "client_usr":("You are the client who is not sure about \
        trying insulin due to a list of concerns you have, \
        listed as follows\n %s \nPlease reply with only one \
            short and succinct sentence. \
            Use one of the response types from \
    %s Now start the chat. Mention what response type you are using \
        from the provided list. The output \
            is a JSON tuple with {\"ResponseType\":, \"Utterance\":\"} \
            Take the previous conversation into account %s").replace("[ ]+"," ").strip(),
    
    "counselor_sys":("In the following conversation, \
    you will play a counselor who wants to persuade a\
        diabetic patient to try insulin for better control \
    of their health.").replace("[ ]+"," ").strip(),
    
    "counselor_usr": ("You are a counselor who  \
        tries to persuades a client about trying insulin. \
            Please incorporate the information  \
        from %s while convincing the client\nPlease reply with \
    only one short and succinct sentence. Use one of the response types from \
    %s Now start the chat. Mention what response type you are using \
        from the provided list.\
        The output \
            is a JSON tuple with {\"ResponseType\":, \"Utterance\":\"} \
            Take the previous conversation into account %s").replace("[ ]+"," ").strip()
    
}        




#op = getChatGPTResponse(USRPROMPT+"\nConversation: "+conversation)


def getChatGPTResponse(userprompt, sysprompt="",temperature=0.7):
    time.sleep(2)
    if sysprompt!="":
        completion = openai.ChatCompletion.create(
        model = CHATGPT_MODEL,

            messages = [ # Change the prompt parameter to the messages parameter
                    {'role': 'system', 'content': sysprompt},
                    {'role': 'user', 'content': userprompt},
                ],
                temperature = temperature
        )
    else:
         completion = openai.ChatCompletion.create(
         model = CHATGPT_MODEL,

             messages = [ # Change the prompt parameter to the messages parameter
                     {'role': 'user', 'content': userprompt},
                 ],
                 temperature = temperature
         )

    return completion['choices'][0]['message']['content']



def getClientResponse(persona, issue_description, pconv):
    
    CLIENT_SYS=TEMPLATES["client_sys"].replace("[ ]+"," ")
            #with the \
        #following persona "+persona
        
    CLIENT_USR=TEMPLATES["client_usr"].replace("[ ]+"," ") % (issue_description, str(DBCAct_Client), pconv)
            
    
    return getChatGPTResponse(CLIENT_USR, CLIENT_SYS)


def getCounselorResponse(issue_solution, pconv):
    
    COUNSELOR_SYS= TEMPLATES["counselor_sys"]
    COUNSELOR_USR= TEMPLATES["counselor_usr"]% (issue_solution, str(DBCAct_Counselor), pconv)
            
    
    return getChatGPTResponse(COUNSELOR_USR, COUNSELOR_SYS)


