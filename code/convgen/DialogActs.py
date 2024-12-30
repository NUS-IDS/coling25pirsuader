#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 13:36:04 2024

@author: sdas
"""

DBCAct = {
         #both client and counselor
         'greet': 'Please say hello or chat randomly.',
         'thank': 'Thank',
         'general_agreement': 'Convey agreement to the mentioned information',
         'neutral_to_information': 'Neutral reaction the mentioned information',
         'counter_information': 'Please counter the information provided.',
         'chitchat': 'Engage in chitchat on general topics',
         'acknowledge':'provide acknowledgement',
         'closing': 'Provide remarks for ending the conversation',
         'end_conversation': 'say goodbye and wrap up conversation',
         'request_information': 'Ask for new factual information',
         'provide_information': 'Please provide information in response to an inquiry not related to diabetes.',
         'express_interest': 'Express the willingness to learn or hear more about the subject brought up by the speaker; demonstrate curiosity.',

         #counselor
         'provide_insulin_information': 'Provide information in response to a question on insulin or diabetes.', 
         'ask_concerns': 'Ask if have other concerns related to insulin', 
         'propose': 'Please suggest trying insulin', 
         'personal_related_inquiry': 'Ask about some personal information related to the context',
         'task_related_inquiry': 'Ask about desire to try insulin for better diabetes control',
         'logical_appeal': 'Provide logical reasoning to why they should try insulin',
         'emotion_appeal': 'Emotionally appeal to why they should try insulin',
         'credibility_appeal': 'Use research studies to convince why they should try insulin',
         'ask_about_consequence': 'Ask about the result of the described action or situation',
         'ask_about_antecedent': 'Ask about the reason or cause of the described state or event.',
         'ask_for_confirmation': 'Confirm the agreement to try insulin',
         'suggest_a_solution': 'Provide a specific solution to a problem in a form of a question',
         'suggest_a_reason': 'Suggest a specific reason or cause of the event or state described by the speaker in a form of a question',
         'express_concern': 'Express anxiety or worry about the subject brought up by the speaker.',
         'offer_relief': 'Reassure the speaker who is anxious or distressed',
         'sympathize': 'express feelings of pity and sorrow for the speaker\'s hardships',
         'support': 'Offer approval, comfort, or encouragement to the speaker, demonstrate an interest in and concern for the speaker\'s success.',
         'amplify_excitement': 'Reinforce the speaker\'s feeling of excitement.',
         'motivate': 'Encourage the speaker to move onward',
         'compliment': 'Encourage the speaker on a job well done',

       ##Client-only
         'affirm': 'Please give an affirmative response to an ask_for_confirmation.', #client
         'deny_to_try': 'Please respond negatively to trying insulin',
         'agree_to_try': 'Please respond positively to trying insulin',
         #######3
  }

def getClosest(inp):

    best_match=""
    best_mval = 0
    for key in DBCAct.keys():
        if key in inp:
            if len(key)> best_mval:
                best_mval= len(key)
                best_match = key
        elif inp in key:
            if len(inp) > best_mval:
                best_mval = len(inp)
                best_match = key

    return best_match


DBCAct_Counselor = {
    #both client and counselor
    'greet': 'Please say hello or chat randomly.',
    'thank': 'Thank',
    'general_agreement': 'Convey agreement to the mentioned information',
    'neutral_to_information': 'Neutral reaction the mentioned information',
    'counter_information': 'Please counter the information provided.',
    'chitchat': 'Engage in chitchat on general topics',
    'acknowledge':'provide acknowledgement',
    'closing': 'Provide remarks for ending the conversation',
    'end_conversation': 'say goodbye and wrap up conversation',
    'request_information': 'Ask for new factual information',
    'provide_information': 'Please provide information in response to an inquiry not related to diabetes.',
    'express_interest': 'Express the willingness to learn or hear more about the subject brought up by the speaker; demonstrate curiosity.',

    #counselor
    'provide_insulin_information': 'Provide information in response to a question on insulin or diabetes.', 
    'ask_concerns': 'Ask if have other concerns related to insulin', 
    'propose': 'Please suggest trying insulin', 
    'personal_related_inquiry': 'Ask about some personal information related to the context',
    'task_related_inquiry': 'Ask about desire to try insulin for better diabetes control',
    'logical_appeal': 'Provide logical reasoning to why they should try insulin',
    'emotion_appeal': 'Emotionally appeal to why they should try insulin',
    'credibility_appeal': 'Use research studies to convince why they should try insulin',
    'ask_about_consequence': 'Ask about the result of the described action or situation',
    'ask_about_antecedent': 'Ask about the reason or cause of the described state or event.',
    'ask_for_confirmation': 'Confirm the agreement to try insulin',
    'suggest_a_solution': 'Provide a specific solution to a problem in a form of a question',
    'suggest_a_reason': 'Suggest a specific reason or cause of the event or state described by the speaker in a form of a question',
    'express_concern': 'Express anxiety or worry about the subject brought up by the speaker.',
    'offer_relief': 'Reassure the speaker who is anxious or distressed',
    'sympathize': 'express feelings of pity and sorrow for the speaker\'s hardships',
    'support': 'Offer approval, comfort, or encouragement to the speaker, demonstrate an interest in and concern for the speaker\'s success.',
    'amplify_excitement': 'Reinforce the speaker\'s feeling of excitement.',
    'motivate': 'Encourage the speaker to move onward',
    'compliment': 'Encourage the speaker on a job well done',
     }


DBCAct_Client = {
    #both client and counselor
    'greet': 'Please say hello or chat randomly.',
    'thank': 'Thank',
    'general_agreement': 'Convey agreement to the mentioned information',
    'neutral_to_information': 'Neutral reaction the mentioned information',
    'counter_information': 'Please counter the information provided.',
    'chitchat': 'Engage in chitchat on general topics',
    'acknowledge':'provide acknowledgement',
    'closing': 'Provide remarks for ending the conversation',
    'end_conversation': 'say goodbye and wrap up conversation',
    'request_information': 'Ask for new factual information',
    'provide_information': 'Please provide information in response to an inquiry not related to diabetes.',
    'express_interest': 'Express the willingness to learn or hear more about the subject brought up by the speaker; demonstrate curiosity.',

   
  ##Client-only
    'affirm': 'Please give an affirmative response to an ask_for_confirmation.', #client
    'deny_to_try': 'Please respond negatively to trying insulin',
    'agree_to_try': 'Please respond positively to trying insulin',
    #######3 
         }

for key in DBCAct_Counselor:
    if key not in DBCAct_Client:
        print (key)
