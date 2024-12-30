#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:25:53 2024

@author: sdas
"""


model_id="google/flan-t5-large"
max_source_length=512
max_target_length=16 #128
model_out_path="/tmp/ft5_counselor_dacts"

train_csv="../../data/dacts/train.csv"
test_csv="../../data/dacts/test.csv"

