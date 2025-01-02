This directory contains the code for training the two reward models CAR and PBR described in the paper
using the provided <a href="../../data/forrl">training data</a>.

For PBR, this is minimally modified code from TRL's <a href="https://huggingface.co/docs/trl/en/reward_trainer">reward trainer</a>. 
Specify the initial <a href="../dact">model</a> and  paths by directly editing the code.

For CAR, this is training a straightforward RoBERTa-based classifier.
