This directory contains the code for training the two reward models CAR and PBR described in the paper
using the provided <a href="../../data/forrl">training data</a>.

For PBR, this is minimally modified code from Text Reinforcement Library's <a href="https://huggingface.co/docs/trl/en/reward_trainer">reward trainer</a>
using preference pairs data whereas for CAR, this is a straightforward RoBERTa-based five-class classifier.

Specify relevant model names and data paths by directly editing the code.
