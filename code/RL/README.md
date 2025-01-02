

The initial dialog act model (trained using "silver" data) can be further fine-tuned using rewards through 
reinforcement learning. The relevant training data for the two rewards discussed in the paper
is <a href="../../data/forrl">provided</a>. Use the code in <a href="../rewards">rewards</a>, to first learn reward models and specify their paths
with the data file locations in \*Config files after which use the code from trl_trainer_* to obtain
the final model.
