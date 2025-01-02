Specify some model parameters in FlanTrainConfig and run the finetune script for training the initial dialog act model (using
supervised learning). The training data from <a href="../../data/dacts">dacts</a> is "silver"--
obtained by annotations from OpenAI model/zero-shot setting and not manually-annotated with dialog acts. We used 
a suitable prompt along with conversation snippets for learning a dialog act predictor with FlanT5-large (Google/Huggingface).
