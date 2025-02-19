This repository contains the code and data shared under Academic Free License
 for our COLING 2025 paper:
<br>
<b><i>PIRsuader: A Persuasive Chatbot for Mitigating Psychological Insulin
Resistance in Type-2 Diabetic Patients</i></b> 
<hr>
A demo of PIRsuader setup on NAO robot by Mouad Hakam is <a href="nao_demo_short.mp4">here</a>

and a short video presentation of the paper is <a href="https://drive.google.com/file/d/1tX0cb5tfGgIjcol6yZVWEyp2pnRAvQqd/view?usp=sharing">here</a>.
<hr>
<b>Setup</b>:
The shared code runs successfully with the following library versions on <b>Python 3.11.4</b> 

openai==0.28.0 <br>
nltk==3.8.1<br>
pytorch-transformers==1.2.0<br>
torch==2.0.1<br>
torchvision==0.15.2<br>
trl==0.9.6<br>
sentence-transformers==2.2.2<br>
transformers==4.39.3<br>
wandb==0.15.11<br>
accelerate==0.28.0<br>
peft==0.5.0<br>

The pip list of our environment is <a href="pip_list.txt">included</a>.
<hr>
<b>How to run</b>:
Generally speaking, we provide each "executable" with a configuration file where you need to specify details such as 
OPENAI key or hugging-face model names.
The three main executables will be related to:
<ol>
 <li>Conversation Generation (with or without predicted dialog acts--<i>convgen</i> directory)</li>
 <li>Dialog act prediction model (initial model--<i>dact</i> directory, further fine-tuning using Reinforcement Learning--<i>RL</i> directory)</li>
 <li>Reward modeling (--CAR and PBR subdirectories in <i>rewards</i> directory)</li>
</ol>

Please find the details in "README"s for each directory.




