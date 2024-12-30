# https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/flan-t5-samsum-summarization.ipynb
#####
#GSDAS mostly from above, changes to reward functions and settings for KL divergence problem according to the
##mentioned Github issues, see 
###
#https://github.com/huggingface/trl/issues/235
#https://github.com/huggingface/trl/issues/679
##############
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from random import randrange
from transformers import HfArgumentParser, pipeline, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed, AutoModelForSeq2SeqLMWithValueHead
from transformers import DataCollatorForSeq2Seq
import torch

from peft import LoraConfig
from tqdm import tqdm

from trl.core import LengthSampler
from PBRUtils import compute_reward, CONV_DELIMITER
import PBRConfig as pbrconfig
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize

import math
import sys
tqdm.pandas()

nltk.download("punkt")
model_id=pbrconfig.init_model_path
max_source_length=pbrconfig.max_source_length
max_target_length=pbrconfig.max_target_length
outdir=pbrconfig.model_out_dir
dataset = load_dataset('csv', data_files={'train': pbrconfig.train_data_file})

print(f"Train dataset size: {len(dataset['train'])}")
#print(f"Test dataset size: {len(dataset['test'])}")
trdata = dataset['train']

print (type(trdata))
print (len(trdata))



########## End Load dataset 

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=8, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "whether to use peft"})
    #GSDAS use_seq2seq: Optional[bool] = field(default=False, metadata={"help": "whether to use seq2seq models"})
    use_seq2seq: Optional[bool] = field(default=True, metadata={"help": "whether to use seq2seq models"})
    kl_penalty: Optional[str] = field(
        default="kl",
        metadata={
            "help": "kl penalty options: 'kl': model_logp - ref_logp,  'abs': abs(kl),  'mse': mean squared error mse(kl) and 'full': the actual kl for all tokens in the distribution"
        },
    )
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    seed: Optional[int] = field(default=0, metadata={"help": "the random seed"})
    use_score_scaling: Optional[bool] = field(default=False, metadata={"help": "Use score scaling"})
    use_score_norm: Optional[bool] = field(
        default=False, metadata={"help": "Use score normalization. Only applicable if use_score_scaling is True"}
    )
    score_clip: Optional[float] = field(default=None, metadata={"help": "Score clipping"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

config = PPOConfig(
    model_name=model_id,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    kl_penalty=script_args.kl_penalty,
    seed=script_args.seed,
    use_score_scaling=script_args.use_score_scaling,
    use_score_norm=script_args.use_score_norm,
    score_clip=script_args.score_clip,
)

# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

trl_model_class = (
    AutoModelForCausalLMWithValueHead if not script_args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead
)

# Now let's build the model, the reference model, and the tokenizer.
if not script_args.use_peft:
    ref_model = trl_model_class.from_pretrained(model_id)
    device_map = None
    peft_config = None
else:
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        bias="none",
        task_type="EMSTAG",
    )
    ref_model = None
    device_map = {"": 0}

#print (device_map)
model = trl_model_class.from_pretrained(
    config.model_name,
    device_map=device_map,
    peft_config=peft_config,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.add_tokens([CONV_DELIMITER])
def build_dataset(config, ds):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(sample):
        
        #sample["input_ids"] = tokenizer.encode(addPrompt(sample["source"]), padding=True, max_length=max_source_length, truncation=True) #for some reason the suggested padding="max_length" throws an error in sentiment_pipeline something about tensor length mismatch 
        #SOMEHOW ADDING PROMPT DOES NOT WORK WELL: GSDAS NOTE
        
        sample["input_ids"] = tokenizer.encode((sample["source"]), padding=True, max_length=max_source_length, truncation=True) #for some reason the suggested padding="max_length" throws an error in sentiment_pipeline something about tensor length mismatch 
 #       print ("DEBUG Inside build_dataset, source")
 #       print (sample["source"])
        sample["query"] = sample["source"] #tokenizer.decode(sample["input_ids"]), 
        ## Above fix by GSDAS just pass directly or may miss some
        ## tokens am using to post-process from input during reward computation
        ## need specific delimiters that we do not want tokenized
        
        
        
 #       print ("DEBUG query\n")
 #       print (sample["query"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds



# We retrieve the dataloader by calling the `build_dataset` function.
trdataset = build_dataset(config, trdata)
def collator(data):
#    print ()
#    print (type(data))
#    print (data[0])
#    print ("##########GSDAS#######, basically below just loops and collates, look up keys in first item and collate for all items in the list")
    return dict((key, [d[key] for d in data]) for key in data[0])


# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=trdataset, data_collator=collator)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.

#GSDAS https://github.com/huggingface/trl/issues/235
#https://github.com/huggingface/trl/issues/679
generation_kwargs = {
    "max_new_tokens": max_target_length,
    "min_length": 3, #-1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    #"eos_token_id": -1,
    "eos_token_id":  tokenizer.eos_token_id,
}

start=0
ne=pbrconfig.num_rlepochs
for e in range (0, ne):
    print ("Overall epoch# "+str(e))
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):

        start+=1
        query_tensors = batch["input_ids"]
        print (type(batch["query"]))
        #print (batch.keys())
        response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
        
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        
        texts = [(q, r) for q, r in zip(batch["query"], batch["response"])]
           

        #some dummy rewards
        rewards = [ torch.tensor((compute_reward(text[0])), \
                                 device=model.pretrained_model.device) for text in texts]
#    rewards = [ torch.tensor(1.0, device=model.pretrained_model.device) for text in texts]
        if start%10==0:
            print ("########")
            print ("Batch "+str(start))
            print (batch["query"][0])
            print (batch["response"][0])
            print (rewards)

    # Run PPO step
        #gsdas temp 
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        #gsdas temp 
        ppo_trainer.log_stats(stats, batch, rewards)




model.save_pretrained(outdir) 
tokenizer.save_pretrained(outdir) 
print ("Model written to "+outdir)




