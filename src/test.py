import pandas as pd
import torch

from datasets import load_dataset
from qlora_finetune_phi3.configs.configuration import ExpConfig, bnb_config
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

dataset = load_dataset(ExpConfig.dataset_name)

dialogues = dataset['test'][0:10]['dialogue']
human_baseline_summaries = dataset['test'][0:10]['summary']

original_model_summaries = []

#Original Model
device_map = {"": 0}
original_model = AutoModelForCausalLM.from_pretrained(
    ExpConfig.model_name,
    device_map=device_map,
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_auth_token=True
).eval()

original_model_tokenizer = AutoTokenizer.from_pretrained(
    ExpConfig.model_name,
    trust_remote_code=True,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
    use_fast=False,   
)
original_model_tokenizer.pad_token = original_model_tokenizer.eos_token



for idx, dialogue in tqdm(enumerate(dialogues), total=len(dialogues)):
    human_baseline_output = human_baseline_summaries[idx]
    prompt = f"Instruct: Summarize the following conversation.\n{dialogue}\nOutput:\n"
    
    with torch.no_grad():
        original_model_res = original_model.generate(
            original_model_tokenizer(prompt, return_tensors="pt", max_length=1024).input_ids.to("cuda:0"),
        )
    
    gen_sent = original_model_tokenizer.batch_decode(
        original_model_res
    )[0]
    original_model_summaries.append(gen_sent)

df = pd.DataFrame(columns=['original_model_summary'], data=original_model_summaries)
df.to_csv("original_model_summaries.csv", index=False)