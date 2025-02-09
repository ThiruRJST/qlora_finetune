from datasets import load_dataset
from huggingface_hub import interpreter_login
from qlora_finetune_phi3.configs.configuration import (
    ExpConfig,
    bnb_config
)
from qlora_finetune_phi3 import logger

from transformers import AutoTokenizer, AutoModelForCausalLM

#login hugingface hub
interpreter_login()

#Original Model
device_map = {"": 0}
original_model = AutoModelForCausalLM.from_pretrained(
    ExpConfig.model_name,
    device_map=device_map,
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_auth_token=True
)

original_model_tokenizer = AutoTokenizer.from_pretrained(
    ExpConfig.model_name,
    trust_remote_code=True,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
    use_fast=False,   
)
original_model_tokenizer.pad_token = original_model_tokenizer.eos_token

