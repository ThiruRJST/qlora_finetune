import torch

from dataclasses import dataclass

from transformers import (
    BitsAndBytesConfig,
    GenerationConfig,
    TrainingArguments
)

compute_type = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_type,
    bnb_4bit_use_double_quant=False
)

@dataclass
class ExpConfig:
    dataset_name: str = "neil-code/dialogsum-test"
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"