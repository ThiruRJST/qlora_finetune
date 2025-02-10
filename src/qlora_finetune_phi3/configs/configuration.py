import time
import torch

from dataclasses import dataclass

from transformers import (
    BitsAndBytesConfig,
    GenerationConfig,
    TrainingArguments
)

from peft import LoraConfig


compute_type = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_type,
    bnb_4bit_use_double_quant=False
)
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

output_dir = f"peft-dialogue-summary-training-{str(int(time.time()))}"
peft_training_args = TrainingArguments(
    output_dir=output_dir,
    warmup_steps=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=1000,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",
    logging_steps=25,
    logging_dir="qlora-logs",
    save_strategy="steps",
    save_steps=25,
    evaluation_strategy="steps",
    eval_steps=25,
    do_eval=True,
    gradient_checkpointing=True,
    report_to="none",
    overwrite_output_dir="True",
    group_by_length=True,
)


@dataclass
class ExpConfig: #
    dataset_name: str = "neil-code/dialogsum-test"
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    compute_type: torch.dtype = compute_type
    bnb_config: BitsAndBytesConfig = bnb_config
    lora_config: LoraConfig = lora_config
    peft_training_args: TrainingArguments = peft_training_args