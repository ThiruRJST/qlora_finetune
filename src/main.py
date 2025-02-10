import os
import transformers

from datasets import load_dataset
from huggingface_hub import interpreter_login
from peft import (
    get_peft_model, 
    prepare_model_for_kbit_training
)
from qlora_finetune_phi3.configs.configuration import ExpConfig
from qlora_finetune_phi3 import logger
from qlora_finetune_phi3.data.data_sampler import (
    get_max_length,
    preprocess_dataset
)
from qlora_finetune_phi3.trainer import print_trainable_parameters
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer

#login hugingface hub
os.environ["HF_TOKEN"] = "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"



if __name__ == "__main__":
    #Original Model
    logger.info(f"Loading original model: {ExpConfig.model_name}")
    device_map = {"": 0}
    original_model = AutoModelForCausalLM.from_pretrained(
        ExpConfig.model_name,
        device_map=device_map,
        quantization_config=ExpConfig.bnb_config,
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
    logger.info(f"Original model and tokenizer loaded successfully: {ExpConfig.model_name}")

    #dataset
    logger.info(f"Preprocessing and splitting dataset: {ExpConfig.dataset_name}")
    dataset = load_dataset(ExpConfig.dataset_name)
    max_length = get_max_length(original_model)

    #preprocess dataset
    train_dataset = preprocess_dataset(
        tokenizer=original_model_tokenizer,
        max_len=max_length,
        dataset=dataset['train'],
        seed=2024
    )

    eval_dataset = preprocess_dataset(
        tokenizer=original_model_tokenizer,
        max_len=max_length,
        dataset=dataset['validation'],
        seed=2024
    )
    logger.info(f"Dataset preprocessed and split successfully: {ExpConfig.dataset_name}")

    logger.info("Preparing model with qlora...")
    original_model = prepare_model_for_kbit_training(
        original_model
    )
    original_model.gradient_checkpointing_enable()
    peft_model = get_peft_model(
        original_model,
        ExpConfig.lora_config
    )
    print(print_trainable_parameters(peft_model))
    logger.info("Model prepared with qlora successfully")

    logger.info("Iniating the training...")
    peft_model.config.use_cache = False
    peft_trainer = Trainer(
        model=peft_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=ExpConfig.peft_training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer=original_model_tokenizer,
            mlm=False
        )
    )
    peft_trainer.train()