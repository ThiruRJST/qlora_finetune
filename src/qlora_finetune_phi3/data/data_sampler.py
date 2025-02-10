from functools import partial
from transformers import AutoTokenizer

def create_prompt_format(sample: dict) -> dict:
    """
    Format various fields of the sample('instruction', 'output')
    Then concatenate them using two newline characters
    :param sample: Sample dictionary

    Args:
        sample (_type_): _description_
    """
    
    INTRO_BLOB = "Below is an instruction that describes a task. Write a response that appropirately completes the request."
    INSTRUCTION_BLOB = "### Instruct: Summarize the below conversation."
    RESPONSE_BLOB = "### Output:"
    END_BLOB = "### End"
    
    intro_blob = f"\n{INTRO_BLOB}"
    instruction = f"{INSTRUCTION_BLOB}"
    input_context = f"{sample['dialogue']}" if sample['dialogue'] else None
    response = f"{RESPONSE_BLOB}\n{sample['summary']}"
    end = f"{END_BLOB}"
    
    parts = [part for part in [intro_blob, instruction, input_context, response, end] if part]
    
    formatted_prompt = "\n\n".join(parts)
    sample['text'] = formatted_prompt
    
    return sample

def get_max_length(model):
    model_config = model.config
    max_len = None
    
    for length_settings in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_len = getattr(model_config, length_settings, None)        
        if max_len:
            break
        
    if not max_len:
        max_len = 1024
    return max_len

def preprocess_batch(batch, tokenizer, max_len)        :
    """Tokenizing a batch

    Args:
        batch (_type_): A batch data with batch size: b
        tokenizer (_type_): Tokenizer object to tokenize the data
        max_len (_type_): Maximum length of the tokenized data
    """
    
    return tokenizer(
        batch['text'],
        max_length=max_len,
        truncation=True,
    )

def preprocess_dataset(tokenizer: AutoTokenizer, max_len: int, seed: int, dataset):
    """Format and tokenize the dataset for training"""    
    dataset = dataset.map(create_prompt_format)
    
    _preprocess_fn = partial(
        preprocess_batch,
        max_len=max_len,
        tokenizer=tokenizer
    )
    
    dataset = dataset.map(
        _preprocess_fn,
        batched=True,
        remove_columns=["id", "topic", "dialogue", "summary"]
    )
    
    dataset = dataset.filter(
        lambda sample: len(sample["input_ids"]) < max_len
    )
    
    dataset = dataset.shuffle(seed=seed)
    
    return dataset