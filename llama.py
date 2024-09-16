import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import json
from collections import Counter
from torch import cuda, bfloat16
from helper_functions import *

def adjust_probabilities(original_probs, prompts):
    """ Adjusts output probabilities based on relative frequency and marginal probabilities
    
    :param original_probs: tensor of probabilities from one inference prompt
    :param prompts: list of sample prompts to calculate marginal probabilities

    :return: new output probabilities
    """
    # Tokenize the sample prompts
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # Generate logits from the model and convert to probabilities
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

    # Calculate marginal probabilities 
    token_prob_sum = probs.sum(dim=(0, 1))
    marginal_probs = token_prob_sum / probs.numel()

    # Obtain frequencies for words in the model tokenizer
    vocab_size = logits.shape[-1]
    token_ids = torch.arange(vocab_size, device=model.device)
    tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())
    
    # Set frequencies based on the frequency_dict
    frequency_tensor = torch.ones(vocab_size, device=model.device) * (1.0 / total_tokens)
    token_freqs = [frequency_dict.get(token.lstrip('▁'), 1.0 / total_tokens) for token in tokens]
    frequency_tensor = torch.tensor(token_freqs, device=model.device).unsqueeze(0).unsqueeze(0)

    adjusted_probs = original_probs * frequency_tensor / marginal_probs

    return adjusted_probs


def get_new_prediction(prompts):
    """ 
    :param prompts: batch of prompts to run inference on

    :return: [new next word predictions, new probabilities]
    """

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits
    
    original_probs = torch.softmax(logits, dim=-1)
    adjusted_probs = adjust_probabilities(original_probs, prompts)

    preds_ids = torch.argmax(adjusted_probs, dim=-1)
    preds = [tokenizer.decode(pred_id) for pred_id in preds_ids[:, -1]]
    
    return preds, adjusted_probs

def get_loss(probs, target, loss='accuracy'):
    logits = torch.log(probs)[:, -1, :] 

    if loss == 'cross_entropy':
        target = torch.tensor(target, dtype=torch.long).unsqueeze(0).to(model.device)
        result = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target).item()
    elif loss == 'accuracy':
        predicted_token_id = torch.argmax(logits, dim=-1).item()
        result = int(predicted_token_id == target)

    return result


tokenizer = AutoTokenizer.from_pretrained('./tokenizer_files')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained('./model_files')

with open('token_counts.json', 'r', encoding='utf-8') as f:
    frequency_dict = json.load(f)
    total_tokens = sum(frequency_dict.values())
    frequency_dict = {token: freq / total_tokens for token, freq in frequency_dict.items()}

with open("prompts_and_labels.pkl", "rb") as fp:  
    prompts_and_labels = pickle.load(fp)
    sample_prompts = [row[0] for row in prompts_and_labels]
    sample_labels = [tokenizer.decode(torch.tensor([row[1]], dtype=torch.long)) for row in prompts_and_labels]

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

# original_preds = [res[0]['generated_text'].split()[-1] for res in pipeline(sample_prompts[:3], max_new_tokens=1)]

prompt = 'the banana is'
# print(f'Original prediction: {pipeline(prompt, max_new_tokens=1)[0]["generated_text"]}')
# print(f'New prediction: {prompt} {get_new_prediction(prompt)[0]}')

prompts = ['i want to', 'hey what is going']
print(get_new_prediction(prompts)[0])