import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import json
from collections import Counter
from torch import cuda, bfloat16


def adjust_probabilities(original_probs, sample_prompts):
    """ Adjusts output probabilities based on relative frequency and marginal probabilities
    
    :param original_probs: tensor of probabilities from one inference prompt
    :param sample_prompts: list of sample prompts to calculate marginal probabilities

    :return: new output probabilities
    """
    # Tokenize the sample prompts
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(sample_prompts, return_tensors="pt", padding=True).to(model.device)

    # Generate logits from the model and convert to probabilities
    with torch.no_grad():
        logits = model(**inputs).logits
        vocab_size = logits.shape[-1]
    probs = torch.softmax(logits, dim=-1)

    # Calculate marginal probabilities 
    token_prob_sum = probs.sum(dim=(0, 1))
    total_positions = probs.shape[0] * probs.shape[1]
    marginal_probs = token_prob_sum / total_positions 

    # Obtain frequencies for words in the model tokenizer
    frequency_tensor = torch.ones(vocab_size) * 1.0/total_tokens
    for token_id in range(vocab_size):
        token = tokenizer.convert_ids_to_tokens(token_id).lstrip('▁')
        if token in frequency_dict:
            frequency_tensor[token_id] = frequency_dict[token]
    frequency_tensor = frequency_tensor.unsqueeze(0).unsqueeze(0).to(model.device)

    adjusted_probs = original_probs * frequency_tensor / marginal_probs

    return adjusted_probs


def get_new_prediction(prompt, target):
    """ 
    :param prompt: the prompt to run inference on
    :param target: torch tensor of the encoded next word token of the prompt

    :return: next predicted word after adjusting output probabilities
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits
    
    original_probs = torch.softmax(logits, dim=-1)
    adjusted_probs = adjust_probabilities(original_probs, sample_prompts)

    pred = tokenizer.decode(torch.argmax(adjusted_probs, dim=-1)[0][-1])
    original_loss = get_prob_loss(original_probs, target)
    adjusted_loss = get_prob_loss(adjusted_probs, target)

    return pred, original_loss, adjusted_loss


def get_prob_loss(probs, target):

    # get last row of predictions (next word after full prompt)
    logits = torch.log(probs[:, -1, :]) 
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target)

    return loss

def get_accuracies():
    pass

def initialize_model(model_name="meta-llama/Llama-2-7b-hf"):
    """Creates tokenizer and model if downloaded files are not available"""

    access_token = os.environ.get('hf_access_token', None)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": 0},
        quantization_config=bnb_config,
        token=access_token
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    return tokenizer, model


with open('token_counts.json', 'r', encoding='utf-8') as f:
    frequency_dict = json.load(f)
    total_tokens = sum(frequency_dict.values())
    frequency_dict = {token: freq / total_tokens for token, freq in frequency_dict.items()}

with open("prompts_and_labels.pkl", "rb") as fp:  
    prompts_and_labels = pickle.load(fp)
    sample_prompts = [row[0] for row in prompts_and_labels]

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

tokenizer = AutoTokenizer.from_pretrained('./tokenizer_files')
model = AutoModelForCausalLM.from_pretrained('./model_files')

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

prompt = 'hey what is going'
# print(f'Original prediction: {pipeline(prompt, max_new_tokens=1)[0]["generated_text"]}')
# print(f'New prediction: {prompt} {get_new_prediction(prompt)}')

print(get_new_prediction(prompt, torch.tensor([373], dtype=torch.long).to(model.device)))