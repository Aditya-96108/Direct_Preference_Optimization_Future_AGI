import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re

base_model_name = "EleutherAI/gpt-neo-1.3B"
model_path = "models"  # Fixed: Flat unzip to 'models/'

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(base_model, model_path)

def generate_response(model, tokenizer, prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

def extract_final_answer(response):
    match = re.search(r"\\boxed\{(.*?)\}", response)
    if match: return match.group(1)
    numbers = re.findall(r"\d+", response)
    return numbers[-1] if numbers else None

question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
prompt = f"{question}\nLet's think step by step.\n"
response = generate_response(model, tokenizer, prompt)
final = extract_final_answer(response)
print(f"Response:\n{response}\nFinal: {final}")