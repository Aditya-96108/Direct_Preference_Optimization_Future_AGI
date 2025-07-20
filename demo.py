import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

base_model_name = "EleutherAI/gpt-neo-1.3B"
model_path = "models"  

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
    return numbers[-1] if numbers else "Not found"

def demo_inference(question):
    prompt = f"{question}\nLet's think step by step.\n"
    response = generate_response(model, tokenizer, prompt)
    steps = [s.strip() for s in response.split("\n") if s.strip()]
    output = "Steps:\n" + "\n".join(steps)
    final = extract_final_answer(response)
    output += f"\n\nFinal Answer: \\boxed{{{final}}}"
    return output

iface = gr.Interface(
    fn=demo_inference,
    inputs=gr.Textbox(label="Math Question"),
    outputs=gr.Textbox(label="Reasoning & Answer"),
    title="Stepwise DPO Model Demo",
    description="Enter a GSM8K-style question to see step-by-step reasoning from the trained model."
)

iface.launch()