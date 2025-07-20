from datasets import load_from_disk
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import traceback
import sys


data_path = "data"         
model_path = "models"      

print("🔵 Script started.")
print(f"📁 Dataset path: {data_path}")

try:
    test_data = load_from_disk(data_path)
    print(f"✅ Dataset loaded with {len(test_data)} examples.")
    if len(test_data) == 0:
        print("⚠️ Warning: Empty dataset - no generations will be performed.")
except Exception as e:
    print("❌ Error loading dataset:", e)
    traceback.print_exc(file=sys.stdout)
    sys.exit(1)


print("🔄 Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded successfully.")
except Exception as e:
    print("❌ Error loading tokenizer:", e)
    traceback.print_exc(file=sys.stdout)
    sys.exit(1)


print("🔄 Loading base model...")
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-neo-1.3B",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    print("✅ Base model loaded successfully.")
except Exception as e:
    print("❌ Error loading base model:", e)
    traceback.print_exc(file=sys.stdout)
    sys.exit(1)


print("🔄 Loading PEFT model...")
try:
    trained_model = PeftModel.from_pretrained(base_model, model_path)
    print("✅ PEFT model loaded successfully.")
except Exception as e:
    print("❌ Error loading PEFT model:", e)
    traceback.print_exc(file=sys.stdout)
    sys.exit(1)


def generate_response(model, tokenizer, prompt, max_new_tokens=50):
    try:
        print(f"➡️ Generating response for prompt: {prompt[:50]}...")
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True
        )
        print("✅ Generation complete.")
        return response
    except Exception as e:
        print("❌ Error during generation:", e)
        traceback.print_exc(file=sys.stdout)
        return "Generation failed."


def evaluate_model(model, tokenizer, data):
    for example in tqdm(data):
        prompt = example['prompt'] + "\nLet's think step by step.\n"
        print(f"\n🧪 Processing example: {example['prompt'][:50]}...")
        response = generate_response(model, tokenizer, prompt)
        print(f"📝 Prompt: {example['prompt']}\n💡 Response: {response}\n")


print("\n🚀 Trained Model Generations:")
evaluate_model(trained_model, tokenizer, test_data)


print("\n📄 Pre-computed Results:")
try:
    with open("results/results.txt", "r") as f:
        print(f.read())
except Exception as e:
    print("❌ Error reading results.txt:", e)
    traceback.print_exc(file=sys.stdout)
