from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from dotenv import load_dotenv
import torch
import time
import os

load_dotenv()

login(token=os.getenv("HF_TOKEN"))

print("Loading model from cache...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    dtype=torch.float16,
    device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# ── Test prompt ────────────────────────────────────────────────
prompt = "Artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

print(f"\nPrompt: '{prompt}'")
print(f"Input tokens: {inputs['input_ids'].shape[1]}")

# ── Measurement ────────────────────────────────────────────────
NUM_TOKENS = 2  # how many tokens to generate

print(f"\nGenerating {NUM_TOKENS} tokens...")
start = time.time()

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=NUM_TOKENS,
        do_sample=False,       # greedy — deterministic
        temperature=None,
        top_p=None,
    )

end = time.time()

# ── Results ────────────────────────────────────────────────────
generated_tokens = output.shape[1] - inputs['input_ids'].shape[1]
elapsed = end - start
speed = generated_tokens / elapsed

print(f"\n=== RESULTS ===")
print(f"Generated tokens:  {generated_tokens}")
print(f"Total time:        {elapsed:.2f} seconds")
print(f"Speed:             {speed:.2f} tokens/sec")
print(f"\nGenerated text:")
print(tokenizer.decode(output[0], skip_special_tokens=True))