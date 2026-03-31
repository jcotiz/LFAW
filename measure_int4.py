from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
from dotenv import load_dotenv
import torch
import time
import os

load_dotenv()

login(token=os.getenv("HF_TOKEN"))

# ── INT8 configuration ─────────────────────────────────────────
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print("Loading model in INT8...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=quantization_config,
    device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# ── Same test as before for comparison ────────────────────────
prompt = "Artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

NUM_TOKENS = 50

print(f"Generating {NUM_TOKENS} tokens in INT8...")
start = time.time()

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=NUM_TOKENS,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

end = time.time()

generated_tokens = output.shape[1] - inputs['input_ids'].shape[1]
elapsed = end - start
speed = generated_tokens / elapsed

print(f"\n=== INT8 RESULTS ===")
print(f"Generated tokens:  {generated_tokens}")
print(f"Total time:        {elapsed:.2f} seconds")
print(f"Speed:             {speed:.2f} tokens/sec")
print(f"\nGenerated text:")
print(tokenizer.decode(output[0], skip_special_tokens=True))