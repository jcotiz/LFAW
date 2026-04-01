from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
from dotenv import load_dotenv
import torch
import time
import os

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print("Loading model in INT4...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=quantization_config,
    device_map="cuda",
    attn_implementation="sdpa"
)



tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

MAX_CONTEXT_TOKENS = 256

prompt = "Artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
input_ids = inputs["input_ids"]

NUM_TOKENS = 50

# ── Warmup sin compile ─────────────────────────────────────────
print("Warming up...")
warmup_ids = tokenizer("Hello", return_tensors="pt").to("cuda")["input_ids"]
with torch.no_grad():
    model(warmup_ids, use_cache=True)
print("Warmup done, measuring...")

# ── Loop manual ────────────────────────────────────────────────
past_key_values = None
print(type(past_key_values))
print(dir(past_key_values))
current_ids = input_ids
generated_tokens = []
token_times = []

print(f"Generating {NUM_TOKENS} tokens...")
start = time.time()

with torch.no_grad():
    for i in range(NUM_TOKENS):
        t0 = time.time()

        outputs = model(
            input_ids=current_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_tokens.append(next_token.item())
        past_key_values = outputs.past_key_values

        # ── Sliding window ──────────────────────────────────
        cache_length = past_key_values.get_seq_length()
        if cache_length > MAX_CONTEXT_TOKENS:
            past_key_values.crop(MAX_CONTEXT_TOKENS)

        current_ids = next_token
        t1 = time.time()
        token_times.append(t1 - t0)

        if next_token.item() == tokenizer.eos_token_id:
            break

end = time.time()

elapsed = end - start
speed = len(generated_tokens) / elapsed
vram_used = torch.cuda.memory_allocated() / 1024**3
vram_reserved = torch.cuda.memory_reserved() / 1024**3
avg_time = sum(token_times) / len(token_times)
first_token = token_times[0]
last_token = token_times[-1]

print(f"\n=== MANUAL LOOP + SLIDING WINDOW ===")
print(f"Generated tokens:  {len(generated_tokens)}")
print(f"Total time:        {elapsed:.2f} seconds")
print(f"Speed:             {speed:.2f} tokens/sec")
print(f"VRAM allocated:    {vram_used:.2f} GB")
print(f"VRAM reserved:     {vram_reserved:.2f} GB")
print(f"\n── Time per token ──")
print(f"First token:       {first_token:.3f} s")
print(f"Last token:        {last_token:.3f} s")
print(f"Average:           {avg_time:.3f} s")
print(f"Constant?:         {'✅ KV Cache working' if abs(first_token - last_token) < 0.5 else '❌ check cache'}")
print(f"\nGenerated text:")
output_ids = torch.cat(
    [input_ids, torch.tensor([generated_tokens]).to("cuda")], dim=1
)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))