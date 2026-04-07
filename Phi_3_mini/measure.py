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
    bnb_4bit_compute_dtype=torch.float32,  # GTX 1060 safe
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print("Loading Phi-3 in INT4...")

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"
)

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    trust_remote_code=True
)

device = next(model.parameters()).device

MAX_CONTEXT_TOKENS = 256

prompt = "<|user|>\nArtificial intelligence is<|end|>\n<|assistant|>"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)

NUM_TOKENS = 50

# Warmup
print("Warming up...")
warmup_ids = tokenizer(
    "<|user|>\nHello<|end|>\n<|assistant|>",
    return_tensors="pt"
)["input_ids"].to(device)

with torch.no_grad():
    model(warmup_ids, use_cache=True)

print("Warmup done, measuring...")

# KEY FUNCTION
def apply_sliding_window(past_key_values, max_len):
    # Case 1: DynamicCache (Llama, etc)
    if hasattr(past_key_values, "crop"):
        if past_key_values.get_seq_length() > max_len:
            past_key_values.crop(max_len)
        return past_key_values

    # Case 2: tuple (Phi-3, etc)
    new_past = []
    for k, v in past_key_values:
        # k, v: [batch, heads, seq_len, head_dim]
        if k.shape[2] > max_len:
            k = k[:, :, -max_len:, :]
            v = v[:, :, -max_len:, :]
        new_past.append((k, v))
    return tuple(new_past)


# Loop manual
past_key_values = None
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

        past_key_values = apply_sliding_window(
            outputs.past_key_values,
            MAX_CONTEXT_TOKENS
        )

        current_ids = next_token

        t1 = time.time()
        token_times.append(t1 - t0)

        if next_token.item() == tokenizer.eos_token_id:
            break

end = time.time()

elapsed = end - start
speed = len(generated_tokens) / elapsed

if torch.cuda.is_available():
    vram_used = torch.cuda.memory_allocated() / 1024**3
    vram_reserved = torch.cuda.memory_reserved() / 1024**3
else:
    vram_used = 0
    vram_reserved = 0

avg_time = sum(token_times) / len(token_times)
first_token = token_times[0]
last_token = token_times[-1]
steady_times = token_times[5:]
variation = max(steady_times) - min(steady_times)

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

print("Cache type:", type(outputs.past_key_values))
print("Stable?:", "✅" if variation < 0.1 else "❌")

print(f"\nGenerated text:")

output_ids = torch.cat(
    [input_ids, torch.tensor([generated_tokens]).to(device)],
    dim=1
)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))