from llama_cpp import Llama
import time
import os
from dotenv import load_dotenv

# CONFIG 
load_dotenv()
MODEL_PATH = os.getenv("PATH_PM3")  # adjust the path if needed
NUM_TOKENS = 50
MAX_CONTEXT = 256



print("Loading Phi-3 GGUF (Q4_K_M)...")

llm = Llama(
    model_path= MODEL_PATH,
    n_ctx=MAX_CONTEXT,        # equivalent to your MAX_CONTEXT_TOKENS
    n_gpu_layers=-1,          # -1 = all layers on GPU (equivalent to device_map="auto")
    n_batch=256,              # same as context - less overhead
    n_ubatch=256,
    n_threads=4,              # CPU threads for layers that dont fit in VRAM
    use_mmap=True,            # efficient loading from disk
    verbose=False,
)

prompt = "<|user|>\nArtificial intelligence is<|end|>\n<|assistant|>"

#  WARMUP 
print("Warming up...")
_ = llm(
    "<|user|>\nHello<|end|>\n<|assistant|>",
    max_tokens=5,
    echo=False,
)
print("Warmup done, measuring...")

# BENCHMARK 
token_times = []
generated_tokens = []

print(f"Generating {NUM_TOKENS} tokens...")
start = time.time()

# Stream token by token (equivalent to your manual loop)
stream = llm(
    prompt,
    max_tokens=NUM_TOKENS,
    stream=True,
    echo=False,
    stop=["<|end|>", "<|endoftext|>"],
)

full_text = ""
for i, chunk in enumerate(stream):
    t0 = time.time()
    token_text = chunk["choices"][0]["text"]
    full_text += token_text
    generated_tokens.append(token_text)
    t1 = time.time()
    token_times.append(t1 - t0)

end = time.time()

#  METRICS  
elapsed = end - start
speed = len(generated_tokens) / elapsed
avg_time = sum(token_times) / len(token_times) if token_times else 0
first_token = token_times[0] if token_times else 0
last_token = token_times[-1] if token_times else 0
steady_times = token_times[5:] if len(token_times) > 5 else token_times
variation = (max(steady_times) - min(steady_times)) if steady_times else 0

# VRAM via llama_cpp metadata
model_meta = llm.metadata if hasattr(llm, "metadata") else {}

print(f"\n=== LLAMA.CPP + GGUF Q4_K_M ===")
print(f"Generated tokens:  {len(generated_tokens)}")
print(f"Total time:        {elapsed:.2f} seconds")
print(f"Speed:             {speed:.2f} tokens/sec")

print(f"\n── Time per token ──")
print(f"First token:       {first_token:.3f} s")
print(f"Last token:        {last_token:.3f} s")
print(f"Average:           {avg_time:.3f} s")
print(f"Stable?:           {'✅' if variation < 0.1 else '❌'} (variation: {variation:.3f}s)")

print(f"\nGenerated text:")
print(prompt + full_text)

#  DIRECT COMPARISON 
print(f"\n── Comparison reference ──")
print(f"Previous code:      ~4.62 tok/s  (BitsAndBytes NF4 + float32)")
print(f"This result:        {speed:.1f} tok/s  (GGUF Q4_K_M + llama.cpp CUDA)")
delta = ((speed - 4.62) / 4.62) * 100
print(f"Difference:         {delta:+.1f}%")