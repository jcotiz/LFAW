from llama_cpp import Llama
import time
import os
from dotenv import load_dotenv


load_dotenv()
MODEL_PATH = os.getenv("PATH_QW7")  
NUM_TOKENS = 1024
MAX_CONTEXT = 4096

SYSTEM_PROMPT = "You are a Python expert. always gonna give the best practices and check the code 2 times before give me"

def build_prompt(user_input: str) -> str:
    return f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""


user_code = "que significa speed:.1f , el :.1f que se le coloca despues a una variable ? que funcion tiene ?"

PROMPT = build_prompt(user_code)
WARMUP_PROMPT = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
STOP_TOKENS = ["<|im_end|>", "<|endoftext|>"]

print("Loading Qwen2.5-Coder-7B GGUF (Q4_K_M)...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=MAX_CONTEXT,        
    n_gpu_layers=-1,          
    n_batch=512,              
    n_ubatch=512,
    n_threads=4,             
    use_mmap=True,            
    verbose=False,
)

# WARMUP
print("Warming up...")
_ = llm(
    WARMUP_PROMPT,
    max_tokens=5,
    echo=False,
)
print("Warmup done, measuring...")


token_times = []
generated_tokens = []
prev_time = None 

print(f"Generating {NUM_TOKENS} tokens...")
start = time.time()

stream = llm(
    PROMPT,
    max_tokens=NUM_TOKENS,
    stream=True,
    echo=False,
    stop=STOP_TOKENS,
)

full_text = ""
for i, chunk in enumerate(stream):
    now = time.time()

    token_text = chunk["choices"][0]["text"]
    full_text += token_text
    generated_tokens.append(token_text)

    # FIX: measure time BETWEEN tokens, not processing time of the chunk
    # The original code measured t1-t0 after the token arrived (near zero)
    # This measures the actual generation gap between consecutive tokens
    if prev_time is not None:
        token_times.append(now - prev_time)
    prev_time = now

end = time.time()

# METRICS
elapsed = end - start
speed = len(generated_tokens) / elapsed
avg_time = sum(token_times) / len(token_times) if token_times else 0
first_token = token_times[0] if token_times else 0
last_token = token_times[-1] if token_times else 0
steady_times = token_times[5:] if len(token_times) > 5 else token_times
variation = (max(steady_times) - min(steady_times)) if steady_times else 0

print(f"\n=== QWEN2.5-CODER-7B + GGUF Q4_K_M ===")
print(f"Generated tokens:  {len(generated_tokens)}")
print(f"Total time:        {elapsed:.2f} seconds")
print(f"Speed:             {speed:.2f} tokens/sec")

print(f"\n── Time per token ──")
print(f"First token:       {first_token:.3f} s")
print(f"Last token:        {last_token:.3f} s")
print(f"Average:           {avg_time:.3f} s")
print(f"Stable?:           {'✅' if variation < 0.1 else '❌'} (variation: {variation:.3f}s)")

print(f"\nGenerated text:")
print(full_text)

# COMPARISON vs Phi-3-mini baseline
print(f"\n── Comparison reference ──")
print(f"Phi-3-mini 3B Q4_K_M:   ~6.1 tok/s  (llama.cpp CPU, Xeon 2630v4)")
print(f"This result:             {speed:.1f} tok/s  (Qwen2.5-Coder 7B Q4_K_M)")
delta = ((speed - 6.1) / 6.1) * 100
print(f"Difference vs Phi-3:    {delta:+.1f}%")