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
    device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# ── torch.compile ──────────────────────────────────────────────
# Toca específicamente el forward() de cada una de las 32 capas:
#   - Linear Q, K, V, O  (atención)
#   - FFN gate, up, down (feed forward)
#   - RMSNorm
# Fusiona esas operaciones en kernels CUDA únicos
# ADVERTENCIA: el primer generate() tardará 1-3 min compilando
# Los siguientes serán rápidos
print("Compiling model (first run will be slow)...")
model = torch.compile(model, mode="reduce-overhead")

prompt = "Artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

NUM_TOKENS = 50

# ── Warmup obligatorio ─────────────────────────────────────────
# torch.compile compila en la PRIMERA ejecución (lazy compilation)
# Si medimos esa, incluimos el tiempo de compilación en los resultados
# El warmup descarta esa primera ejecución lenta
print("Warming up (compiling CUDA kernels, wait)...")
warmup_inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
with torch.no_grad():
    model.generate(
        **warmup_inputs,
        max_new_tokens=5,       # pocos tokens, solo para compilar
        do_sample=False,
        temperature=None,
        top_p=None,
    )
print("Warmup done, now measuring...")

print(f"Generating {NUM_TOKENS} tokens in INT4 + compiled...")
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

# ── VRAM ───────────────────────────────────────────────────────
vram_used = torch.cuda.memory_allocated() / 1024**3
vram_reserved = torch.cuda.memory_reserved() / 1024**3

print(f"\n=== INT4 + torch.compile RESULTS ===")
print(f"Generated tokens:  {generated_tokens}")
print(f"Total time:        {elapsed:.2f} seconds")
print(f"Speed:             {speed:.2f} tokens/sec")
print(f"VRAM allocated:    {vram_used:.2f} GB")
print(f"VRAM reserved:     {vram_reserved:.2f} GB")
print(f"\nGenerated text:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
