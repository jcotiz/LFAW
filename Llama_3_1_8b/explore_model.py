from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from dotenv import load_dotenv
import torch
import os

load_dotenv()

login(token=os.getenv("HF_TOKEN"))

print("Downloading model... (first time takes 15 min, then its cached)")

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.float16,
    device_map="cpu"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# Basic inspection
print("\n=== ARCHITECTURE ===")
print(f"Number of layers:     {len(model.model.layers)}")
print(f"Model dimension:      {model.config.hidden_size}")
print(f"FFN dimension:        {model.config.intermediate_size}")
print(f"Attention heads:      {model.config.num_attention_heads}")

print("\n=== LAYER 0 WEIGHTS ===")
layer = model.model.layers[0]
print(f"gate_proj shape: {layer.mlp.gate_proj.weight.shape}")
print(f"q_proj shape:    {layer.self_attn.q_proj.weight.shape}")
print(f"k_proj shape:    {layer.self_attn.k_proj.weight.shape}")

print("\n=== TOTAL PARAMETERS ===")
total = sum(p.numel() for p in model.parameters())
print(f"{total / 1e9:.2f} B parameters")