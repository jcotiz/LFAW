from llama_cpp import Llama
 

llm = Llama.from_pretrained(
    repo_id="bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
    filename="Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
    n_gpu_layers=-1,  
    n_ctx=256,
    verbose=True,
)
 