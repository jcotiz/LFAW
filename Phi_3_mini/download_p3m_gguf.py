from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
    filename="Phi-3-mini-4k-instruct-q4.gguf",  # this work for me GPU GTX 1060 6GB VRAM 6.1 PASCAL 
    n_gpu_layers=-1,
    n_ctx=256,
    verbose=True,
)