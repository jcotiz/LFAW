import llama_cpp
print(llama_cpp.__version__)

from llama_cpp import Llama
print("Supports GPU:", llama_cpp.llama_supports_gpu_offload())