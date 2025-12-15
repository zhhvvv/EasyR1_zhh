    
import torch
from vllm import LLM
# 测试是否能初始化


llm = LLM(
    model="/lustre/projects/med-multi-llm/haohan/models/Qwen3-VL-4B-Instruct",  # 测试有问题的
    tensor_parallel_size=1,
    enforce_eager=True,
)
print("✓ Qwen3-VL works")
