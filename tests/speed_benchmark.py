import os
import sys
# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from typing import cast
from Auto_HookPoint import HookedTransformerAdapter, auto_hook
from test_models import get_hf_cases, small_llama_config
import torch
from transformers import LlamaForCausalLM
import timeit
import gc


#DUMMY check that wrapping isn't massively slowing down the model 
def run_benchmark(model, input_ids, num_runs=5):
    # Warm-up run
    model.forward(input_ids=input_ids)
    
    def benchmark_func():
        model.forward(input_ids=input_ids)
    
    times = timeit.repeat(benchmark_func, number=1, repeat=num_runs)
    return sum(times) / len(times)

# Set up models and input
llama_mini = cast(LlamaForCausalLM, get_hf_cases()[0][0])
wrapped_llama = auto_hook(llama_mini)
input_ids = torch.randint(0, small_llama_config.vocab_size, (1, 12))

# Run benchmarks
num_runs = 5
original_times = []
wrapped_times = []

for i in range(num_runs):
    gc.collect()
    
    # Run original model
    avg_time_original = run_benchmark(llama_mini, input_ids)
    original_times.append(avg_time_original)
    print(f"Run {i+1}: Original model average time: {avg_time_original:.6f} seconds")
    
    # Clear memory
    gc.collect()
    
    avg_time_wrapped = run_benchmark(wrapped_llama, input_ids)
    wrapped_times.append(avg_time_wrapped)
    print(f"Run {i+1}: Wrapped model average time: {avg_time_wrapped:.6f} seconds")

# Calculate overall averages
print(f"\nOverall average time for original model: {sum(original_times) / num_runs:.6f} seconds")
print(f"Overall average time for wrapped model: {sum(wrapped_times) / num_runs:.6f} seconds")