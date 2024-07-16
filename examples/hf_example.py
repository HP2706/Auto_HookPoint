from torch import nn
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Automatic_Hook import auto_hook, check_auto_hook
from transformers.models.llama import LlamaForCausalLM
from transformers.models.llama import  LlamaConfig

#Dummy config for llama hf
small_llama_config = LlamaConfig(
    vocab_size=1000,
    hidden_size=8,
    intermediate_size=16,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=4,
    hidden_act="silu",
    max_position_embeddings=512,
    initializer_range=0.02,
    rms_norm_eps=1e-6,
    use_cache=True,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    pretraining_tp=1,
    tie_word_embeddings=False,
    rope_theta=10000.0,
    rope_scaling=None,
    attention_bias=False,
    attention_dropout=0.1,
    mlp_bias=False,
)

input_kwargs = {
    'input_ids': torch.randint(0, 10, (10, 10)), 
    'labels': torch.randint(0, 10, (10, 10)),
    'return_dict': True
}


model = LlamaForCausalLM(config=small_llama_config)
hooked_model = auto_hook(model)
print(hooked_model.hook_dict.items())
check_auto_hook(LlamaForCausalLM, input_kwargs, {'config': small_llama_config})