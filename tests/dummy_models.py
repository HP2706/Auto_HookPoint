from Models import BaseTransformerConfig, VanillaTransformerBlock
from transformers.models.llama import  LlamaConfig
from transformers.models.mixtral import MixtralConfig
import torch.nn as nn
import torch

small_llama_config = LlamaConfig(
    vocab_size=1000,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=4,
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

class SimpleModelWithModuleDict(nn.Module):
    def __init__(self):
        super().__init__()
        self.bla = nn.ModuleDict({"0": nn.Linear(10, 10), "1": nn.Linear(10, 10)})

    def get_forward_shape(self):
        return torch.Size([1, 10])

    def forward(self, input_ids):
        x = input_ids
        x = self.bla["0"](x)
        x = self.bla["1"](x)
        return x

class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner1 = nn.Linear(10, 10)
    
    def get_forward_shape(self):
        return torch.Size([1, 10])

    def forward(self, input_ids):
        x = input_ids
        return self.inner1(x)

class SimpleNestedModuleList(nn.Module):
    def __init__(self):
        super().__init__()
        self.bla = nn.ModuleList([SimpleModule(), SimpleModule()])

    def get_forward_shape(self):
        return self.bla[0].get_forward_shape()

    def forward(self, input_ids):
        x = input_ids
        for module in self.bla:
            x = module(x)
        return x
    
class ComplexNestedModule(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = BaseTransformerConfig(
            d_model = 128,
            n_layers = 3,
            num_heads = 4,
            is_training=True,
        )
        self.cfg = cfg
        self.bla = nn.ModuleList([VanillaTransformerBlock(cfg)])

    def get_forward_shape(self):
        return torch.Size([1, 10, self.cfg.d_model])

    def forward(self, input_ids):
        x = input_ids
        for module in self.bla:
            x = module(x)
        return x
