from transformers.models.llama import  LlamaConfig
from transformers.models.mixtral import MixtralConfig
import torch.nn as nn
import torch
from transformers.models.llama import LlamaForCausalLM


small_mixtral_config = MixtralConfig(
    vocab_size=1000,
    hidden_size=18,
    intermediate_size=32,
    num_hidden_layers=1,
    num_attention_heads=1,
    num_key_value_heads=1,
    max_position_embeddings=512,
    num_experts_per_tok=2,
    num_local_experts=4,
    attention_dropout=0.1,
)

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

class SimpleModelWithModuleDict(nn.Module):
    def __init__(self):
        super().__init__()
        self.bla = nn.ModuleDict({"0": nn.Linear(10, 10), "1": nn.Linear(10, 10)})

    def get_forward_shape(self):
        return torch.Size([1, 10])

    def forward(self, x):
        x = self.bla["0"](x)
        x = self.bla["1"](x)
        return x

class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner1 = nn.Linear(10, 10)
    
    def get_forward_shape(self):
        return torch.Size([1, 10])

    def forward(self, x):
        return self.inner1(x)

class SimpleNestedModuleList(nn.Module):
    def __init__(self):
        super().__init__()
        self.bla = nn.ModuleList([SimpleModule(), SimpleModule()])

    def get_forward_shape(self):
        return self.bla[0].get_forward_shape()

    def forward(self, x):
        for module in self.bla:
            x = module(x)
        return x
    
class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_hidden = cfg["d_mlp"] * cfg["dict_mult"]
        d_mlp = cfg["d_mlp"]
        l1_coeff = cfg["l1_coeff"]
        dtype = torch.float32
        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_mlp, d_hidden, dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, d_mlp, dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_mlp, dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff

    def forward(self, x):
        x_cent = x - self.b_dec
        acts = torch.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coeff * (acts.float().abs().sum())
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss
    