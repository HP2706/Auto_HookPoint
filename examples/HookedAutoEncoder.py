'''
Often you want to hook not only nn.Module operations but also nn.Parameter operations.
I have built a miniversion of HookedRootModule but for nn.Parameter. 
Which means most mathematical operations that can be performed on an nn.Parameter instance can now be hooked.
'''

from torch import nn
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Auto_HookPoint import auto_hook
from Auto_HookPoint.check import check_auto_hook
from transformer_lens.hook_points import HookPoint

# example implementation taken from neel nandas excellent autoencoder tutorial: https://colab.research.google.com/drive/1u8larhpxy8w4mMsJiSBddNOzFGj7_RTn#scrollTo=MYrIYDEfBtbL
# this is used to show that we can hook models that only use nn.Parameter
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
        return x_reconstruct
    
autoencoder = auto_hook(AutoEncoder({"d_mlp": 10, "dict_mult": 10, "l1_coeff": 10, "seed": 1}))
print(autoencoder.hook_dict.items())
# dict_items([('hook_point', HookPoint()), ('W_enc.hook_point', HookPoint()), ('W_dec.hook_point', HookPoint()), ('b_enc.hook_point', HookPoint()), ('b_dec.hook_point', HookPoint())])
input_kwargs = {'x': torch.randn(10, 10)}
init_kwargs = {'cfg': {'d_mlp': 10, 'dict_mult': 10, 'l1_coeff': 10, 'seed': 1}}
check_auto_hook(AutoEncoder, input_kwargs, init_kwargs)