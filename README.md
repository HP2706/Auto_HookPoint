# Automatic_Hook

AutoHooked is a Python library that makes it possible to use arbitrary models in transformer_lens. 
This happens via an auto_hook function that wraps your pytorch model and applies hookpoint for every major 

## Features

- Works with both `nn.Module` and `nn.Parameter` operations
- Can be use both as a class decorator or on an already instantiated model 

## Installation

```bash
pip install Automatic_Hook
```

## Usage

###Usage as decorator

```python
from Automatic_Hook import auto_hook
import torch.nn as nn

@auto_hook
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        #self.fc1_hook_point = HookPoint() NOW NOT NEEDED

    def forward(self, x):
        # self.fc1_hook_point(self.fc1(x)) NOW NOT NEEDED
        return self.fc1(x)

model = MyModel()
print(model.hook_dict.items())  # dict_items([('hook_point', HookPoint()), ('fc1.hook_point', HookPoint())])
```

### Wrap an instance

AutoHooked can also work with models that use `nn.Parameter`, such as this AutoEncoder example:

```python
from Automatic_Hook import auto_hook
import torch
from torch import nn

# taken from neel nandas excellent autoencoder tutorial: https://colab.research.google.com/drive/1u8larhpxy8w4mMsJiSBddNOzFGj7_RTn#scrollTo=MYrIYDEfBtbL
class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_hidden = cfg["d_mlp"] * cfg["dict_mult"]
        d_mlp = cfg["d_mlp"]
        dtype = torch.float32
        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(d_mlp, d_hidden, dtype=dtype)))
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(d_hidden, d_mlp, dtype=dtype)))
        self.b_enc = nn.Parameter(
            torch.zeros(d_hidden, dtype=dtype)
        )
        self.b_dec = nn.Parameter(
            torch.zeros(d_mlp, dtype=dtype)
        )

    def forward(self, x):
        x_cent = x - self.b_dec
        acts = torch.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        return x_reconstruct

autoencoder = auto_hook(AutoEncoder({"d_mlp": 10, "dict_mult": 10, "l1_coeff": 10, "seed": 1}))
print(autoencoder.hook_dict.items())
# dict_items([('hook_point', HookPoint()), ('W_enc.hook_point', HookPoint()), ('W_dec.hook_point', HookPoint()), ('b_enc.hook_point', HookPoint()), ('b_dec.hook_point', HookPoint())])
```

If this was to be done manually the code would be way less clean:

```python
class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_hidden = cfg['d_mlp'] * cfg['dict_mult']
        d_mlp = cfg['d_mlp']
        dtype = torch.float32
        torch.manual_seed(cfg['seed'])
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(d_mlp, d_hidden, dtype=dtype)
            )
        )
        self.W_enc_hook_point = HookPoint()
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(d_hidden, d_mlp, dtype=dtype)
            )
        )
        self.W_dec_hook_point = HookPoint()
        self.b_enc = nn.Parameter(
            torch.zeros(d_hidden, dtype=dtype)
        )
        self.b_enc_hook_point = HookPoint()
        self.b_dec = nn.Parameter(
            torch.zeros(d_mlp, dtype=dtype)
        )
        self.b_dec_hook_point = HookPoint()

    def forward(self, x):
        x_cent = self.b_dec_hook_point(x - self.b_dec)
        acts = torch.relu(self.b_enc_hook_point(self.W_enc_hook_point(x_cent @ self.W_enc) + self.b_enc))
        x_reconstruct = self.b_dec_hook_point(self.W_dec_hook_point(acts @ self.W_dec) + self.b_dec)
        return x_reconstruct
```

## Note 

There might be edge cases not supported for some weird reason so a function 'check_auto_hook' is provided to run the model class on all internal tests.

Note however that these might not always be informative, but can give hints/indications.

```python
from Automatic_Hook import check_auto_hook
hooked_model = auto_hook(model)
input_kwargs = {'x': torch.randn(10, 10)}
init_kwargs = {'cfg': {'d_mlp': 10, 'dict_mult': 10, 'l1_coeff': 10, 'seed': 1}}
check_auto_hook(AutoEncoder, input_kwargs, init_kwargs)
```

if strict is set to True a runtime error will be raised if the tests fail else 
a warning.

## Backward(bwd) Hook

Some trouble might occur this is specifcally when a model or its inner-components returns a non-tensor object which is then passed to a hook. I am working on how to resolve this. However this would still work if those hooks are just disabled.
