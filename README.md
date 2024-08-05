# Auto_HookPoint

Auto_HookPoint is a Python library that seamlessly integrates arbitrary PyTorch models with transformer_lens. It provides an `auto_hook` function that automatically wraps your PyTorch model, applying HookPoints to every `nn.Module` and select `nn.Parameter` instances within the model structure. 

## Features

- Works with both `nn.Module` and `nn.Parameter` operations
- Can be used both as a class decorator or on an already instantiated model 
- Makes code cleaner

## Installation

```bash
pip install Auto_HookPoint
```

## Usage

### Usage as decorator

```python
from Auto_HookPoint import auto_hook
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

orig_model = model.unwrap() #get back the original model

```

### Wrap an instance

Auto_HookPoint can also work with models that use `nn.Parameter`, such as this AutoEncoder example:

```python
from Auto_HookPoint import auto_hook
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


input_kwargs = {'x': torch.randn(10, 10)}

def hook_fn(x, hook=None, hook_name=None):
    print('hello from hook:', hook_name)
    return x

autoencoder.run_with_hooks(
    **input_kwargs, 
    fwd_hooks=[
        (hook_name, partial(hook_fn, hook_name=hook_name))
        for hook_name in autoencoder.hook_dict.keys()
    ]
)

#if you want full typing support after hooking your model
# a hacky solution would be:
class Model(HookedRootModule, AutoEncoder):
    pass

autoencoder = cast(Model, autoencoder)
# autoencoder.forward() is now type hinted in vscode
```

## auto_hook + huggingface transformers

auto_hook can also work with hf-models

```python
from Auto_HookPoint import auto_hook, check_auto_hook
from transformers.models.mamba.modeling_mamba import MambaForCausalLM, MambaConfig
import torch

model = MambaForCausalLM(mamba_cfg)
model = auto_hook(model)
print('model.mod_dict', model.hook_d_dict.keys()) 
```

## auto_hook + manual hookpointing

As auto_hook will not hook arbitrary tensor manipulation functions, sometimes manual hooking will be necessary. for instance if using torch.relu() instead of nn.Relu(). Luckily auto_hook does not modify the existing hooks, so you can still use them.   

```python
@auto_hook
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.relu_hook_point = HookPoint()
    def forward(self, x):
        x = self.linear(x)
        x = self.relu_hook_point(torch.relu(x))
        return x
```

## Train SAE´s with sae_lens

with auto_hook you can train a SparseAutoEncoder on any huggingface transformers model via sae_lens


```python
#most of the credit for this example goes to https://gist.github.com/joelburget
#check https://github.com/HP2706/Auto_HookPoint/blob/main/examples/sae_lens.py for a complete example
from Auto_HookPoint import HookedTransformerAdapter 
#install via: pip install sae_lens
from sae_lens import SAETrainingRunner, LanguageModelSAERunnerConfig

cfg = LanguageModelSAERunnerConfig(
    model_name=model_name,
    hook_name="model.norm.hook_point",
    ...
)

hooked_model = HookedTransformerAdapter(Cfg(device="cuda", n_ctx=512), hf_model_name= model_name)
sparse_autoencoder = SAETrainingRunner(cfg, override_model=hooked_model).run()
```
### Note on SAE-Lens Integration:
1. Not all hook_points are compatible as run_with_cache only works for hook_points that return pure tensors which most 
hf transformers block does not do. This is a limitation that can only be removed by changin transformer_lens, sae_lens or both.
2. SAE-Lens expects activations with shape [batch, sequence_length, hidden_size].
   Some hookpoints (e.g., MixtralSparseMoeBlock gate) may not work due to different shapes.
3. If your model has more than one nn.Embedding attribute specify which one is the input embedding via the `input_embedding_name` parameter in HookedTransformerAdapter. 
Note that after the model is hooked the naming of the self.model.embed_tokens(nn.Embedding) attribute becomes self.model._module.model._module.embed_tokens._module.weight
4. auto_hook does not yet support premature stopping via stop_at_layer in the forward pass, which would make building the activation_store in sae_lens impractible for very large models.

## Note 

To ensure comprehensive coverage and identify potential edge cases, the 'check_auto_hook' function is provided. This utility runs the model class through a suite of internal tests, helping to validate the auto-hooking process and catch any unexpected behaviors or unsupported scenarios.

Note however that these might not always be informative specifically the bwd_hook test function should generally be ignored.

```python
from Auto_HookPoint import check_auto_hook
hooked_model = auto_hook(model)
input_kwargs = {'x': torch.randn(10, 10)}
init_kwargs = {'cfg': {'d_mlp': 10, 'dict_mult': 10, 'l1_coeff': 10, 'seed': 1}}
check_auto_hook(AutoEncoder, input_kwargs, init_kwargs)
```

If strict is set to True, a runtime error will be raised if the tests fail; otherwise, 
a warning will be issued. 

## Note on Backward Hooks (bwd_hooks)
Some issues might occur when using backward hooks. As auto_hook hooks anything that is an instance of nn.Module, modules that return non-tensor objects will also be hooked. It is advised to only use backward hooks on hookpoints that returns tensors as output.
