from torch import nn
from utils import has_implemented_forward, iterate_module
from transformer_lens.hook_points import HookPoint
from Components.AutoHooked import HookedInstance, auto_hook
from typing import Any, List, Union
from collections import Counter
from typing import TypeVar, Type
from tests.test_models import HF_MODELS
import torch
from inspect import isclass

T = TypeVar('T', bound=nn.Module)

def get_duplicates(lst : list[str]) -> list[str]:
    return [item for item, count in Counter(lst).items() if count > 1]

BUILT_IN_MODULES = [
    nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm ,nn.LayerNorm, nn.Embedding,
    nn.RNN, nn.LSTM, nn.GRU, nn.RNNCell, nn.LSTMCell, nn.GRUCell, 
    # Add more built-in module types as needed
]

def prepare_model_and_input(
    module_class_or_instance : Union[T, Type[T]], 
    inp_shape : Union[torch.Size, List[int]], 
    kwargs : dict[str, Any]
):
    is_hf = check_is_hf(module_class_or_instance)
    if isclass(module_class_or_instance):
        model = init_model(auto_hook(module_class_or_instance), kwargs, is_hf)
    else:
        model = auto_hook(module_class_or_instance)
    
    vocab = kwargs.get('vocab_size', 10) if isinstance(kwargs, dict) else kwargs.vocab_size
    
    if is_hf:
        assert vocab is not None, "vocab_size is required for HF models"
        input = torch.randint(0, vocab, inp_shape)
    else:
        input = torch.randn(inp_shape)
    return model, input

def check_is_hf(model: Union[T, Type[T]]) -> bool:
    return model in HF_MODELS or type(model) in HF_MODELS

def init_model(module_class, kwargs, is_hf: bool = False):
    if is_hf:
        return module_class(config=kwargs)
    return module_class(**kwargs)

def generate_expected_hookpoints(model : Union[HookedInstance, nn.Module],  prefix='') -> list[str]:
    expected_hooks = set()

    if isinstance(model, HookedInstance):
        model = model._module

    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        print("name, module", name, module)
        if isinstance(module, HookPoint):
            continue
        
        if isinstance(module, (nn.ModuleList, nn.Sequential, nn.ModuleDict)):
            #these are module containers only their children should be hooked
            for key, child in iterate_module(module):
                expected_hooks.update(generate_expected_hookpoints(child, f"{full_name}.{key}"))
            expected_hooks.update(generate_expected_hookpoints(module, full_name))
        elif not full_name.endswith('.hook_point'):
            expected_hooks.add(f"{full_name}.hook_point")  
            
            if not any(isinstance(module, built_in) for built_in in BUILT_IN_MODULES):
                expected_hooks.update(generate_expected_hookpoints(module, full_name))

    # Only add parameter hook points for non-built-in modules
    if not any(isinstance(model, built_in) for built_in in BUILT_IN_MODULES):
        for name, param in model.named_parameters(recurse=False):
            print(f"name, param", name, param)
            full_name = f"{prefix}.{name}" if prefix else name
            if not full_name.endswith('.hook_point'):
                expected_hooks.add(f"{full_name}.hook_point")
        
    return list(expected_hooks)
