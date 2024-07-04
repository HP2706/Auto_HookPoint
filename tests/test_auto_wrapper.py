from inspect import isclass
from typing import List, Tuple
from Components.AutoHooked import AutoHookedRootModule, WrappedModule, auto_hooked
from transformer_lens.hook_points import HookPoint
from Models import BaseTransformerConfig, VanillaTransformerBlock
from utils import generate_expected_hookpoints
import torch.nn as nn
import torch
import pytest
from functools import partial

def check_hook_types(hook_list : List[Tuple[str, str]]):
    assert all(isinstance(t, HookPoint) for t in [tup[1] for tup in hook_list])


def generic_check_hook_fn_works(model):
    counter = {'value': 0} #GLOBAL state
    def print_shape(x, hook=None, hook_name=None):
        counter['value'] += 1
        return x

    hook_names = [hook_name for hook_name, _ in model.list_all_hooks()]
    model.run_with_hooks(
        x=torch.randn(1, 10),
        fwd_hooks=[(hook_name, partial(print_shape, hook_name=hook_name)) for hook_name in hook_names]
    )
    assert counter['value'] == len(hook_names), f"counter['value'] == len(hook_names), {counter['value']} == {len(hook_names)}"
    print("TEST PASSED")

def generic_check_all_hooks(model):
    expected_hookpoints = generate_expected_hookpoints(model)

    # Compare with actual hookpoints
    hook_list = model.list_all_hooks()
    check_hook_types(hook_list)
    actual_hookpoints = [name for name, _ in hook_list]

    # Find missing hookpoints
    missing_hookpoints = set(expected_hookpoints) - set(actual_hookpoints)

    if missing_hookpoints:
        raise ValueError(
            f"Missing hookpoints: {missing_hookpoints} \n\n"
            f"Expected hookpoints: {expected_hookpoints} \n\n"
            f"Actual hookpoints: {actual_hookpoints} \n\n"
        )


class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner1 = nn.Linear(10, 10)

    def forward(self, x):
        return self.inner1(x)

class SimpleNestedModuleList(nn.Module):
    def __init__(self):
        super().__init__()
        self.bla = nn.ModuleList([SimpleModule(), SimpleModule()])

    def forward(self, x):
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
        self.bla = nn.ModuleList([VanillaTransformerBlock(cfg)])


# Test cases
@pytest.mark.parametrize("module_class", [
    SimpleModule ,
    #SimpleNestedModuleList,
    #ComplexNestedModule,
    SimpleModule(),
    #SimpleNestedModuleList(),
    #ComplexNestedModule(),
])
def test_hook_fn_works(module_class):
    if isclass(module_class):
        print("IS CLASS")
        model = auto_hooked(module_class)()
    else:
        print("IS INSTANCE")
        model = auto_hooked(module_class)
    generic_check_hook_fn_works(model)

@pytest.mark.parametrize("module_class", [
    SimpleModule,
    #SimpleNestedModuleList,
    #ComplexNestedModule,
    SimpleModule(),
    #SimpleNestedModuleList(),
    #ComplexNestedModule(),
])
def test_check_all_hooks(module_class):
    if isclass(module_class):
        model = auto_hooked(module_class)()
    else:
        model = auto_hooked(module_class)
    generic_check_all_hooks(model)

