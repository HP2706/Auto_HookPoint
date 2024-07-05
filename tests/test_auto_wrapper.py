from inspect import isclass
from typing import List, Tuple, Protocol, TypeVar, Union
from Components.AutoHooked import WrappedInstance, auto_hooked
from transformer_lens.hook_points import HookPoint
from Models import BaseTransformerConfig, VanillaTransformerBlock
from test_utils import generate_expected_hookpoints
import torch.nn as nn
import torch
import pytest
from functools import partial
from abc import ABC


class Model(nn.Module):...

T = TypeVar('T', bound=Model)

def check_hook_types(hook_list : List[Tuple[str, str]]):
    assert all(isinstance(t, HookPoint) for t in [tup[1] for tup in hook_list])

def generic_check_hook_fn_works(model : T, inp_shape : torch.Size):
    counter = {'value': 0} #GLOBAL state
    def print_shape(x, hook=None, hook_name=None):
        print(f"HOOK NAME: {hook_name}")
        counter['value'] += 1
        return x

    hook_names = [hook_name for hook_name, _ in model.list_all_hooks()]
    print(f"HOOK NAMES: {hook_names}")

    model.run_with_hooks(
        x=torch.randn(inp_shape),
        fwd_hooks=[(hook_name, partial(print_shape, hook_name=hook_name)) for hook_name in hook_names]
    )
    assert counter['value'] == len(hook_names), f"counter['value'] == len(hook_names), {counter['value']} == {len(hook_names)}"
    print("TEST PASSED")

def generic_check_all_hooks(model, inp_shape : torch.Size):
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
    print("TEST PASSED")

class SimpleModelWithModuleDict(Model):
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

class SimpleNestedModuleList(Model):
    def __init__(self):
        super().__init__()
        self.bla = nn.ModuleList([SimpleModule(), SimpleModule()])

    def get_forward_shape(self):
        return self.bla[0].get_forward_shape()

    def forward(self, x):
        for module in self.bla:
            x = module(x)
        return x
    
class ComplexNestedModule(Model):
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

    def forward(self, x):
        for module in self.bla:
            x = module(x)
        return x

# Test cases
@pytest.mark.parametrize("module_class, inp_shape", [
    (SimpleModule , torch.Size([1, 10])),
    (SimpleModelWithModuleDict, torch.Size([1, 10])),
    (SimpleNestedModuleList, torch.Size([1, 10])),
    (ComplexNestedModule, torch.Size([1, 10, 128])),
    (SimpleModule(), torch.Size([1, 10])),
    (SimpleModelWithModuleDict(), torch.Size([1, 10])),
    (SimpleNestedModuleList(), torch.Size([1, 10])),
    (ComplexNestedModule(), torch.Size([1, 10, 128])),
])
def test_hook_fn_works(module_class : T, inp_shape : torch.Size):
    if isclass(module_class):
        print("IS CLASS")
        model = auto_hooked(module_class)()
    else:
        print("IS INSTANCE")
        model = auto_hooked(module_class)
    generic_check_hook_fn_works(model, inp_shape)

@pytest.mark.parametrize("module_class, inp_shape", [
    (SimpleModule, torch.Size([1, 10])),
    (SimpleModelWithModuleDict, torch.Size([1, 10])),
    (SimpleNestedModuleList, torch.Size([1, 10])),
    (ComplexNestedModule, torch.Size([1, 10])),
    (SimpleModule(), torch.Size([1, 10])),
    (SimpleModelWithModuleDict(), torch.Size([1, 10])),
    (SimpleNestedModuleList(), torch.Size([1, 10])),
    (ComplexNestedModule(), torch.Size([1, 10])),
])
def test_check_all_hooks(module_class : T, inp_shape : torch.Size):
    # NOTE we are implicitly testing wrap_cls() and wrap_instance()
    # if auto_hooked is give a cls as input it will wrap the cls with wrap_class() and return the wrapped cls
    # if an instance is given as input it will wrap the instance with wrap_instance() and return the wrapped instance
    # SO ORDER matters
    if isclass(module_class):
        model = auto_hooked(module_class)()
    else:
        model = auto_hooked(module_class)
    generic_check_all_hooks(model, inp_shape)


@pytest.mark.parametrize("module_class", [
    SimpleModule,
    SimpleModelWithModuleDict,
    SimpleNestedModuleList,
    ComplexNestedModule,
])
def test_check_unwrap_cls_works(module_class : T):
    if not isclass(module_class):
        raise ValueError("module_class must be a class in this test")
    
    wrapped_cls = auto_hooked(module_class)
    assert module_class == wrapped_cls.unwrap_cls()

@pytest.mark.parametrize("module_class", [
    SimpleModule(),
    SimpleModelWithModuleDict(),
    SimpleNestedModuleList(),
    ComplexNestedModule(),
])
def test_check_unwrap_instance_works(module_class : T):
    if isclass(module_class):
        raise ValueError("module_class must be an instance in this test")
    
    wrapped_cls = auto_hooked(module_class)
    assert module_class == wrapped_cls.unwrap_instance()
