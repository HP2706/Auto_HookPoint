from inspect import isclass
from typing import Any, List, Optional, Tuple, Protocol, Type, TypeVar, Union
from Components.AutoHooked import WrappedInstance, auto_hooked
from transformer_lens.hook_points import HookPoint
from test_utils import generate_expected_hookpoints, get_duplicates
import torch.nn as nn
import torch
import pytest
from functools import partial
from .dummy_models import (
    SimpleModule, 
    SimpleModelWithModuleDict, 
    SimpleNestedModuleList, 
    ComplexNestedModule,
    small_llama_config
)
from transformers.models.llama import LlamaForCausalLM
from transformers.models.mixtral import MixtralForCausalLM
#for testing 

T = TypeVar('T', bound=nn.Module)

def check_hook_types(hook_list : List[Tuple[str, str]]):
    assert all(isinstance(t, HookPoint) for t in [tup[1] for tup in hook_list])

def generic_check_hook_fn_works(model : T, inp_tensor : torch.Tensor):
    counter = {'value': 0} #GLOBAL state
    def print_shape(x, hook=None, hook_name=None):
        print(f"HOOK NAME: {hook_name}")
        counter['value'] += 1
        return x

    hook_names = [hook_name for hook_name, _ in model.list_all_hooks()]
    #print(f"HOOK NAMES: {hook_names}")

    model.run_with_hooks(
        input_ids=inp_tensor, # assume d_vocab is at least 1000
        fwd_hooks=[(hook_name, partial(print_shape, hook_name=hook_name)) for hook_name in hook_names]
    )
    assert counter['value'] == len(hook_names), f"counter['value'] == len(hook_names), {counter['value']} == {len(hook_names)}"
    print("TEST PASSED")

def generic_check_all_hooks(model, inp_shape : torch.Size):
    expected_hookpoints = generate_expected_hookpoints(model)
    print('expected_hookpoints', expected_hookpoints)
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


HF_MODELS = [LlamaForCausalLM]

# Define common test cases
CLASS_TEST_CASES = [
    (SimpleModule, torch.Size([1, 10]), {}),
    (SimpleModelWithModuleDict, torch.Size([1, 10]), {}),
    (SimpleNestedModuleList, torch.Size([1, 10]), {}),
    (ComplexNestedModule, torch.Size([1, 10, 128]), {}),
    (LlamaForCausalLM, torch.Size([1, 10]), small_llama_config),
]

INSTANCE_TEST_CASES = [
    (LlamaForCausalLM(small_llama_config), torch.Size([1, 10]), small_llama_config),
    (SimpleModule, torch.Size([1, 10]), {}),
    (SimpleModelWithModuleDict, torch.Size([1, 10]), {}),
    (SimpleNestedModuleList, torch.Size([1, 10]), {}),
    (ComplexNestedModule, torch.Size([1, 10, 128]), {}),
]
ALL_TEST_CASES = CLASS_TEST_CASES + INSTANCE_TEST_CASES 

def init_model(module_class, kwargs, is_hf : bool = False):
    if is_hf:
        return module_class(config=kwargs)
    return module_class(**kwargs)

def check_is_hf(model : Union[T, Type[T]]) -> bool:
    return model in HF_MODELS or type(model) in HF_MODELS


@pytest.mark.parametrize("module_class, inp_shape, kwargs", ALL_TEST_CASES)
def test_hook_fn_works(
    module_class: T, 
    inp_shape: torch.Size, 
    kwargs: dict[str, Any]
):
    is_hf = check_is_hf(module_class)
    if isclass(module_class):
        model = init_model(auto_hooked(module_class), kwargs, is_hf)
    else:
        model = auto_hooked(module_class)
    
    if isinstance(kwargs, dict):
        vocab = kwargs.get('vocab_size', 10) #TODO 
    else:
        vocab = kwargs.vocab_size #TODO 

    if is_hf:
        assert vocab is not None, "vocab_size is required for HF models"
        input = torch.randint(0, vocab, inp_shape)
    else:
        input = torch.randn(inp_shape)
    generic_check_hook_fn_works(model, input)

@pytest.mark.parametrize("module_class, inp_shape, kwargs", ALL_TEST_CASES)
def test_check_all_hooks(
    module_class: T, 
    inp_shape: torch.Size, 
    kwargs: dict[str, Any]
):
    is_hf = check_is_hf(module_class)
    if isclass(module_class):
        model = init_model(auto_hooked(module_class), kwargs, is_hf)
    else:
        model = auto_hooked(module_class)
    generic_check_all_hooks(model, inp_shape)

@pytest.mark.parametrize("module_class, _, __", CLASS_TEST_CASES)
def test_check_unwrap_cls_works(
    module_class: T, 
    _, 
    __ : dict[str, Any]
):
    wrapped_cls = auto_hooked(module_class)
    assert module_class == wrapped_cls.unwrap_cls()

@pytest.mark.parametrize("module_instance, _, kwargs", INSTANCE_TEST_CASES)
def test_check_unwrap_instance_works(
    module_instance: T, 
    _, 
    kwargs: dict[str, Any]
):
    is_hf = check_is_hf(module_instance)
    if isclass(module_instance):
        module_instance = init_model(module_instance, kwargs, is_hf)
    else:
        module_instance = module_instance
    
    wrapped_instance = auto_hooked(module_instance)
    assert module_instance == wrapped_instance.unwrap_instance()

@pytest.mark.parametrize("module, _ , kwargs", ALL_TEST_CASES)
def test_duplicate_hooks(
    module: Union[T, Type[T]], 
    _ : torch.Size, 
    kwargs: dict[str, Any]
):
    is_hf = check_is_hf(module)
    if isclass(module):
        wrapped = init_model(auto_hooked(module), kwargs, is_hf)
        assert isinstance(wrapped, WrappedInstance), f"wrapped is not an instance of WrappedInstance, but {type(wrapped)}"
    else:
        wrapped = auto_hooked(module)
    
    hooks = [hook_name for hook_name, _ in wrapped.list_all_hooks()]
    assert len(hooks) == len(set(hooks)), f"Duplicate hooks: {hooks} , hooks: {hooks} duplicates: {get_duplicates(hooks)}"

@pytest.mark.parametrize("module, _ , kwargs", CLASS_TEST_CASES)
def test_generate_expected_hookpoints(
    module: Type[T], 
    _ : torch.Size, 
    kwargs: dict[str, Any]
):
    is_hf = check_is_hf(module)
    no_hook_expected = generate_expected_hookpoints(
        init_model(module, kwargs, is_hf)
    )
    hook_expected_1 = generate_expected_hookpoints(
        init_model(auto_hooked(module), kwargs, is_hf)
    )
    hook_expected_2 = generate_expected_hookpoints(
        auto_hooked(init_model(module, kwargs, is_hf))
    )
    assert no_hook_expected == hook_expected_1, f"no_hook_expected == hook_expected_1, {no_hook_expected} == {hook_expected_1}"
    assert no_hook_expected == hook_expected_2, f"no_hook_expected == hook_expected_2, {no_hook_expected} == {hook_expected_2}"