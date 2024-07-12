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
    (SimpleModule(), torch.Size([1, 10]), {}),
    (SimpleModelWithModuleDict(), torch.Size([1, 10]), {}),
    (SimpleNestedModuleList(), torch.Size([1, 10]), {}),
    (ComplexNestedModule(), torch.Size([1, 10, 128]), {}),
]
ALL_TEST_CASES = CLASS_TEST_CASES + INSTANCE_TEST_CASES 

def init_model(module_class, kwargs, is_hf: bool = False):
    if is_hf:
        return module_class(config=kwargs)
    return module_class(**kwargs)

def check_is_hf(model: Union[T, Type[T]]) -> bool:
    return model in HF_MODELS or type(model) in HF_MODELS

def prepare_model_and_input(module_class_or_instance, inp_shape, kwargs):
    is_hf = check_is_hf(module_class_or_instance)
    if isclass(module_class_or_instance):
        model = init_model(auto_hooked(module_class_or_instance), kwargs, is_hf)
    else:
        model = auto_hooked(module_class_or_instance)
    
    vocab = kwargs.get('vocab_size', 10) if isinstance(kwargs, dict) else kwargs.vocab_size
    
    if is_hf:
        assert vocab is not None, "vocab_size is required for HF models"
        input = torch.randint(0, vocab, inp_shape)
    else:
        input = torch.randn(inp_shape)
    
    return model, input

@pytest.mark.parametrize("module_class, inp_shape, kwargs", ALL_TEST_CASES)
def test_hook_fn_works(module_class: T, inp_shape: torch.Size, kwargs: dict[str, Any]):
    model, input = prepare_model_and_input(module_class, inp_shape, kwargs)
    generic_check_hook_fn_works(model, input)

@pytest.mark.parametrize("module_class, inp_shape, kwargs", INSTANCE_TEST_CASES)
def test_check_all_hooks(module_class: T, inp_shape: torch.Size, kwargs: dict[str, Any]):
    model, _ = prepare_model_and_input(module_class, inp_shape, kwargs)
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
def test_check_unwrap_instance_works(module_instance: T, _, kwargs: dict[str, Any]):
    model, _ = prepare_model_and_input(module_instance, _, kwargs)
    unwrapped = model.unwrap_instance()
    assert isinstance(unwrapped, type(module_instance)), f"Unwrapped instance {unwrapped} is not an instance of {type(module_instance)}"
    
@pytest.mark.parametrize("module, _ , kwargs", ALL_TEST_CASES)
def test_duplicate_hooks(module: Union[T, Type[T]], _: torch.Size, kwargs: dict[str, Any]):
    model, _ = prepare_model_and_input(module, _, kwargs)
    hooks = [hook_name for hook_name, _ in model.list_all_hooks()]
    assert len(hooks) == len(set(hooks)), f"Duplicate hooks: {hooks} , hooks: {hooks} duplicates: {get_duplicates(hooks)}"

@pytest.mark.parametrize("module, _ , kwargs", CLASS_TEST_CASES)
def test_generate_expected_hookpoints(module: Type[T], _: torch.Size, kwargs: dict[str, Any]):
    is_hf = check_is_hf(module)
    no_hook_expected = generate_expected_hookpoints(init_model(module, kwargs, is_hf))
    hook_expected_1 = generate_expected_hookpoints(init_model(auto_hooked(module), kwargs, is_hf))
    hook_expected_2 = generate_expected_hookpoints(auto_hooked(init_model(module, kwargs, is_hf)))
    assert no_hook_expected == hook_expected_1 == hook_expected_2, "Expected hookpoints do not match"