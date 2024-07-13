from inspect import isclass
from typing import Any, List, Optional, Tuple, Protocol, Type, TypeVar, Union
from Components.AutoHooked import HookedInstance, auto_hook
from transformer_lens.hook_points import HookPoint
from test_utils import (
    generate_expected_hookpoints, 
    get_duplicates, 
    prepare_model_and_input
)
import torch.nn as nn
import torch
import pytest
from functools import partial
from .test_models import (
    SimpleModule, 
    SimpleModelWithModuleDict, 
    SimpleNestedModuleList, 
    ComplexNestedModule,
    small_llama_config,
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
    print("inp_tensor", inp_tensor)

    model.run_with_hooks(
        input_ids=inp_tensor, # assume d_vocab is at least 1000
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

# Define common test cases
CLASS_TEST_CASES = [
    (SimpleModule, torch.Size([1, 10]), {}),
    (SimpleModelWithModuleDict, torch.Size([1, 10]), {}),
    (SimpleNestedModuleList, torch.Size([1, 10]), {}),
    (ComplexNestedModule, torch.Size([1, 10, 128]), {}),
    (LlamaForCausalLM, torch.Size([1, 10]), {'config' : small_llama_config}),
]

INSTANCE_TEST_CASES = [
    (LlamaForCausalLM(config=small_llama_config), torch.Size([1, 10]), {'config' : small_llama_config}),
    (SimpleModule(), torch.Size([1, 10]), {}),
    (SimpleModelWithModuleDict(), torch.Size([1, 10]), {}),
    (SimpleNestedModuleList(), torch.Size([1, 10]), {}),
    (ComplexNestedModule(), torch.Size([1, 10, 128]), {}),
]
ALL_TEST_CASES = CLASS_TEST_CASES + INSTANCE_TEST_CASES 

@pytest.mark.parametrize("module_class, inp_shape, kwargs", ALL_TEST_CASES)
def test_hook_fn_works(
    module_class: T, 
    inp_shape: torch.Size, 
    kwargs: dict[str, Any]
):
    model, input = prepare_model_and_input(module_class, inp_shape, kwargs)
    generic_check_hook_fn_works(model, input)

@pytest.mark.parametrize("module_class, inp_shape, kwargs", INSTANCE_TEST_CASES)
def test_check_all_hooks(
    module_class: T, 
    inp_shape: torch.Size, 
    kwargs: dict[str, Any]
):
    model, _ = prepare_model_and_input(module_class, inp_shape, kwargs)
    generic_check_all_hooks(model, inp_shape)


@pytest.mark.parametrize("module, _, kwargs", ALL_TEST_CASES)
def test_check_unwrap_works(
    module: Union[T, Type[T]], 
    _: torch.Size, 
    kwargs: dict[str, Any]
):
    if isclass(module):
        unwrapped = auto_hook(module).unwrap_cls()
    else:
        model, _ = prepare_model_and_input(module, _, kwargs) #type: ignore
        unwrapped = model.unwrap_instance()
    assert unwrapped == module, f"Unwrapped {unwrapped} is not the same as the original {module}"
    
@pytest.mark.parametrize("module, _ , kwargs", ALL_TEST_CASES)
def test_duplicate_hooks(module: Union[T, Type[T]], _: torch.Size, kwargs: dict[str, Any]):
    model, _ = prepare_model_and_input(module, _, kwargs) #type: ignore
    hooks = [hook_name for hook_name, _ in model.list_all_hooks()]
    assert len(hooks) == len(set(hooks)), f"Duplicate hooks: {hooks}, hooks: {hooks} duplicates: {get_duplicates(hooks)}"

@pytest.mark.parametrize("module, _ , kwargs", CLASS_TEST_CASES)
def test_generate_expected_hookpoints(
    module: Type[T], 
    _: torch.Size, 
    kwargs: dict[str, Any]
):
    no_hook_expected = generate_expected_hookpoints(module(**kwargs))
    hook_expected_1 = generate_expected_hookpoints(auto_hook(module)(**kwargs))
    hook_expected_2 = generate_expected_hookpoints(auto_hook(module(**kwargs)))
    assert no_hook_expected == hook_expected_1 == hook_expected_2, "Expected hookpoints do not match"