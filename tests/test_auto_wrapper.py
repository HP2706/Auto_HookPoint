from inspect import isclass
from typing import Any, Dict, List, Optional, Tuple, Protocol, Type, TypeVar, Union
from Components.AutoHooked import HookedModule, auto_hook
from transformer_lens.hook_points import HookPoint
from transformers.modeling_outputs import CausalLMOutputWithPast
from test_utils import (
    generate_expected_hookpoints, 
    get_duplicates, 
)
import torch.nn as nn
import torch
import pytest
from functools import partial
from .test_models import (
    AutoEncoder,
    SimpleModule, 
    SimpleModelWithModuleDict, 
    SimpleNestedModuleList, 
    small_llama_config,
    small_mixtral_config
)
from transformers.models.llama import LlamaForCausalLM
from transformers.models.mixtral import MixtralForCausalLM
#for testing 

T = TypeVar('T', bound=nn.Module)
P = TypeVar('P', bound=nn.Parameter)

def check_hook_types(
    hook_list : List[Tuple[str, str]]
):
    assert all(isinstance(t, HookPoint) for t in [tup[1] for tup in hook_list])


def generic_check_hook_fn_bwd_works(
    model : T, 
    input : Dict[str, torch.Tensor]
):
    counter = {'value': 0, 'hooks' :[]} #GLOBAL state
    
    def skip_grad(output_grad: torch.Tensor, hook: Any, hook_name : str = ''):
        counter['value'] += 1
        return (output_grad,)

    hook_names = [hook_name for hook_name, _ in model.list_all_hooks()]
    output = model.run_with_hooks(**input, bwd_hooks=[('hook_point', partial(skip_grad, hook_name=name)) for name in model.hook_dict.keys()])

    print('output', output)
    if isinstance(output, torch.Tensor):
        if output.shape != []:
            loss = output.sum()
        else:
            loss = output
    elif isinstance(output, CausalLMOutputWithPast):
        loss = output.loss
    else:
        for elm in output:
            if isinstance(elm, torch.Tensor) and elm.requires_grad:
                if elm.shape != []:
                    loss = elm.sum()  # Reduce to scalar
                else:
                    loss = elm
                break
        else:
            raise ValueError("No suitable tensor found for backward pass")

        
    loss.backward()
    #hooks not used
    unused_hooks = set(hook_names) - set(counter['hooks'])
    hooks_multiple_times = list(set([hook for hook in counter['hooks'] if counter['hooks'].count(hook) > 1]))
    assert (
        counter['value'] == len(hook_names) or len(hooks_multiple_times) > 0 #or len(unused_hooks) > 0
    ), (
        f"counter['value'] == len(hook_names) and len(hooks_multiple_times) == 0, "
        f"{counter['value']} == {len(hook_names)}, "
        f"{hook_names} unused: {unused_hooks} "
        f"hooks called multiple times: {hooks_multiple_times}"
    )
    print("TEST PASSED")


def generic_check_hook_fn_fwd_works(
    model : T, 
    input : Dict[str, torch.Tensor]
):
    #NOTE this test is a bit cursed, is there a way to check that all 
    # the hooks that are part of the call graph are called?? instead of using heuristics
    # it is perfectly normal that some hooks are not called for instance if the model is sparse

    counter = {'value': 0, 'hooks' :[]} #GLOBAL state

    def print_shape(x, hook=None, hook_name=None):
        counter['value'] += 1
        counter['hooks'].append(hook_name)
        return x

    hook_names = [hook_name for hook_name, _ in model.list_all_hooks()]

    model.run_with_hooks(
        **input,
        fwd_hooks=[(hook_name, partial(print_shape, hook_name=hook_name)) for hook_name in hook_names],
    )

    #hooks not used
    unused_hooks = set(hook_names) - set(counter['hooks'])
    hooks_multiple_times = list(set([hook for hook in counter['hooks'] if counter['hooks'].count(hook) > 1]))
    assert (
        counter['value'] == len(hook_names) or len(hooks_multiple_times) > 0 #or len(unused_hooks) > 0
    ), (
        f"counter['value'] == len(hook_names) and len(hooks_multiple_times) == 0, "
        f"{counter['value']} == {len(hook_names)}, "
        f"{hook_names} unused: {unused_hooks} "
        f"hooks called multiple times: {hooks_multiple_times}"
    )
    print("TEST PASSED")

def generic_check_all_hooks(model):
    expected_hookpoints = generate_expected_hookpoints(model)
    # Compare with actual hookpoints
    hook_list = model.list_all_hooks()
    check_hook_types(hook_list)
    actual_hookpoints = [name for name, _ in hook_list]

    # Find missing hookpoints
    missing_hookpoints = set(expected_hookpoints) - set(actual_hookpoints)
    additional_hookpoints = set(actual_hookpoints) - set(expected_hookpoints)

    if missing_hookpoints:
        raise ValueError(
            f"Missing hookpoints: {missing_hookpoints} \n\n"
            f"Additional hookpoints: {additional_hookpoints} \n\n"
            f"Expected hookpoints: {expected_hookpoints} \n\n"
            f"Actual hookpoints: {actual_hookpoints} \n\n"
        )
    print("TEST PASSED")

#module instance, input 
def get_test_cases():
    return [
        (SimpleModule(), {'x' : torch.randn(1, 10)}),
        (AutoEncoder(cfg = {"d_mlp": 10, "dict_mult": 1, "l1_coeff": 1, "seed": 1}), {'x' : torch.randn(1, 10)}),
        (SimpleModelWithModuleDict(), {'x' : torch.randn(1, 10)}),
        (SimpleNestedModuleList(), {'x' : torch.randn(1, 10)}),
        (LlamaForCausalLM(config=small_llama_config), {'input_ids' : torch.randint(0, 1000, (1, 10))}),
        (MixtralForCausalLM(config=small_mixtral_config), {'input_ids' : torch.randint(0, 1000, (1, 10))})
    ]

@pytest.mark.parametrize("module, input", get_test_cases())
def test_hook_fn_fwd_works(
    module: T, 
    input : Dict[str, torch.Tensor]
):
    model = auto_hook(module)
    generic_check_hook_fn_fwd_works(model, input)
"""
#TODO
@pytest.mark.parametrize("module, input", get_test_cases())
def test_hook_fn_bwd_works(
    module: T, 
    input : Dict[str, torch.Tensor]
):
    model = auto_hook(module)
    generic_check_hook_fn_bwd_works(model, input)"""

@pytest.mark.parametrize("module, _", get_test_cases())
def test_check_all_hooks(
    module: T, 
    _
):
    model = auto_hook(module)
    generic_check_all_hooks(model)

@pytest.mark.parametrize("module, _", get_test_cases())
def test_check_unwrap_works(
    module: T, 
    _,
):
    model = auto_hook(module)
    unwrapped = model.unwrap()
    assert unwrapped == module, f"Unwrapped {unwrapped} is not the same as the original {module}"
    
@pytest.mark.parametrize("module, _ ", get_test_cases())
def test_duplicate_hooks(
    module: T, 
    _
):
    model = auto_hook(module)
    hooks = [hook_name for hook_name, _ in model.list_all_hooks()]
    assert len(hooks) == len(set(hooks)), f"Duplicate hooks: {hooks}, hooks: {hooks} duplicates: {get_duplicates(hooks)}"

@pytest.mark.parametrize("module, _ ", get_test_cases())
def test_generate_expected_hookpoints(
    module: T, 
    _, 
):
    no_hook_expected = generate_expected_hookpoints(module)
    hook_expected_2 = generate_expected_hookpoints(auto_hook(module))

    diff1 = list(set(no_hook_expected) - set(hook_expected_2)) 
    diff2 = list(set(hook_expected_2) - set(no_hook_expected))
    assert set(no_hook_expected) == set(hook_expected_2), f"Expected hookpoints do not match: {no_hook_expected} != {hook_expected_2} diff1: {diff1} diff2: {diff2}"

@pytest.mark.parametrize(
    "module", 
    nn.Parameter(torch.randn(10))
)
def test_HookedParameter_unwrap(
    module: P, 
):
    model = auto_hook(module)
    assert model.unwrap() == module, f"Unwrapped {model.unwrap()} is not the same as the original {module}"

@pytest.mark.parametrize(
    "module", 
    [nn.Parameter(torch.randn(10))]
)
def test_HookedParameter_hook(
    module: P, 
):
    model = auto_hook(module)
    counter = {'value': 0, 'hooks' :[]} #GLOBAL state
    x = torch.ones(10).float()
    def print_shape(x, hook=None, hook_name=None):
        counter['value'] += 1
        counter['hooks'].append(hook_name)
        counter['shape'] = x.shape
        return x
    
    model.add_hook('hook_point', partial(print_shape, hook_name='hook_point')) #type: ignore
    model*1 # we multiply to test if the math ops methods are hooked correctly
    assert counter['value'] == 1, f"Counter value is not 1, {counter['value']}"
    assert counter['hooks'] == ['hook_point'], f"Hooks are not ['hook_point'], {counter['hooks']}"
    assert counter['shape'] == x.shape, f"Shape is not the same, {counter['shape']} != {x.shape}"
