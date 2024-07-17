import os
import sys
# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union
from Auto_HookPoint.hook import auto_hook
from transformer_lens.hook_points import HookPoint
from transformers.utils.generic import ModelOutput
from .test_utils import (
    generate_expected_hookpoints, 
    get_duplicates, 
)
import torch.nn as nn
import torch
import pytest
from functools import partial
from .test_models import get_test_cases, get_base_cases, get_hf_cases
#for testing 

T = TypeVar('T', bound=nn.Module)
P = TypeVar('P', bound=nn.Parameter)

def check_hook_types(
    hook_list : List[Tuple[str, str]]
):
    assert all(isinstance(t, HookPoint) for t in [tup[1] for tup in hook_list])

def generic_hook_check(
    model: T,
    input: Dict[str, torch.Tensor],
    hook_fn,
    is_backward: bool
):
    counter = {'value': 0, 'hooks': []}

    hook_names = [hook_name for hook_name, _ in model.list_all_hooks()]
    hooks = [(name, partial(hook_fn, hook_name=name)) for name in hook_names]
    
    if is_backward:
        output = model.run_with_hooks(**input, bwd_hooks=hooks)
        loss = get_loss(output)
        loss.backward()
    else:
        model.run_with_hooks(**input, fwd_hooks=hooks)

    unused_hooks = set(hook_names) - set(counter['hooks'])
    hooks_multiple_times = list(set([hook for hook in counter['hooks'] if counter['hooks'].count(hook) > 1]))
    
    assert (
        counter['value'] == len(hook_names) or len(hooks_multiple_times) > 0 or len(unused_hooks) > 0
    ), (
        f"counter['value'] == len(hook_names) and len(hooks_multiple_times) == 0, "
        f"{counter['value']} == {len(hook_names)}, "
        f"{hook_names} unused: {unused_hooks} "
        f"hooks called multiple times: {hooks_multiple_times}"
    )
    print("TEST PASSED")

def get_loss(output):
    if isinstance(output, torch.Tensor):
        return output.sum() if output.shape != [] else output
    elif issubclass(type(output), ModelOutput):
        return output.loss
    else:
        for elm in output:
            if isinstance(elm, torch.Tensor) and elm.requires_grad:
                return elm.sum() if elm.shape != [] else elm
        raise ValueError("No suitable tensor found for backward pass")


def generic_check_hook_fn_bwd_works(model: T, input: Dict[str, torch.Tensor]):
    counter = {'value': 0, 'hooks': []}
    def backward_hook(grad_output, hook: Optional[HookPoint] = None, hook_name: str = '') -> Union[Any, None]:
        counter['value'] += 1
        counter['hooks'].append(hook_name)
        if isinstance(grad_output, tuple):
            if any(g is not None and g.requires_grad for g in grad_output):
                return grad_output
        elif grad_output is not None and grad_output.requires_grad:
            return (grad_output,)
    
    generic_hook_check(model, input, backward_hook, is_backward=True)

def generic_check_hook_fn_fwd_works(model: T, input: Dict[str, torch.Tensor]):
    counter = {'value': 0, 'hooks': []}
    def hook_wrapper(x, hook=None, hook_name=None):
        counter['value'] += 1
        counter['hooks'].append(hook_name)
        return x
    generic_hook_check(model, input, hook_wrapper, is_backward=False)

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

@pytest.mark.parametrize("module, input", get_test_cases())
def test_hook_fn_fwd_works(
    module: T, 
    input : Dict[str, torch.Tensor]
):
    model = auto_hook(module)
    generic_check_hook_fn_fwd_works(model, input)


@pytest.mark.parametrize("module, input", get_base_cases())
def test_fwd_hook_fn_edit(
    module: T, 
    input : Dict[str, torch.Tensor]
):
    model = auto_hook(module)
    
    def hook_wrapper(x, hook=None, hook_name=None) -> torch.Tensor:
        if isinstance(x, tuple):
            return tuple(g + 1 for g in x if isinstance(g, torch.Tensor))
        elif isinstance(x, torch.Tensor):
            return x + 1
        else:
            return x
    
    
    hook_names = [hook_name for hook_name, _ in model.list_all_hooks()]
    hooks = [(name, partial(hook_wrapper, hook_name=name)) for name in hook_names]
    
    # Run model with hooks
    output_with_hooks = model.run_with_hooks(**input, fwd_hooks=hooks)
    
    # Run model without hooks
    output_without_hooks = model(**input)
    
    # Check if the difference is equal to the number of hooks
    diff = output_with_hooks - output_without_hooks
    
    assert torch.all(diff != 0), f"Expected non-zero difference for all elements, but got: {diff}"
    print("TEST PASSED")

@pytest.mark.parametrize("module, input", get_base_cases())
def test_bwd_hook_fn_edit(module: T, input: Dict[str, torch.Tensor]):
    model = auto_hook(module)

    def backward_hook(grad_output, hook=None, hook_name: str = ''):
        if isinstance(grad_output, tuple):
            return tuple(g * 2 if isinstance(g, torch.Tensor) and g.requires_grad else g for g in grad_output)
        return (grad_output * 2,) if isinstance(grad_output, torch.Tensor) else (grad_output,)
    
    def get_grad_dict(loss):
        loss.backward()
        grad_dict = {name: param.grad.clone() if param.grad is not None else None 
                     for name, param in model.named_parameters()}
        model.zero_grad()
        return grad_dict

    hook_names = [hook_name for hook_name, _ in model.list_all_hooks()]
    hooks = [(name, partial(backward_hook, hook_name=name)) for name in hook_names]
    
    # No hooks forward pass
    no_hook_grad_dict = get_grad_dict(get_loss(model.forward(**input)))

    # Forward pass with hooks
    hook_grad_dict = get_grad_dict(get_loss(model.run_with_hooks(**input, bwd_hooks=hooks)))

    for name, param in model.named_parameters():
        #we check that the gradients are not the same
        #TODO make a better test
        assert not torch.allclose(no_hook_grad_dict[name], hook_grad_dict[name]), f"{name} grads are the same but they should be different"
        


@pytest.mark.parametrize("module, input", get_test_cases())
def test_hook_fn_bwd_works(
    module: T, 
    input : Dict[str, torch.Tensor]
):
    model = auto_hook(module)
    generic_check_hook_fn_bwd_works(model, input)

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
