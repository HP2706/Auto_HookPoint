import os
import sys
# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
import copy
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union
from Auto_HookPoint.hook import auto_hook
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer
from transformers.utils.generic import ModelOutput
from .test_utils import (
    generate_expected_hookpoints, 
    get_duplicates, 
)
import torch.nn as nn
import torch
import pytest
from functools import partial
from .test_models import get_combined_cases, get_base_cases
#for testing 

T = TypeVar('T', bound=nn.Module)
P = TypeVar('P', bound=nn.Parameter)

def check_hook_types(
    hook_list : List[Tuple[str, str]]
):
    assert all(isinstance(t, HookPoint) for t in [tup[1] for tup in hook_list])

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
    
    generic_hook_check(model, input, backward_hook, is_backward=True, counter=counter)
    return counter  # Return the counter for assertion in the test function

def generic_check_hook_fn_fwd_works(model: T, input: Dict[str, torch.Tensor]):
    counter = {'value': 0, 'hooks': []}
    def hook_wrapper(x, hook=None, hook_name=None):
        counter['value'] += 1
        counter['hooks'].append(hook_name)
        return x
    generic_hook_check(model, input, hook_wrapper, is_backward=False, counter=counter)
    return counter  # Return the counter for assertion in the test function

def generic_hook_check(
    model: T,
    input: Dict[str, torch.Tensor],
    hook_fn,
    is_backward: bool,
    counter: Dict[str, Any]
):
    hooks = [(name, partial(hook_fn, hook_name=name)) for (name, _) in model.hook_dict.items()]
    
    if is_backward:
        output = model.run_with_hooks(**input, bwd_hooks=hooks)
        loss = get_loss(output)
        loss.backward()
    else:
        output = model.run_with_hooks(**input, fwd_hooks=hooks)
    
    hooks_multiple_times = list(set([hook for hook in counter['hooks'] if counter['hooks'].count(hook) > 1]))
    
    hook_names = [hook_name for hook_name, _ in hooks]
    assert (
        counter['value'] == len(hooks)
    ), (
        f"counter['value'] = {counter['value']}",
        f"len(hooks) = {len(hooks)}",
        f"{counter['value']} == {len(hook_names)}",
        f"hooks called multiple times: {hooks_multiple_times}",
        f"hooks not called: {set(hook_names) - set(counter['hooks'])}",
        f"hooks called: {set(counter['hooks'])}"
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



def generic_check_all_hooks(model):
    expected_hookpoints = generate_expected_hookpoints(model)
    # Compare with actual hookpoints
    hook_list = model.hook_dict.items()
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

@pytest.mark.parametrize("module, input", get_combined_cases())
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
    
    
    hook_names = [hook_name for hook_name, _ in model.hook_dict.items()]
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

    hook_names = [hook_name for hook_name, _ in model.hook_dict.items()]
    hooks = [(name, partial(backward_hook, hook_name=name)) for name in hook_names]
    
    # No hooks forward pass
    no_hook_grad_dict = get_grad_dict(get_loss(model.forward(**input)))

    # Forward pass with hooks
    hook_grad_dict = get_grad_dict(get_loss(model.run_with_hooks(**input, bwd_hooks=hooks)))

    for name, param in model.named_parameters():
        #we check that the gradients are not the same
        #TODO make a better test
        assert not torch.allclose(no_hook_grad_dict[name], hook_grad_dict[name]), f"{name} grads are the same but they should be different"
        
@pytest.mark.parametrize("module, input", get_combined_cases())
def test_unwrap_works(
    module: T, 
    input : Dict[str, torch.Tensor]
):
    model_pre = module
    pre_named_modules = [(name, type(module)) for name, module in model_pre.named_modules()]
    wrapped_model = auto_hook(model_pre)
    unwrapped_model = wrapped_model.unwrap()
    post_named_modules = [(name, type(module)) for name, module in unwrapped_model.named_modules()]
    assert pre_named_modules == post_named_modules, f"Expected {pre_named_modules}, got {post_named_modules}"


@pytest.mark.parametrize("module, input", get_combined_cases())
def test_to_works(
    module: T, 
    input : Dict[str, torch.Tensor]
):
    print("module parameters", list(module.named_parameters()))
    model = auto_hook(module.to(torch.float32))
    model.to(torch.float16)
    try:
        assert next((model.parameters())).dtype == torch.float16, f"Expected dtype '{torch.float16}', got {model.linear.weight.dtype}"
    except StopIteration:
        raise ValueError(f"No parameters found in the model {list(model.named_parameters())}")

@pytest.mark.parametrize("module, input", get_combined_cases())
def test_wrapping(
    module: T, 
    input : Dict[str, torch.Tensor]
):
    model_pre = module
    pre_named_parameters = [(name, param.clone().detach()) for name, param in model_pre.named_parameters()]
    model = auto_hook(model_pre)

    post_named_parameters = list(model.named_parameters())

    incorrect_elms = []
    for pre_name, pre_param in pre_named_parameters:
        found = False
        for post_name, post_param in post_named_parameters:
            if pre_name == post_name and torch.equal(pre_param, post_param):
                found = True
                break
        if not found:
            incorrect_elms.append((pre_name, pre_param))

    assert len(incorrect_elms) == 0, f"These elements should be in post_named_parameters: {incorrect_elms}, but found only {[name for name, _ in post_named_parameters]}"

@pytest.mark.parametrize("module, input", get_combined_cases())
def test_hook_fn_bwd_works(
    module: T, 
    input : Dict[str, torch.Tensor]
):
    model = auto_hook(module)
    generic_check_hook_fn_bwd_works(model, input)

@pytest.mark.parametrize("module, _", get_combined_cases())
def test_check_all_hooks(
    module: T, 
    _
):
    model = auto_hook(module)
    generic_check_all_hooks(model)

@pytest.mark.parametrize("module, _", get_combined_cases())
def test_check_unwrap_works(
    module: T, 
    _,
):
    model = auto_hook(module)
    unwrapped = model.unwrap()
    assert unwrapped == module, f"Unwrapped {unwrapped} is not the same as the original {module}"
    
@pytest.mark.parametrize("module, _ ", get_combined_cases())
def test_duplicate_hooks(
    module: T, 
    _
):
    model = auto_hook(module)
    hooks = [hook_name for hook_name, _ in model.hook_dict.items()]
    assert len(hooks) == len(set(hooks)), f"Duplicate hooks: {hooks}, hooks: {hooks} duplicates: {get_duplicates(hooks)}"

@pytest.mark.parametrize("module, _ ", get_combined_cases())
def test_generate_expected_hookpoints(
    module: T, 
    _, 
):
    no_hook_expected = generate_expected_hookpoints(module)
    hook_expected_2 = generate_expected_hookpoints(auto_hook(module))

    diff1 = list(set(no_hook_expected) - set(hook_expected_2)) 
    diff2 = list(set(hook_expected_2) - set(no_hook_expected))
    assert set(no_hook_expected) == set(hook_expected_2), f"Expected hookpoints do not match: {no_hook_expected} != {hook_expected_2} diff1: {diff1} diff2: {diff2}"

@pytest.mark.parametrize("module, _ ", get_combined_cases())
def test_hook_point_name(
    module: T, 
    _
):
    model = auto_hook(module)
    for name, hook_point in model.hook_dict.items():
        assert hook_point.name is not None, f"Hook point {name} has no name"

@pytest.mark.parametrize("module, _ ", get_combined_cases())
def test_auto_hook_module_naming(
    module: T, 
    _
):
    modules = [(name, module.__class__.__name__) for name, module in module.named_modules()]
    hooked_model = auto_hook(module)
    auto_modules = [(name, module.__class__.__name__) for name, module in hooked_model.named_modules() if not isinstance(module, HookPoint)]
    assert len(modules) == len(auto_modules), f"Modules length do not match: {len(modules)} != {len(auto_modules)}"

    for (name1, module_name1), (name2, module_name2) in zip(modules, auto_modules):
        assert name1 == name2, f"Names do not match: {name1} != {name2}"
        print(module_name1, module_name2)
        assert module_name1 == module_name2, f"Modules class names do not match: {module_name1} != {module_name2}"

def run_hook_test(model, hook_name, input_data):
    hook_called = {'value': False}
    
    def hook(x, hook=None, hook_name=None):
        hook_called['value'] = True
        return x
    
    model.run_with_hooks(**input_data, fwd_hooks=[(hook_name, hook)])
    
    assert hook_called['value'], f"{hook_name} was not called after auto_hook was applied"
    assert hook_name in model.hook_dict, f"{hook_name} is not in the hook_dict"
    print(f"TEST PASSED for {hook_name}")

def test_manual_hook_point_decorator():
    '''Tests if the auto_hook decorator works with manual hook point'''
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
    
    model = TestModel()
    run_hook_test(model, 'relu_hook_point', {'x': torch.randn(10)})

def test_manual_hook_point_instance():
    '''Tests if auto_hook works with pre-existing hook points in HookedTransformer'''
    model = HookedTransformer(hooked_transformer_cfg)
    hooked_model = auto_hook(model)
    
    input_kwargs = {
        'input': torch.randint(0, 10, (1, 10)), 
        'tokens': torch.randint(0, 2, (1, 10))
    }
    run_hook_test(hooked_model, 'blocks.3.hook_attn_out', input_kwargs)
