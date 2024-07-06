from inspect import isclass
from Components.AutoHooked import WrappedClass, WrappedInstance, auto_hooked
from functools import partial
from typing import Any, AnyStr, Optional, Type, TypeVar, Union, cast
import torch
import torch.nn as nn
from test_utils import generate_expected_hookpoints
from abc import ABC, abstractmethod

T = TypeVar("T", bound=nn.Module)

def check_auto_hooked(
    model : Union[T, Type[T]],
    kwargs : dict[str, Any] = {},
    model_init_kwargs : Optional[dict[str, Any]] = None
) -> str:#WrappedInstance[T]:
    '''
    this class will run the wrapped model on an input and check each hook works as expected
    the auto_hooked function might not fully support the wrapping of any torch.module so to catch edge
    cases this function allows the user to check if some hooks might not work and whether this is detrimental
    '''
    if len(kwargs) > 1:
        raise ValueError("This function only supports one input tensor at the moment")
    elif len(kwargs) == 0:
        raise ValueError("No content in kwargs, this is not allowed")

    hooked_model = auto_hooked(model)
    if isclass(model):
        if model_init_kwargs is None:
            raise ValueError("model_init_kwargs must be provided if you pass a class to this function")
        hooked_model = hooked_model(**model_init_kwargs)
    
    hooked_model = cast(WrappedInstance[T], hooked_model)
    found_hook_names = [hook_name for hook_name, _ in hooked_model.list_all_hooks()]
    expected_hook_names = generate_expected_hookpoints(hooked_model)

    missing_hooks = list(set(expected_hook_names) - set(found_hook_names))
    extra_hooks = list(set(found_hook_names) - set(expected_hook_names))

    base_message = (
    f"Length of hook_names: {len(found_hook_names)} does not match",
    f"Length of expected_hook_names: {len(expected_hook_names)}",
    f"Hooks missing from auto_wrapped model: {missing_hooks}",
    f"Extra hooks in auto_wrapped model: {extra_hooks}"
    )

    if len(found_hook_names) != len(expected_hook_names):
        print("The number of hooks in the auto_wrapped model doesn't match the expected number.")
        if len(found_hook_names) < len(expected_hook_names):
            print("The auto_wrapped model is missing hooks:")
        else:
            print("The auto_wrapped model has extra hooks:")
        print(*base_message, sep="\n")

    #NOW we test whether the hooks are also working
    global_counter = {'value' : 0, 'hook_list' : []}

    def print_shape(x, hook=None, hook_name=None):
        global_counter['value'] += 1
        global_counter['hook_list'].append(hook_name)
        return x

    hooked_model.run_with_hooks(
        #idx = torch.randint(1, 10, (1, 128)),
        **kwargs,
        fwd_hooks=[(name, partial(print_shape, hook_name=name)) for name in found_hook_names]
    )
    active_hooks = global_counter['hook_list']
    missing_calls = set(expected_hook_names) - set(active_hooks)
    extra_calls = set(active_hooks) - set(expected_hook_names)


    if len(active_hooks) < len(found_hook_names):
        not_called = [hook for hook in active_hooks if hook not in found_hook_names]
        print(
            "the number of hook calls does not match", 
            "the number of hooks that should be called are less than the actual number of hooks called",
            f"{len(not_called)} hooks were not called although they are in the mod_dict",
            f"the concerned hooks are {not_called}"
        )
    elif missing_calls:
        print("these hooks were not called, consider whether this could be a problem", missing_calls)
    elif extra_calls:
        extra_call_counts = {hook: active_hooks.count(hook) for hook in extra_calls}
        print(
            f"These hooks were called more than once. Is this intended?",
            *[f"{hook}: {count}" for hook, count in extra_call_counts.items()]
        )

    return hooked_model

