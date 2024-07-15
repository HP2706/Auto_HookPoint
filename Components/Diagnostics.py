import inspect
from tests import test_auto_wrapper
from typing import Any, Callable, List, Optional, Type, TypeVar, Union, cast
import torch.nn as nn
from Components.AutoHooked import HookedModule, auto_hook
import torch

T = TypeVar("T", bound=nn.Module)

def check_auto_hook(
    model: Type[T],
    input_kwargs: dict[str, Any],
    init_kwargs : dict[str, Any] = {},
    strict : bool = False
) -> HookedModule[T]:

    #iterate over both the instance and the
    #NOTE we check that the tests pass whether 
    #it is autohooked from an instance or a cls
    
    #Run all test functions from test_auto_wrapper
    test_functions = get_test_functions(test_auto_wrapper)
    for test_func in test_functions:  
        model_instance = model(**init_kwargs)
        try:
            test_func(model_instance, input_kwargs)      
            print(f"Test passed: {test_func.__name__}")
        except Exception as e:
            if strict:
                raise  # Re-raise the exception if strict mode is enabled
            else:
                print(f"Test failed: {test_func.__name__}")
                print(f"Error: {str(e)}")
    return auto_hook(model(**init_kwargs))

def get_test_functions(module) -> List[Callable]:
    return [
        getattr(module, name) for name, func in inspect.getmembers(module)
        if inspect.isfunction(func) and name.startswith('test_') and name not in [
            'test_HookedParameter_unwrap', 'test_HookedParameter_hook' , 'test_check_auto_hook'
        ]
    ]