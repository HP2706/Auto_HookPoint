import inspect
import warnings
from typing import Any, Callable, List, Type, TypeVar
import torch.nn as nn
from AutoHook.hook import HookedModule, auto_hook
from .tests import test_auto_hook

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
    test_functions = get_test_functions(test_auto_hook)
    for test_func in test_functions:  
        model_instance = model(**init_kwargs)
        try:
            test_func(model_instance, input_kwargs)      

        except Exception as e:
            message = f"Test failed: {test_func.__name__} with error: {str(e)}"
            if strict:
                raise Exception(message)  # Re-raise the exception if strict mode is enabled
            else:
                warnings.warn(message) 

    return auto_hook(model(**init_kwargs))

def get_test_functions(module) -> List[Callable]:
    return [
        getattr(module, name) for name, func in inspect.getmembers(module)
        if inspect.isfunction(func) and name.startswith('test_') and name not in [
            'test_HookedParameter_unwrap', 'test_HookedParameter_hook' , 'test_check_auto_hook'
        ]
    ]