import os
import sys

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
import inspect
import warnings
from typing import Any, Callable, List, Type, TypeVar
import torch.nn as nn
from Auto_HookPoint.hook import HookedModule, auto_hook

from tests import test_auto_hook

# ... rest of the code remains unchanged ...

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

    failed_tests = []

    test_functions = get_test_functions(test_auto_hook)
    for test_func in test_functions:  
        model_instance = model(**init_kwargs)
        try:
            test_func(model_instance, input_kwargs)      
        except Exception as e:
            message = f"Test failed: {test_func.__name__} with error: {str(e)}"
            failed_tests.append(message)

    if failed_tests:
        str_format = '\n'.join(failed_tests)
        if strict:
            raise Exception(f"Tests failed: {str_format}")  # Re-raise the exception if strict mode is enabled
        else:
            warnings.warn(f"the following tests failed: {str_format}") 

    return auto_hook(model(**init_kwargs))

def get_test_functions(module) -> List[Callable]:
    return [
        getattr(module, name) for name, func in inspect.getmembers(module)
        if inspect.isfunction(func) and name.startswith('test_') and name not in [
            'test_HookedParameter_unwrap', 'test_HookedParameter_hook' , 'test_check_auto_hook'
        ]
    ]