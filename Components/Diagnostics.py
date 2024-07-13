import inspect
from tests import test_auto_wrapper
from typing import Any, Callable, List, Optional, Type, TypeVar, Union, cast
import torch.nn as nn
from Components.AutoHooked import HookedInstance, auto_hook
import torch

T = TypeVar("T", bound=nn.Module)

def check_auto_hook(
    model_type: Type[T],
    input_shape : Union[torch.Size, List[int]],
    kwargs: dict[str, Any] = {},
) -> HookedInstance[T]:
    assert inspect.isclass(model_type), "model_type must be a class"

    #iterate over both the instance and the
    #NOTE we check that the tests pass whether 
    #it is autohooked from an instance or a cls
    
    #Run all test functions from test_auto_wrapper
    test_functions = get_test_functions(test_auto_wrapper)
    for test_func in test_functions:
        print("model", model_type, "test_func", test_func)
        #try:
        test_func(model_type, input_shape, kwargs)
        print(f"Test passed: {test_func.__name__}")
        #except Exception as e:
        #    print(f"Test failed: {test_func.__name__}")
        #    print(f"Error: {str(e)}") 
    model = auto_hook(model_type(**kwargs))
    return model

def get_test_functions(module) -> List[Callable]:
    return [
        getattr(module, name) for name, func in inspect.getmembers(module)
        if inspect.isfunction(func) and name.startswith('test_')
    ]