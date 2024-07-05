from typing import Iterable, Tuple, Union
from transformer_lens.hook_points import HookPoint
import warnings
from torch import nn
import inspect

def iterate_module(
    module : Union[nn.ModuleList, nn.Sequential, nn.ModuleDict]
) -> Iterable[Union[Tuple[str, nn.Module], Tuple[int, nn.Module]]]:
    if isinstance(module, (nn.ModuleList, nn.Sequential)):
        return enumerate(module)
    elif isinstance(module, nn.ModuleDict):
        return module.items()
   


def has_implemented_forward(module : nn.Module):
    '''
    this is for filtering torch modules by whether they have forward method
    note ModuleDict and ModuleList will have hasattr and callable 
    but you are not supposed to call forward on them.
    '''
    if not hasattr(module, "forward") or not callable(module.forward):
        return False
    
    try:
        source = inspect.getsource(module.forward)
        value = "raise NotImplementedError" not in source and "_forward_unimplemented" not in source
    except (OSError, TypeError):
        # If we can't get the source (e.g., built-in or C extension), assume it's implemented
        value = True
    
    if not value and not isinstance(module, (nn.ModuleList, nn.ModuleDict)):
        warnings.warn(
            f"Module {module} has forward method but"
            f"it is not implemented and is not a container module"
        )

    return value
