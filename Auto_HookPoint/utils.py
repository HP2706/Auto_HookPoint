from typing import Union, Any, Callable
from torch import nn
from typing import Iterable, Tuple
import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def iterate_module(
    module : Union[nn.ModuleList, nn.Sequential, nn.ModuleDict]
) -> Iterable[Union[Tuple[str, nn.Module], Tuple[int, nn.Module]]]:
    if isinstance(module, (nn.ModuleList, nn.Sequential)):
        return enumerate(module)
    elif isinstance(module, nn.ModuleDict):
        return module.items()

def process_container_module(
    container: Union[nn.ModuleList, nn.Sequential, nn.ModuleDict],
    process_func: Callable[[Any], Any]
) -> Union[nn.ModuleList, nn.Sequential, nn.ModuleDict]:
    """
    Process a container module (ModuleList, Sequential, or ModuleDict) by applying a function to each submodule.

    Args:
        container (Union[nn.ModuleList, nn.Sequential, nn.ModuleDict]): The container module to process.
        process_func (Callable[[Any], Any]): A function to apply to each submodule.

    Returns:
        Union[nn.ModuleList, nn.Sequential, nn.ModuleDict]: A new container of the same type with processed submodules.

    Example:
        # For unwrapping:
        unhooked_container = process_container_module(
            submodule,
            lambda m: m.unwrap() if isinstance(m, HookedModule) else m
        )

        # For wrapping:
        hooked_container = process_container_module(
            submodule,
            lambda m: auto_hook(m)
        )
    """
    new_container = type(container)()
    for key_or_idx, m in iterate_module(container):
        processed_m = process_func(m)
        if isinstance(key_or_idx, str):
            new_container[key_or_idx] = processed_m  # type: ignore
        else:
            new_container.append(processed_m)
    return new_container