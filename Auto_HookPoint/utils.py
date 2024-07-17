from typing import Iterable, Tuple, Union
from torch import nn

def iterate_module(
    module : Union[nn.ModuleList, nn.Sequential, nn.ModuleDict]
) -> Iterable[Union[Tuple[str, nn.Module], Tuple[int, nn.Module]]]:
    if isinstance(module, (nn.ModuleList, nn.Sequential)):
        return enumerate(module)
    elif isinstance(module, nn.ModuleDict):
        return module.items()
