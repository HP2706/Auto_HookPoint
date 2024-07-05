from torch import nn
from utils import has_implemented_forward, iterate_module
from transformer_lens.hook_points import HookPoint
from Components.AutoHooked import WrappedInstance
from typing import Union
from collections import Counter

def get_duplicates(lst : list[str]) -> list[str]:
    return [item for item, count in Counter(lst).items() if count > 1]


def generate_expected_hookpoints(model : Union[WrappedInstance, nn.Module],  prefix='') -> list[str]:
    expected_hooks = set()
    if isinstance(model, WrappedInstance):
        model = model._module

    for name, module in model.named_children():

        if isinstance(module, HookPoint):
            # We do not allow for a HookPoint to be a child of another HookPoint
            continue

        full_name = f"{prefix}.{name}" if prefix else name
        if not full_name.endswith('.hook_point') and has_implemented_forward(module):
            expected_hooks.add(f"{full_name}.hook_point")  

        if isinstance(module, (nn.ModuleList, nn.Sequential, nn.ModuleDict)):
            for key, child in iterate_module(module):
                expected_hooks.update(generate_expected_hookpoints(child, f"{full_name}.{key}"))
        expected_hooks.update(generate_expected_hookpoints(module, full_name))

    return list(expected_hooks)

