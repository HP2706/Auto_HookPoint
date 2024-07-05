from torch import nn
from utils import has_implemented_forward, iterate_module
from transformer_lens.hook_points import HookPoint
from Components.AutoHooked import WrappedInstance
from typing import Union


def generate_expected_hookpoints(model : Union[WrappedInstance, nn.Module],  prefix=''):
    expected_hooks = []
    if isinstance(model, WrappedInstance):
        model = model._module

    for name, module in model.named_children():

        if isinstance(module, HookPoint):
            # We do not allow for a HookPoint to be a child of another HookPoint
            continue

        full_name = f"{prefix}.{name}" if prefix else name
        if not full_name.endswith('.hook_point') and has_implemented_forward(module):
            expected_hooks.append(f"{full_name}.hook_point")  

        if isinstance(module, (nn.ModuleList, nn.Sequential, nn.ModuleDict)):
            for key, child in iterate_module(module):
                expected_hooks.extend(generate_expected_hookpoints(child, f"{full_name}.{key}"))
        expected_hooks.extend(generate_expected_hookpoints(module, full_name))

    return expected_hooks

