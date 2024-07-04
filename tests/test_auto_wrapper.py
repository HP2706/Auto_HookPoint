from typing import List, Tuple
from Components.AutoHooked import AutoHookedRootModule, WrappedModule, auto_hooked
from transformer_lens.hook_points import HookPoint
from test_model import BaseTransformerConfig, VanillaTransformerBlock
import torch.nn as nn

def get_hook_names(hook_list : List[Tuple[str, str]]):
    return [tup[0] for tup in hook_list]

def check_hook_names(hook_list : List[Tuple[str, str]], target_names : List[str]):
    hook_names = get_hook_names(hook_list)
    for name in target_names:
        assert name in hook_names, f"{name} not in hook_names {hook_names}"

def check_hook_types(hook_list : List[Tuple[str, str]]):
    assert all(isinstance(t, HookPoint) for t in [tup[1] for tup in hook_list])

class Test2(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner1 = nn.Linear(10, 10)

class Test1(nn.Module):
    def __init__(self):
        super().__init__()
        self.bla = nn.ModuleList([Test2(), Test2()])

def test_basic_auto_hooked():
    model = auto_hooked(Test2)()
    target_names = ['inner1.hook_point']
    hook_list = model.list_all_hooks()
    check_hook_names(hook_list, target_names)
    check_hook_types(hook_list)

def test_nested_modules_auto_hooked():
    model = auto_hooked(Test1)()
    target_names = ['bla.0.hook_point', 'bla.1.hook_point', 'bla.0.inner1.hook_point', 'bla.1.inner1.hook_point']
    hook_list = model.list_all_hooks()
    check_hook_names(hook_list, target_names)
    check_hook_types(hook_list)


class ModelTest(nn.Module):
    def __init__(self, cfg: BaseTransformerConfig):
        super().__init__()
        cfg = BaseTransformerConfig(
            d_model = 128,
            n_layers = 3,
            num_heads = 4,
            is_training=True,
        )
        self.bla = nn.ModuleList([VanillaTransformerBlock(cfg)])


def generate_expected_hookpoints(model, prefix=''):
    expected_hooks = []
    
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        
        # Add hook_point for the current module
        if not full_name.endswith('.hook_point'):
            expected_hooks.append(f"{full_name}.hook_point")        

        if isinstance(module, nn.ModuleList):
            for i, child in enumerate(module):
                expected_hooks.extend(generate_expected_hookpoints(child, f"{full_name}.{i}"))
        
        # Recursively process child modules
        expected_hooks.extend(generate_expected_hookpoints(module, full_name))
    
    return expected_hooks

# Usage example
def test_auto_hooked_model_nested():
    cfg = BaseTransformerConfig(
        d_model = 128,
        n_layers = 3,
        num_heads = 4,
        is_training=True,
    )
    model = auto_hooked(ModelTest)(cfg)
    expected_hookpoints = generate_expected_hookpoints(model)
    print(expected_hookpoints)

    # Compare with actual hookpoints
    actual_hookpoints = [name for name, _ in model.list_all_hooks()]
    print(actual_hookpoints)

    # Find missing hookpoints
    missing_hookpoints = set(expected_hookpoints) - set(actual_hookpoints)
    print("\nMissing hookpoints:")

    if missing_hookpoints:
        raise ValueError(
            f"Missing hookpoints: {missing_hookpoints} \n\n"
            f"Expected hookpoints: {expected_hookpoints} \n\n"
            f"Actual hookpoints: {actual_hookpoints} \n\n"
        )

