from typing import List, Tuple
from Components.AutoHooked import AutoHookedRootModule, WrappedModule, auto_hooked
from transformer_lens.hook_points import HookPoint
from Models import BaseTransformerConfig, VanillaTransformerBlock
from utils import generate_expected_hookpoints
import torch.nn as nn
import torch
from functools import partial

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

    def forward(self, x):
        return self.inner1(x)

class Test1(nn.Module):
    def __init__(self):
        super().__init__()
        self.bla = nn.ModuleList([Test2(), Test2()])

    def forward(self, x):
        return self.bla(x)

def test_basic_auto_hooked():
    model = auto_hooked(Test2)()
    target_names = ['inner1.hook_point']
    hook_list = model.list_all_hooks()
    check_hook_names(hook_list, target_names)
    check_hook_types(hook_list)

def basic_hook_test(model):
    counter = {'value': 0} #GLOBAL state
    def print_shape(x, hook=None, hook_name=None):
        counter['value'] += 1
        return x

    hook_names = [hook_name for hook_name, _ in model.list_all_hooks()]
    model.run_with_hooks(
        x=torch.randn(1, 10),
        fwd_hooks=[(hook_name, partial(print_shape, hook_name=hook_name)) for hook_name in hook_names]
    )
    assert counter['value'] == len(hook_names), f"counter['value'] == len(hook_names), {counter['value']} == {len(hook_names)}"
    print("TEST PASSED")

def test_hooks_work_auto_hooked_instance():
    model = auto_hooked(Test2())
    basic_hook_test(model)

def test_hooks_work_auto_hooked_class():
    model = auto_hooked(Test2)()
    basic_hook_test(model)

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

if __name__ == "__main__":
    test_hooks_work()

