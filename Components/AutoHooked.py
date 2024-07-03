from typing import List
from torch import nn
from transformer_lens.hook_points import HookPoint, HookedRootModule
from typing import List, TypeVar
def print_hooks(x, hook=None, hook_name=None):
    print(f"NAME {hook_name} shape: {x.shape} x: {x}")
    return x

T = TypeVar("T")

def flatten_dict(d: dict[str, T], prefix: str = "") -> dict[str, T]:
    result = {}
    for k, v in d.items():
        new_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(flatten_dict(v, new_key))
        else:
            result[new_key] = v
    return result

class WrappedModule(nn.Module):
    def __init__(self, module, hook_point):
        super().__init__()
        self.__dict__.update(module.__dict__)
        self._original_forward = module.forward
        self.hook_point = hook_point

    def __repr__(self):
        return f"WrappedModule({self._original_forward.__self__.__class__.__name__})"

    def forward(self, *args, **kwargs):
        return self.hook_point(self._original_forward(*args, **kwargs))

class AutoHookedRootModule(HookedRootModule):
    '''
    This class automatically builds hooks for all modules that are not hooks.
    NOTE this does not mean all edges in the graph are hooked only that the outputs of the modules are hooked.
    for instance torch.softmax(x) is not hooked but self.softmax(x) would be
    '''
    def __init__(self):
        super().__init__()
        self.hook_dict = {}
        self.is_initialized = False

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def setup(self, filter: List[str] = []):
        self._initialize_and_wrap_hooks(filter=filter)
        super().setup()
        self.is_initialized = True

    def _initialize_and_wrap_hooks(
        self, 
        filter: List[str] = [], 
    ):
        change_dict = {}
        for name, module in list(self.named_children()):
            if isinstance(module, HookPoint) and not hasattr(self, f"hook_{name}"):
                raise ValueError(f"HookPoint {name} is not hooked but should be. Don't use hook_{name} as name if not with HookPoint")

            if name in filter or isinstance(module, HookPoint):
                continue
            
            if isinstance(module, nn.Module):
                hook_name = f"hook_{name}"
                wrapped_module = self._wrap_module(module, hook_name)
                change_dict[name] = wrapped_module
                setattr(self, name, wrapped_module)

                if isinstance(module, AutoHookedRootModule):
                    
                    # Preserve hooks of inner AutoHookedRootModule
                    wrapped_module.module = self.hook_forward(module, hook_name)

        # Apply changes after iteration
        for name, value in change_dict.items():
            if name not in self.__dict__:
                setattr(self, name, value)


    def state_dict(self, *args, **kwargs):
        # Override state_dict to avoid recursion
        return nn.Module.state_dict(self, *args, **kwargs)


    def unwrap_module(self, module: WrappedModule) -> nn.Module:
        for attr, value in module.__dict__.items():
            if isinstance(value, WrappedModule):
                self.unwrap_module(value)
            elif isinstance(value, HookPoint):
                pass
            else:
                setattr(module, attr, value)
    
    def unwrap_model(self):
        for name, module in self.named_children():
            if isinstance(module, WrappedModule):
                self.unwrap_module(module)

    def list_all_hooks(self):
        return self.hook_dict.items()

    def _wrap_module(self, module: nn.Module, hook_name: str) -> nn.Module:
        hook_point = HookPoint()
        return WrappedModule(module, hook_point)

    def hook_forward(self, module: 'AutoHookedRootModule', hook_name: str) -> 'AutoHookedRootModule':
        original_forward = module.forward

        def wrapped_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            print(f"hook_forward {hook_name} result: {result}")
            return getattr(self, hook_name)(result)

        module.forward = wrapped_forward
        return module
    