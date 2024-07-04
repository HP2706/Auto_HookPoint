from torch import nn
import torch
from transformer_lens.hook_points import HookPoint, HookedRootModule
from typing import List, TypeVar, Type

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

    def unwrap(self) -> nn.Module:
        unwrapped_module = self._original_forward.__self__
        print("hook_point in self.__dict__", "hook_point" in self.__dict__)
        unwrapped_module.__dict__.update(self.__dict__)
        return unwrapped_module

    def forward(self, *args, **kwargs):
        return self.hook_point(self._original_forward(*args, **kwargs))

class AutoHookedRootModule(HookedRootModule):
    '''
    This class automatically builds hooks for all modules that are not hooks.
    NOTE this does not mean all edges in the graph are hooked only that the outputs of the modules are hooked.
    for instance torch.softmax(x) is not hooked but self.softmax(x) would be
    '''
    def __init__(self, auto_setup=True):
        super().__init__()
        self.hook_dict = {}
        self._setup_called = False
        self._init_finished = False
        self.auto_setup = auto_setup

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._init_finished = True
            if not self._setup_called:
                if self.auto_setup:
                    self.setup()
                else:
                    raise ValueError(
                        "setup() must be explicitly called in the",
                        f"__init__ method of {cls.__name__} when auto_setup is False"
                    )

        cls.__init__ = new_init


    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def setup(self, filter: List[str] = []):
        self._initialize_and_wrap_hooks(filter=filter)
        super().setup()
        self._setup_called = True

    def _initialize_and_wrap_hooks(self, filter: List[str] = [], prefix: str = ""):
        change_dict = {}
        for name, module in list(self.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            
            if not isinstance(module, HookPoint) and 'hook' in name:
                raise ValueError(f"HookPoint {full_name} is not hooked but should be. Don't use hook_{name} as name if not with HookPoint")

            if full_name in filter or isinstance(module, HookPoint):
                continue

            elif isinstance(module, nn.Module):
                hook_name = f"hook_{full_name}"
                wrapped_module = self._wrap_module(module, hook_name)
                change_dict[name] = wrapped_module
                self.hook_dict[hook_name] = wrapped_module.hook_point
                
                # Recursively wrap nested modules
                if isinstance(module, (AutoHookedRootModule, WrappedModule)):
                    if hasattr(module, '_initialize_and_wrap_hooks'):
                        module._initialize_and_wrap_hooks(filter, full_name)
            
            # Apply changes after iteration
            for name, value in change_dict.items():
                setattr(self, name, value)

    def state_dict(self, *args, **kwargs):
        # Override state_dict to avoid recursion
        return nn.Module.state_dict(self, *args, **kwargs)



    def unwrap_module(self, module: WrappedModule) -> nn.Module:
        for attr, value in module.__dict__.items():
            if isinstance(value, WrappedModule):
                value = value.unwrap()
            elif isinstance(value, HookPoint):
                pass
            else:
                setattr(module, attr, value)
    
    def unwrap_model(self):
        '''unwraps model and removes hook_point'''
        for name, module in list(self.named_children()):  # Use list() to avoid RuntimeError
            if isinstance(module, WrappedModule):
                print(f"Unwrapping {name}, module: {module}")
                unwrapped = module.unwrap()
                setattr(self, name, unwrapped)
                if isinstance(unwrapped, AutoHookedRootModule):
                    unwrapped.unwrap_model()  # Recursively unwrap nested AutoHookedRootModules

            if hasattr(unwrapped, 'hook_point'):
                delattr(unwrapped, 'hook_point')

    def list_all_hooks(self):
        return self.hook_dict.items()

    def _wrap_module(self, module: nn.Module, hook_name: str) -> nn.Module:
        hook_point = HookPoint()
        return WrappedModule(module, hook_point)

    def hook_forward(self, module: 'AutoHookedRootModule', hook_name: str) -> 'AutoHookedRootModule':
        original_forward = module.forward

        def wrapped_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            return getattr(self, hook_name)(result)

        module.forward = wrapped_forward
        return module
    
T = TypeVar('T', bound=nn.Module)

def auto_hooked(cls: Type[T]) -> Type[T]:
    class Wrapped(AutoHookedRootModule):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.model = cls(*args, **kwargs)
            
            for attr_name in dir(self.model):
                if not attr_name.startswith('__'):
                    attr_value = getattr(self.model, attr_name)
                    if isinstance(attr_value, nn.Module):
                        if type(attr_value) in [nn.ModuleList, nn.ModuleDict, nn.Sequential]:
                            print(f"wrapping {attr_name} with {type(attr_value)}")
                            for key, item in attr_value.items() if isinstance(attr_value, nn.ModuleDict) else enumerate(attr_value):
                                if isinstance(key, int):
                                    key = f"{type(item).__name__}_{key}"
                                
                                print(f"wrapping {key}")
                                wrapped_attr = auto_hooked_with_args(type(item), item)
                                print(f"wrapped {key}")
                                setattr(self, f"{attr_name}_{key}", wrapped_attr)
                        else:   
                            # Wrap nested nn.Modules with AutoHookedRootModule
                            wrapped_attr = auto_hooked_with_args(type(attr_value), attr_value)
                            setattr(self, attr_name, wrapped_attr)
                    elif not hasattr(self, attr_name):
                        #print(f"not nn.Module {attr_name} to {attr_value}")
                        setattr(self, attr_name, attr_value)
            
            # Remove the model attribute to avoid duplication
            del self.model
            self.setup()

        def forward(self, *args, **kwargs):
            return cls.forward(self, *args, **kwargs)
        
    return Wrapped

def auto_hooked_with_args(cls: Type[nn.Module], instance: nn.Module) -> nn.Module:
    print(f"auto_hooked_with_args called with cls: {cls} and instance: {instance}")
    class WrappedModule(AutoHookedRootModule):
        def __init__(self):
            super().__init__()
            # Copy all attributes from the original instance
            for attr_name, attr_value in instance.__dict__.items():
                if not attr_name.startswith('__'):
                    setattr(self, attr_name, attr_value)
            self.setup()

        def forward(self, *args, **kwargs):
            return cls.forward(self, *args, **kwargs)

    return WrappedModule()