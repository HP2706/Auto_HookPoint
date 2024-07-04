from torch import nn
import torch
from transformer_lens.hook_points import HookPoint, HookedRootModule
from typing import List, Optional, TypeVar, Type, Union

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
        unwrapped_module.__dict__.update(self.__dict__)
        return unwrapped_module

    def forward(self, *args, **kwargs):
        return self.hook_point(self._original_forward(*args, **kwargs))

    def _initialize_and_wrap_hooks(self, filter: List[str] = [], prefix: str = ""):
        module = self._original_forward.__self__
        if hasattr(module, '_initialize_and_wrap_hooks'):
            module._initialize_and_wrap_hooks(filter, prefix)
            self.hook_dict.update(module.hook_dict)
    
    def list_all_hooks(self):
        return self.hook_dict.items()

class AutoHookedRootModule(HookedRootModule):
    '''
    This class automatically builds hooks for all modules that are not hooks.
    NOTE this does not mean all edges in the graph are hooked only that the outputs of the modules are hooked.
    for instance torch.softmax(x) is not hooked but self.softmax(x) would be
    '''
    def __init__(self, auto_setup=True, filter: Optional[List[str]] = []):
        super().__init__()
        if auto_setup and filter is None:
            raise ValueError("filter must be provided if auto_setup is True")
        
        self.filter = filter
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
                    self.setup(filter=self.filter)
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
        children = list(self.named_children())
        for name, module in children:
            full_name = f"{prefix}.{name}" if prefix else name
            
            if not isinstance(module, HookPoint) and 'hook' in name:
                raise ValueError(f"HookPoint {full_name} is not hooked but should be. Don't use hook_{name} as name if not with HookPoint")

            if full_name in filter or isinstance(module, HookPoint):
                continue

            elif isinstance(module, nn.Module):
                if isinstance(module, (nn.ModuleList, nn.ModuleDict, nn.Sequential)):
                    for key, item in module.items() if isinstance(module, nn.ModuleDict) else enumerate(module):
                        wrapped_item = self._wrap_module(item, f"hook_{full_name}_{key}")
                        if isinstance(module, nn.ModuleDict):
                            module[key] = wrapped_item
                        elif isinstance(module, (nn.ModuleList, nn.Sequential)):
                            module[key] = wrapped_item
                        self.hook_dict[f"hook_{full_name}_{key}"] = wrapped_item.hook_point
                    
                    # Don't wrap the container module itself
                    change_dict[name] = module
                else:
                    hook_name = f"hook_{full_name}"
                    wrapped_module = self._wrap_module(module, hook_name)
                    change_dict[name] = wrapped_module
                    self.hook_dict[hook_name] = wrapped_module.hook_point
                
                # Recursively wrap nested modules
                if isinstance(module, (AutoHookedRootModule, WrappedModule)):
                    if hasattr(module, '_initialize_and_wrap_hooks'):
                        module._initialize_and_wrap_hooks(filter, full_name)
                    else:
                        raise ValueError(f"module: {module} has no _initialize_and_wrap_hooks")

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
        wrapped = WrappedModule(module, hook_point)
        if isinstance(module, AutoHookedRootModule):
            wrapped.hook_dict.update(module.hook_dict)
        return wrapped


    def hook_forward(self, module: 'AutoHookedRootModule', hook_name: str) -> 'AutoHookedRootModule':
        original_forward = module.forward

        def wrapped_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            return getattr(self, hook_name)(result)

        module.forward = wrapped_forward
        return module
    
T = TypeVar('T', bound=nn.Module)


def auto_hooked(cls_or_instance: Union[Type[T], T]) -> Union[Type[T], T]:    

    if isinstance(cls_or_instance, nn.Module):
        # Handle instance
        if isinstance(cls_or_instance, (nn.ModuleList, nn.ModuleDict, nn.Sequential)):
            for idx, item in enumerate(cls_or_instance):
                cls_or_instance[idx] = auto_hooked(item)
            return cls_or_instance
        else:
            return _wrap_instance(cls_or_instance)
    
    elif issubclass(cls_or_instance, AutoHookedRootModule):
        # Already an AutoHookedRootModule, return as is
        return cls_or_instance
    else:
        # Handle class
        return _wrap_class(cls_or_instance)

def _wrap_instance(instance: nn.Module) -> nn.Module:
    class WrappedModule(AutoHookedRootModule):
        def __init__(self):
            super().__init__()
            assert type(instance) not in [nn.ModuleList, nn.ModuleDict, nn.Sequential]
            #we dont want to wrap the container module itself

            for attr_name in dir(instance):
                if not attr_name.startswith('__'):
                    setattr(self, attr_name, getattr(instance, attr_name))

        def forward(self, *args, **kwargs):
            return type(instance).forward(self, *args, **kwargs)

    wrapped = WrappedModule()
    return wrapped

def _wrap_class(cls: Type[T]) -> Type[T]:
    class Wrapped(AutoHookedRootModule):
        def __init__(self, *args, **kwargs):
            super().__init__()
            assert getattr(self, 'COPY', None) is None, "COPY attribute already exists"
            self.COPY = cls(*args, **kwargs)
            
            for attr_name in dir(self.COPY):
                #filter out special attributes
                if not attr_name.startswith('__'): 
                    attr_value = getattr(self.COPY, attr_name)
                    if isinstance(attr_value, nn.Module):
                        wrapped_attr = auto_hooked(attr_value)
                        setattr(self, attr_name, wrapped_attr)
                    elif not hasattr(self, attr_name):
                        setattr(self, attr_name, attr_value)
            
            del self.COPY
            self.setup()

        def forward(self, *args, **kwargs):
            return cls.forward(self, *args, **kwargs)
    
    return Wrapped
