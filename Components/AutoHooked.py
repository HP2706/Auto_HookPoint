from __future__ import annotations
from inspect import isclass
from torch import nn
from transformer_lens.hook_points import HookPoint, HookedRootModule
from typing import (
    TypeVar, 
    Generic, 
    Union, 
    Type, 
    Any, 
    Callable, 
    get_type_hints, 
    ParamSpec, 
    Optional, 
    Set, 
    TypeVar, 
    Type, 
    Union, 
    cast, 
    overload
)
from inspect import isclass
import functools
from torch.nn.modules.module import Module


T = TypeVar('T', bound=nn.Module)
P = ParamSpec('P')
R = TypeVar('R')


class WrappedClass(Generic[T]):
    def __init__(self, module_class: Type[T]) -> T: # type: ignore
        self.module_class = module_class

    def __call__(self, *args: Any, **kwargs: Any) -> WrappedInstance[T]:
        instance = self.module_class(*args, **kwargs)
        return auto_hooked(instance)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.module_class, name)

    def unwrap_cls(self) -> Type[T]:
        '''recursively unwraps the module class'''
        print("unwrap cls called")
        for attr, value in self.module_class.__dict__.items():
            if isinstance(value, WrappedClass):
                setattr(self.module_class, attr, value.unwrap_cls())
            elif isinstance(value, type) and issubclass(value, nn.Module):
                if isinstance(value, WrappedClass):
                    setattr(self.module_class, attr, value.unwrap_cls())
                else:
                    wrapped_value = auto_hooked(value)
                    if isinstance(wrapped_value, WrappedClass):
                        setattr(self.module_class, attr, wrapped_value.unwrap_cls())
        return self.module_class

@overload
def auto_hooked(module_or_class: Type[T]) -> WrappedClass[T]: ...

@overload
def auto_hooked(module_or_class: T) -> WrappedInstance[T]: ...

def auto_hooked(module_or_class: Union[T, Type[T]]) -> Union[WrappedInstance[T], WrappedClass[T]]:
    '''
    This function wraps either a module instance or a module class and returns a type that
    preserves the original module's interface plus an additional unwrap method.
    '''
    if isclass(module_or_class):
        return WrappedClass(module_or_class)
    else:
        wrapped = WrappedInstance(module_or_class) # type: ignore
        #NOTE we set the unwrap method to just return module_or_class
        wrapped.unwrap = lambda: module_or_class # type: ignore
        return cast(WrappedInstance[T], wrapped)

class WrappedInstance(HookedRootModule, Generic[T]):
    def __init__(self, module: T):
        super().__init__()
        # NOTE we need to name it in this way to not 
        # to avoid infinite regress and override

        
        self._module = module
        self.hook_point = HookPoint()
        self._create_forward()
        self._wrap_submodules()
        self.setup()

    def __iter__(self):
        return iter(self._module)

    def new_attr_fn(self, name: str) -> Any:
        return getattr(self._module, name)

    #NOTE we override the nn.Module implementation to use _module only
    def named_modules(self, memo: Set[Module] | None = None, prefix: str = '', remove_duplicate: bool = True):
        #NOTE BE VERY CAREFUL HERE
        
        if memo is None:
            memo = set()

        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._module.named_children():
                if module not in memo:
                    submodule_prefix = prefix + ('.' if prefix else '') + name
                    if isinstance(module, WrappedInstance):
                        yield from module.named_modules(memo, submodule_prefix)
                    else:
                        yield submodule_prefix, module
                        if hasattr(module, 'named_modules'):
                            yield from module.named_modules(memo, submodule_prefix)

            if hasattr(self, 'hook_point'):
                hook_point_prefix = prefix + ('.' if prefix else '') + 'hook_point'
                yield hook_point_prefix, self.hook_point

    def unwrap_instance(self) -> T:
        for name, submodule in self._module.named_children():
            if isinstance(submodule, (nn.ModuleList, nn.Sequential)):
                unwrapped_container = type(submodule)()
                for m in submodule:
                    unwrapped_container.append(m.unwrap_instance() if isinstance(m, WrappedInstance) else m)
                setattr(self._module, name, unwrapped_container)
            elif isinstance(submodule, nn.ModuleDict):
                unwrapped_container = type(submodule)()
                for key, m in submodule.items():
                    unwrapped_container[key] = m.unwrap_instance() if isinstance(m, WrappedInstance) else m
                setattr(self._module, name, unwrapped_container)
            elif isinstance(submodule, WrappedInstance):
                setattr(self._module, name, submodule.unwrap_instance())
        return self._module        

    def _wrap_submodules(self):
        for name, submodule in self._module.named_children():
            if isinstance(submodule, (nn.ModuleList, nn.Sequential)):
                wrapped_container = type(submodule)() #initialize the container
                for i, m in enumerate(submodule):
                    wrapped_container.append(auto_hooked(m))
                setattr(self._module, name, wrapped_container)
            elif isinstance(submodule, nn.ModuleDict):
                wrapped_container = type(submodule)()
                for key, m in submodule.items():
                    wrapped_container[key] = auto_hooked(m)
                setattr(self._module, name, wrapped_container)
            else:
                setattr(self._module, name, auto_hooked(submodule))

    def _create_forward(self):
        original_forward = self._module.forward
        original_type_hints = get_type_hints(original_forward)

        @functools.wraps(original_forward)
        def new_forward(*args: Any, **kwargs: Any) -> Any:
            return self.hook_point(original_forward(*args, **kwargs))

        new_forward.__annotations__ = original_type_hints
        self.forward = new_forward  # Assign to instance, not class

    def list_all_hooks(self):
        return [(hook, hook_point) for hook, hook_point in self.hook_dict.items()] 
    