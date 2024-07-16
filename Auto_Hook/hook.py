from __future__ import annotations
from inspect import isclass
import warnings
from torch import nn
import torch
from transformer_lens.hook_points import HookPoint, HookedRootModule
from typing import (
    Generator,
    Optional,
    Type,
    TypeVar, 
    Generic, 
    Union, 
    Any, 
    get_type_hints, 
    Set, 
    cast, 
    overload,
)
import functools
from torch.nn.modules.module import Module

#these are modules where we will not iterate over their parameters
BUILT_IN_MODULES = [
    nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm , nn.LayerNorm,
    nn.RNN, nn.LSTM, nn.GRU, nn.RNNCell, nn.LSTMCell, nn.GRUCell, 
    # Add more built-in module types as needed
]

T = TypeVar('T', bound=nn.Module)
P = TypeVar('P', bound=nn.Parameter)
B = TypeVar('B', bound=Union[nn.Module, nn.Parameter])
class HookedClass(Generic[T]):    
    def __init__(self, module_class: Union[Type[T], Type[P]]):
        self.module_class = module_class

    @overload
    def __call__(self, *args: Any, **kwargs: Any) -> HookedModule[T]: ...
    
    @overload
    def __call__(self, *args: Any, **kwargs: Any) -> HookedParameter[P]: ...
    
    def __call__(self, *args: Any, **kwargs: Any) -> Union[HookedModule[T], HookedParameter[P]]:
        instance = self.module_class(*args, **kwargs)
        hooked = auto_hook(instance)
        if isinstance(hooked, HookedModule):
            return cast(HookedModule[T], hooked)
        else:
            return cast(HookedParameter[P], hooked)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.module_class, name)

    def unwrap(self) -> Type[T]:
        '''recursively unwraps the module class'''
        return cast(Type[T], self.module_class)

# Update the auto_hook function to return the correct type
@overload
def auto_hook(module_or_class: Type[T]) -> HookedClass[T]: ...

@overload
def auto_hook(module_or_class: T) -> HookedModule[T]:...

@overload
def auto_hook(module_or_class: P) -> HookedParameter[P]:...

@overload
def auto_hook(module_or_class: Type[P]) -> HookedClass[P]:...

def auto_hook(
    module_or_class: Union[Type[T], T, Type[P], P]
) -> Union[
        HookedModule[T], HookedClass[T], HookedParameter[P], HookedClass[P]
    ]:
    if isclass(module_or_class):
        return HookedClass(module_or_class)
    '''
    This function wraps either a module instance or a module class and returns a type that
    preserves the original module/Parameters interface but adds a hook_point to all nn.Module and select torch.nn.Parameter classes
    '''
    if isclass(module_or_class):
        return HookedClass(module_or_class)
    if isinstance(module_or_class, (HookedClass, HookedModule, HookedParameter)):
        warnings.warn(f"auto_hook is called with a {type(module_or_class).__name__} instance, returning the original")
        return module_or_class
    
    if isinstance(module_or_class, nn.Module):
        Hooked = HookedModule(module_or_class) # type: ignore
        #NOTE we set the unwrap method to just return module_or_class
        Hooked.unwrap = lambda: module_or_class # type: ignore
        Hooked = cast(HookedModule[T], Hooked)
    elif isinstance(module_or_class, (nn.Parameter, torch.Tensor)):
        Hooked = cast(HookedParameter[P], HookedParameter(module_or_class)) # type: ignore
    else:
        raise ValueError(
            f"Module type {type(module_or_class)} is not supported should, "
            "be one of nn.Module or nn.Parameter/torch.Tensor"
        )
    return Hooked

class HookedParameter(nn.Parameter, HookedRootModule, Generic[P]):
    def __init__(self, parameter : P):
        super().__init__()
        self.param = parameter
        self.hook_point = HookPoint()
        self._wrap_math_ops()
        self.setup()
    
    def _wrap_math_ops(self):
        math_ops = [
            '__add__', '__radd__', '__sub__', '__rsub__', '__mul__', '__rmul__',
            '__matmul__', '__rmatmul__', '__truediv__', '__rtruediv__', '__floordiv__',
            '__rfloordiv__', '__mod__', '__rmod__', '__pow__', '__rpow__', '__neg__', '__abs__'
        ]
        for op in math_ops:
            if hasattr(self, op):
                setattr(self.__class__, op, self._create_wrapped_op(op))

    def _create_wrapped_op(self, op_name):
        def wrapped_op(self, other):
            result = getattr(super(HookedParameter, self), op_name)(other)
            return self.hook_point(result)  # Apply the hook
        return wrapped_op

    def _apply_hook(self, x):
        return self.hook_point(x)

    def setup(self):
        self.mod_dict = {'param': self.param, 'hook_point': self.hook_point}
        self.hook_dict = {'hook_point': self.hook_point}

    def unwrap(self) -> P:
        return cast(P, nn.Parameter(self.data, requires_grad=self.requires_grad))

class HookedModule(HookedRootModule, Generic[T]):
    def __init__(self, module: T):
        super().__init__()
        # NOTE we need to name it in this way to not 
        # to avoid infinite regress and override
        self._module = module
        if not hasattr(self._module, 'hook_point'):
            self.hook_point = HookPoint()
        super().named_children()
        self._create_forward()
        self._wrap_submodules()
        self.setup()
    

    def setup(self):
        self.mod_dict = {}
        self.hook_dict = {}
        
        if hasattr(self, 'hook_point'):
            self.mod_dict['hook_point'] = self.hook_dict['hook_point'] = self.hook_point
        
        def add_module_and_params(module : nn.Module, prefix=''):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                self.mod_dict[full_name] = child
                
                if isinstance(child, HookedModule):
                    hook_point_name = f"{full_name}.hook_point"
                    
                    self.mod_dict[hook_point_name] = self.hook_dict[hook_point_name] = child.hook_point
                    add_module_and_params(child._module, full_name)
                elif isinstance(child, HookPoint):
                    self.hook_dict[full_name] = child
                else:
                    add_module_and_params(child, full_name)
            
            for name, param in module.named_parameters(recurse=False):
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(param, HookedParameter):
                    self.mod_dict[full_name] = param
                    
                    hook_point_name = f'{full_name}.hook_point'
                    self.mod_dict[hook_point_name] = self.hook_dict[hook_point_name] = param.hook_point

        add_module_and_params(self._module)

    def _named_modules(
        self, 
        memo: Optional[Set['Module']] = None, 
        prefix: str = '', 
        remove_duplicate: bool = True
    )-> Generator[Union[tuple[str, Module], Module], None, None]:
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._module._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                #print("yielding", list(module.named_modules(memo, submodule_prefix, remove_duplicate)))
                yield from module.named_modules(memo, submodule_prefix, remove_duplicate)

    #NOTE we override the nn.Module implementation to use _module only
    #NOTE this is not an ideal approach b

    def unwrap(self) -> T:
        '''
        This method recursively unwraps the HookedModule and returns the original module
        '''
        for name, submodule in self._module.named_children():
            if isinstance(submodule, (nn.ModuleList, nn.Sequential)):
                unHooked_container = type(submodule)()
                for m in submodule:
                    unHooked_container.append(m.unwrap() if isinstance(m, HookedModule) else m)
                setattr(self._module, name, unHooked_container)
            elif isinstance(submodule, nn.ModuleDict):
                unHooked_container = type(submodule)()
                for key, m in submodule.items():
                    unHooked_container[key] = m.unwrap() if isinstance(m, HookedModule) else m
                setattr(self._module, name, unHooked_container)
            elif isinstance(submodule, HookedModule):
                setattr(self._module, name, submodule.unwrap())
            elif isinstance(submodule, HookedParameter): 
                ##TODO this is not correct as submodule is not a module and will not be in named_children
                setattr(self._module, name, submodule.unwrap())
        return self._module        

    def _wrap_submodules(self):
        if not any(isinstance(self._module, built_in_module) for built_in_module in BUILT_IN_MODULES):
            #RECURSE is set to false to avoid getting all sub params
            for name, submodule in self._module.named_parameters(recurse=False):
                #print(f"wrapping submodule name {name}, submodule: {type(submodule)}")
                if isinstance(submodule, (nn.Parameter, torch.Tensor)):
                    #NOTE IT IS NOT EASY TO WRAP A HOOKPOINT AROUND NN.PARAMETER
                    #THIS IS CURRENTLY NOT SUPPORTED BUT WILL BE NEEDED TO SUPPORT
                    #NOTE we can still set the hookpoint, but it doesnt provide meaningful utility 
                    # as it is hard to wrap the output of NN.PARAMETER 
                    setattr(self._module, name, auto_hook(submodule))
                else:
                    raise ValueError(f"Submodule {name} is not a nn.Parameter or torch.Tensor")

        for name, submodule in self._module.named_children():
            if isinstance(submodule, HookPoint):
                continue

            elif isinstance(submodule, (nn.ModuleList, nn.Sequential)):
                Hooked_container = type(submodule)() #initialize the container
                for i, m in enumerate(submodule):
                    Hooked_container.append(auto_hook(m))

                setattr(self._module, name, Hooked_container)
            elif isinstance(submodule, nn.ModuleDict):
                Hooked_container = type(submodule)()
                for key, m in submodule.items():
                    Hooked_container[key] = auto_hook(m)
                setattr(self._module, name, Hooked_container)
            else:
                setattr(self._module, name, auto_hook(submodule))

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
    