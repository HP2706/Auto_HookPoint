from __future__ import annotations
from inspect import isclass
import warnings
from torch import nn
import torch
from transformer_lens.hook_points import HookPoint, HookedRootModule
from typing import (
    Type,
    TypeVar, 
    Generic, 
    Union, 
    Any, 
    get_type_hints, 
    cast, 
    overload,
)
import functools
from Auto_HookPoint.utils import process_container_module

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
    '''
    A wrapper class for module classes that allows for automatic hooking of instances.
    '''
    
    def __init__(self, module_class: Union[Type[T], Type[P]]):
        '''
        Initialize the HookedClass with a module class.
        
        Args:
            module_class (Union[Type[T], Type[P]]): The class to be wrapped.
        '''
        self.module_class = module_class

    @overload
    def __call__(self, *args: Any, **kwargs: Any) -> HookedModule[T]: ...
    
    @overload
    def __call__(self, *args: Any, **kwargs: Any) -> HookedParameter[P]: ...
    
    def __call__(self, *args: Any, **kwargs: Any) -> Union[HookedModule[T], HookedParameter[P]]:
        '''
        Create an instance of the wrapped class and hook it.
        
        Returns:
            Union[HookedModule[T], HookedParameter[P]]: A hooked instance of the wrapped class.
        '''
        instance = self.module_class(*args, **kwargs)
        hooked = auto_hook(instance)
        if isinstance(hooked, HookedModule):
            return cast(HookedModule[T], hooked)
        else:
            return cast(HookedParameter[P], hooked)

    def __getattr__(self, name: str) -> Any:
        '''
        Delegate attribute access to the wrapped class.
        '''
        return getattr(self.module_class, name)

    def unwrap(self) -> Type[T]:
        '''
        Recursively unwrap the module class.
        
        Returns:
            Type[T]: The original, unwrapped module class.
        '''
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
    '''
    This function wraps either a module instance or a module class and returns a type that
    preserves the original module/Parameters interface but adds a hook_point to all nn.Module and select torch.nn.Parameter classes
    
    Args:
        module_or_class: The module or class to be hooked.
    
    Returns:
        A hooked version of the input module/parameter or class.
    '''
    if isclass(module_or_class):
        return HookedClass(module_or_class)

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
    '''
    A wrapper for nn.Parameter that adds a hook point and wraps any mathematical operations performed on it
    '''
    
    def __init__(self, parameter : P):
        '''
        Initialize the HookedParameter.
        
        Args:
            parameter (P): The parameter to be wrapped.
        '''
        super().__init__()
        self.param = parameter
        self.hook_point = HookPoint()
        self._wrap_math_ops()
        self.setup()
    
    def _wrap_math_ops(self):
        '''
        Wrap mathematical operations in a hook point
        '''
        math_ops = [
            '__add__', '__radd__', '__sub__', '__rsub__', '__mul__', '__rmul__',
            '__matmul__', '__rmatmul__', '__truediv__', '__rtruediv__', '__floordiv__',
            '__rfloordiv__', '__mod__', '__rmod__', '__pow__', '__rpow__', '__neg__', '__abs__'
        ]
        for op in math_ops:
            if hasattr(self, op):
                setattr(self.__class__, op, self._create_wrapped_op(op))

    def _create_wrapped_op(self, op_name):
        '''
        Create a wrapped version of a mathematical operation.
        
        Args:
            op_name (str): The name of the operation to be wrapped.
        
        Returns:
            function: The wrapped operation.
        '''
        def wrapped_op(self, other):
            result = getattr(super(HookedParameter, self), op_name)(other)
            return self.hook_point(result)  # Apply the hook
        return wrapped_op

    def _apply_hook(self, x):
        '''
        Apply the hook point to the input.
        
        Args:
            x: The input to be hooked.
        
        Returns:
            The hooked input.
        '''
        return self.hook_point(x)

    def setup(self):
        '''
        Set up the module and hook dictionaries.
        '''
        self.mod_dict = {'param': self.param, 'hook_point': self.hook_point}
        self.hook_dict = {'hook_point': self.hook_point}

    def unwrap(self) -> P:
        '''
        Unwrap the HookedParameter to get the original parameter.
        
        Returns:
            P: The original, unwrapped parameter.
        '''
        return cast(P, nn.Parameter(self.data, requires_grad=self.requires_grad))

class HookedModule(HookedRootModule, Generic[T]):
    '''
    A wrapper for nn.Module that adds hook points and wraps submodules.
    '''
    
    def __init__(self, module: T):
        '''
        Initialize the HookedModule.
        
        Args:
            module (T): The module to be wrapped.
        '''
        super().__init__()
        self._module = module
        self.hook_point = getattr(self._module, 'hook_point', HookPoint())
        self._create_forward()
        self._wrap_submodules()
        self.setup()

    def setup(self):
        '''
        Set up the module and hook dictionaries.
        '''
        self.mod_dict = {'hook_point': self.hook_point}
        self.hook_dict = {'hook_point': self.hook_point}
        self._populate_dicts(self._module)

    def _populate_dicts(self, module: nn.Module, prefix=''):
        '''
        Recursively populate the module and hook dictionaries.
        
        Args:
            module (nn.Module): The module to populate from.
            prefix (str): The prefix for the current module's name.
        '''
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            self.mod_dict[full_name] = child
            
            if isinstance(child, HookedModule):
                hook_point_name = f"{full_name}.hook_point"
                self.mod_dict[hook_point_name] = self.hook_dict[hook_point_name] = child.hook_point
                self._populate_dicts(child._module, full_name)
            elif isinstance(child, HookPoint):
                self.hook_dict[full_name] = child
            else:
                self._populate_dicts(child, full_name)
        
        for name, param in module.named_parameters(recurse=False):
            if isinstance(param, HookedParameter):
                full_name = f"{prefix}.{name}" if prefix else name
                self.mod_dict[full_name] = param
                hook_point_name = f'{full_name}.hook_point'
                self.mod_dict[hook_point_name] = self.hook_dict[hook_point_name] = param.hook_point

    def unwrap(self) -> T:
        '''
        Recursively unwrap the HookedModule to get the original module.
        
        Returns:
            T: The original, unwrapped module.
        '''
        for name, submodule in self._module.named_children():
            if isinstance(submodule, (nn.ModuleList, nn.Sequential, nn.ModuleDict)):
                unhooked_container = process_container_module(
                    submodule,
                    lambda m: m.unwrap() if isinstance(m, HookedModule) else m
                )
                setattr(self._module, name, unhooked_container)
            elif isinstance(submodule, (HookedModule, HookedParameter)):
                setattr(self._module, name, submodule.unwrap())
        return self._module

    def _wrap_submodules(self):
        '''
        Recursively wrap submodules with hooks.
        '''
        if not any(isinstance(self._module, built_in_module) for built_in_module in BUILT_IN_MODULES):
            for name, submodule in self._module.named_parameters(recurse=False):
                if isinstance(submodule, (nn.Parameter, torch.Tensor)):
                    setattr(self._module, name, auto_hook(submodule))
                else:
                    raise ValueError(f"Submodule {name} is not a nn.Parameter or torch.Tensor")

        for name, submodule in self._module.named_children():
            if isinstance(submodule, HookPoint):
                continue
            elif isinstance(submodule, (nn.ModuleList, nn.Sequential, nn.ModuleDict)):
                hooked_container = process_container_module(
                    submodule,
                    lambda m: auto_hook(m)
                )
                setattr(self._module, name, hooked_container)
            else:
                setattr(self._module, name, auto_hook(submodule))

    def _create_forward(self):
        '''
        Create a new forward method that calls self.hook_point on the ouput of original forward method
        '''
        original_forward = self._module.forward
        original_type_hints = get_type_hints(original_forward)

        @functools.wraps(original_forward)
        def new_forward(*args: Any, **kwargs: Any) -> Any:
            return self.hook_point(original_forward(*args, **kwargs))

        new_forward.__annotations__ = original_type_hints
        self.forward = new_forward

    def list_all_hooks(self):
        '''
        List all hooks in the module.
        
        Returns:
            list: A list of tuples containing hook names and hook points.
        '''
        return [(hook, hook_point) for hook, hook_point in self.hook_dict.items()] 