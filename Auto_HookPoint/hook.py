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
    nn.Linear, nn.Embedding, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
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

    def __contains__(self, item: Any) -> bool:
        return item in self.module_class

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
        Hooked = cast(HookedModule[T], Hooked)
    elif isinstance(module_or_class, (nn.Parameter, torch.Tensor)):
        Hooked = cast(HookedParameter[P], HookedParameter(module_or_class)) # type: ignore
    else:
        raise ValueError(
            f"Module type {type(module_or_class)} is not supported should, "
            "be one of nn.Module or nn.Parameter/torch.Tensor"
        )
    return Hooked

class HookedParameter(nn.Parameter, Generic[P]):
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
        self.hook_point.name = 'hook_point'
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
    
    _modules: dict[str, nn.Module]
    _parameters: dict[str, nn.Parameter]
    
    def __init__(self, module: T, wrap_submodules: bool = True):
        '''
        Initialize the HookedModule.
        
        Args:
            module (T): The nn.module to be wrapped.
        '''
        super().__init__()
        self.__dict__['_module'] = module #NOTE avoid __getattr__
        self.__dict__['_modules'] = module._modules  
        self.__dict__['_parameters'] = module._parameters
        self.hook_point = HookPoint()
        self._create_forward()
        if wrap_submodules:
            self._wrap_submodules()
        self.setup()
        # Preserve the original class name
        class_name = module.__class__.__name__
        #hacky way to preserve the original class name
        self.__class__ = type(class_name, (HookedModule,), {'__module__': self.__class__.__module__})

    def __getattr__(self, name: str) -> Any:
        '''
        Delegate attribute access to the wrapped module if the attribute
        is not found in the HookedModule.
        '''
        if name in self.__dict__:
            return self.__dict__[name]
        if name in self._modules:
            return self._modules[name]
        if '_module' in self.__dict__:
            return getattr(self.__dict__['_module'], name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def setup(self):
        '''
        Set up the module and hook dictionaries.
        '''
        self.mod_dict = {'hook_point': self.hook_point}
        self.hook_dict = {'hook_point': self.hook_point}
        self._populate_dicts(self._module)

    def _wrap_submodules(self):
        '''
        Recursively wrap submodules with hooks.
        '''
        #wrap parameters if the module is not a built-in module like for instance nn.Linear
        if not any(isinstance(self._module, built_in_module) for built_in_module in BUILT_IN_MODULES):
            for name, submodule in self._module.named_parameters(recurse=False):
                if isinstance(submodule, (nn.Parameter, torch.Tensor)):
                    submodule = cast(nn.Parameter, submodule)
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
            elif any(isinstance(submodule, built_in_module) for built_in_module in BUILT_IN_MODULES):
                # For built-in modules, add a hook_point without wrapping
                submodule.hook_point = HookPoint()
                setattr(self._module, name, HookedModule(submodule, wrap_submodules=False))
            else:
                setattr(self._module, name, auto_hook(submodule))

    def _populate_dicts(self, module: nn.Module, prefix=''):
        '''
        Recursively populate the module and hook dictionaries.
        
        Args:
            module (nn.Module): The module to populate from.
            prefix (str): The prefix for the current module's name.
        '''
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            # Remove '._module' from the full_name
            full_name = full_name.replace('._module', '')
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
                # Remove '._module' from the full_name
                full_name = full_name.replace('._module', '')
                self.mod_dict[full_name] = param
                hook_point_name = f'{full_name}.hook_point'
                self.mod_dict[hook_point_name] = self.hook_dict[hook_point_name] = param.hook_point

        #name the hook_points

        for d in [self.mod_dict.items(), self.hook_dict.items()]:
            for k, v in d:
                if isinstance(v, HookPoint):
                    v.name = k

    def unwrap(self) -> T:
        '''
        Recursively unwrap the HookedModule to get the original module.
        
        Returns:
            T: The original, unwrapped module.
        '''
        for name, submodule in list(self._module.named_children()):
            if isinstance(submodule, (nn.ModuleList, nn.Sequential, nn.ModuleDict)):
                unhooked_container = process_container_module(
                    submodule,
                    lambda m: m.unwrap() if isinstance(m, HookedModule) else m
                )
                setattr(self._module, name, unhooked_container)
                self._module._modules[name] = unhooked_container
            elif isinstance(submodule, (HookedModule, HookedParameter)):
                unwrapped_submodule = submodule.unwrap()
                setattr(self._module, name, unwrapped_submodule)
                self._module._modules[name] = unwrapped_submodule
            elif hasattr(submodule, 'hook_point'):
                # Remove hook_point from built-in modules
                delattr(submodule, 'hook_point')

        # Remove HookPoint instances from _modules
        self._module._modules = {k: v for k, v in self._module._modules.items() if not isinstance(v, HookPoint)}

        # Remove HookPoint instances from the module itself
        for name in list(self._module.__dict__.keys()):
            if isinstance(getattr(self._module, name), HookPoint):
                delattr(self._module, name)

        return self._module

    def _create_forward(self):
        '''
        Create a new forward method that calls self.hook_point on the output of original forward method
        '''
        original_forward = self._module.forward
        original_type_hints = get_type_hints(original_forward) # move typehints

        @functools.wraps(original_forward)
        def new_forward(*args: Any, **kwargs: Any) -> Any:
            return self.hook_point(original_forward(*args, **kwargs))

        new_forward.__annotations__ = original_type_hints
        self.forward = new_forward