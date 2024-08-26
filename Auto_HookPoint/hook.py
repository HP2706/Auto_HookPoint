from __future__ import annotations
from inspect import isclass
import warnings
from torch import nn
import torch
from torch.nn.modules.module import Module
from transformer_lens.hook_points import HookPoint, HookedRootModule
from typing import (
    Optional,
    Set,
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
        HookedModule[T], 
        HookedClass[T], 
        HookedParameter[P], 
        HookedClass[P]
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
    param: P
    hook_point: HookPoint
    
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

        
    def detach(self) -> HookedParameter[P]:
        """
        Detach the parameter and return a new HookedParameter instance.
        NOTE: 
            The detach method must return the same type as the object it's called on
            to maintain consistency with the original Parameter behavior
        """
        detached_param = self.param.detach()
        return HookedParameter(detached_param)

    def unwrap(self) -> P:
        '''
        Unwrap the HookedParameter to get the original parameter.
        
        Returns:
            P: The original, unwrapped parameter.
        '''
        return cast(P, nn.Parameter(self.data, requires_grad=self.requires_grad))

#heavily aspired by this very clean approach. https://github.com/jbloomAus/SAELens/blob/33de7f1abe10de183ffe3ea17b51dbcb991daf37/sae_lens/load_model.py#L76
class HookedModule(HookedRootModule, Generic[T]):
    def __init__(self, model: T):
        super().__init__()        
        self.model = model
        self.__dict__['model'] = model #NOTE avoid __getattr__
        self.__dict__['_modules'] = model._modules  
        self.__dict__['_parameters'] = model._parameters
        self.setup()
        
         # Preserve the original class name
        class_name = model.__class__.__name__
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
        if 'model' in self.__dict__:
            return getattr(self.__dict__['model'], name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
        
    def unwrap(self) -> T:
        def recursive_unwrap(module : nn.Module, prefix=''):
            for name, param in module.named_parameters(recurse=False):
                if isinstance(param, HookedParameter):
                    setattr(module, name, param.unwrap())
            
            for name, child in module.named_children():
                recursive_unwrap(child, f"{prefix}.{name}" if prefix else name)

        recursive_unwrap(self.model) # we only need to wrap parameters 
        self.model._modules = {k: v for k, v in self.model._modules.items() if not isinstance(v, HookPoint)}
        return self.model
        
    def setup(self):
        self.hook_point = HookPoint()
        hook_name = 'hook_point'
        self.hook_point.name = hook_name
        self.mod_dict = {'hook_point' : self.hook_point}
        self.hook_dict: dict[str, HookPoint] = {'hook_point' : self.hook_point}
        self.maybe_hook_params(self.model, '')
        
        for name, module in self.model.named_modules():
            if name == "":
                continue
            
            if isinstance(module, (nn.ModuleList, nn.Sequential, nn.ModuleDict)):
                continue #no need to hook container modules
            
            hook_name = f'{name}.hook_point'
            hook_point = HookPoint()
            hook_point.name = hook_name  # type: ignore

            module.register_forward_hook(hook_factory(hook_point))
            #bwd hook does not appear to be needed
            #module.register_backward_hook(backward_hook_factory(hook_point))

            self.hook_dict[hook_name] = hook_point
            self.mod_dict[hook_name] = hook_point
            self.maybe_hook_params(module, name)
    
    def maybe_hook_params(self, module : nn.Module, name : str):
        if not any(isinstance(module, built_in_module) for built_in_module in BUILT_IN_MODULES):
            for param_name, param in module.named_parameters(recurse=False):
                hooked_param = cast(HookedParameter, HookedParameter(param))
                setattr(module, param_name, hooked_param)

                global_name = param_name if name=='' else f'{name}.{param_name}'
                hook_name = f'{global_name}.hook_point'
                hooked_param.hook_point.name = hook_name
                self.hook_dict[hook_name] = self.mod_dict[hook_name] = hooked_param.hook_point
                
    def forward(self, *args: Any, **kwargs: Any):
        return self.hook_point(self.model.forward(*args, **kwargs))
    
def hook_factory(hook_point: HookPoint):
    def hook_fn(module: Any, input: Any, output: Any) -> Any:
        if isinstance(output, torch.Tensor):
            return hook_point(output)
        elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            return (hook_point(output[0]), *output[1:])
        else:
            return output
    return hook_fn
