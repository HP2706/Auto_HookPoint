from __future__ import annotations
from inspect import isclass
from torch import nn
import torch
from transformer_lens.hook_points import HookPoint, HookedRootModule
from typing import List, Optional, TypeVar, Type, Union, cast, overload

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
            print('listing hooks in init')
            print(self.list_all_hooks())
            print(f'{cls.__name__}.hook_dict', self.hook_dict)
            #print(f'{cls.__name__}.mod_dict', self.mod_dict)
        cls.__init__ = new_init

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def setup(self, filter: List[str] = []):
        self._initialize_and_wrap_hooks(filter=filter)
        print("calling setup\n\n")
        super().setup()
        self._setup_called = True

    def _initialize_and_wrap_hooks(self, filter: List[str] = [], prefix: str = ""):
        change_dict = {}
        children = list(self.named_children())
        assert len(children) > 0, f"No children found for {self.__class__.__name__}"

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
                setattr(module, attr, value)
            elif isinstance(value, HookPoint):
                pass
            else:
                setattr(module, attr, value)
        return module
    
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
    

@overload
def auto_hooked(cls_or_instance: Type[T]) -> Type[T]: ...

@overload
def auto_hooked(cls_or_instance: T) -> T: ...

def auto_hooked(
    cls_or_instance: Union[Type[T], T]
) -> Union[Type[T], T]:
    if not isclass(cls_or_instance):
        if isinstance(cls_or_instance, (nn.ModuleList, nn.ModuleDict, nn.Sequential)):
            for idx, item in iterate_module(cls_or_instance):
                if isinstance(cls_or_instance, nn.ModuleDict):
                    cls_or_instance[str(idx)] = auto_hooked(item)
                else:
                    cls_or_instance[int(idx)] = auto_hooked(item)
            return cls_or_instance
        elif isinstance(cls_or_instance, nn.Module):
            return cast(T, _wrap_instance(cls_or_instance))
        else:
            raise ValueError("auto_hooked expects a class or instance, got: {}".format(cls_or_instance))
    else:
        if issubclass(cls_or_instance, AutoHookedRootModule):
            # Already an AutoHookedRootModule, return as is
            return cast(Type[T], cls_or_instance)
        else:
            # Handle class
            return cast(Type[T], _wrap_class(cls_or_instance))

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
    
    return WrappedModule()


def _wrap_class(cls: Type[T]) -> Type[AutoHookedRootModule]:
    class WrappedModule(AutoHookedRootModule):
        def __init__(self, *args, **kwargs):
            super().__init__()
            object.__setattr__(self, 'BASE_CLASS', cls(*args, **kwargs))
            
            # Wrap all nn.Module attributes of BASE_CLASS
            self._wrap_modules(self.BASE_CLASS._modules)
            print('listing modules in BASE_CLASS INIT')
            print(self.BASE_CLASS._modules)
            print('listing self.named_children')
            print(self.named_children())
            self.setup()

        def _wrap_modules(self, modules):
            for attr_name, attr_value in modules.items():
                if isinstance(attr_value, HookPoint):
                    continue
                if isinstance(attr_value, nn.Module):
                    wrapped_cls = auto_hooked(attr_value)
                    modules[attr_name] = wrapped_cls
                    if hasattr(wrapped_cls, '_modules'):
                        self._wrap_modules(wrapped_cls._modules)

                    self.hook_dict[f"hook_{attr_name}"] = HookPoint()

        def forward(self, *args, **kwargs):
            # Use the hook_point of the Wrapped class
            return self.BASE_CLASS.forward(*args, **kwargs)

        def __getattr__(self, name):
            if name == 'BASE_CLASS':
                return super().__getattribute__(name)
            return getattr(self.BASE_CLASS, name)
        
        def __setattr__(self, name, value):
            if name == 'BASE_CLASS' or not hasattr(self, 'BASE_CLASS'):
                super().__setattr__(name, value)
            else:
                setattr(self.BASE_CLASS, name, value)

        def __repr__(self):
            return f"Wrapped({self.BASE_CLASS.__class__.__name__})"
        
        def __str__(self):
            return f"Wrapped({self.BASE_CLASS.__class__.__name__})"
        
    return WrappedModule
        

""" 
def _wrap_class(cls: Type[T]) -> Type[AutoHookedRootModule]:
    # Get the source code of the class
    source = inspect.getsource(cls)
    
    # Parse the source code into an AST
    tree = ast.parse(textwrap.dedent(source))
    
    # Create an AST transformer
    class ModuleWrapper(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id in ['nn.Module', 'nn.Linear', 'nn.Conv2d']:  # Add more nn modules as needed
                return ast.Call(
                    func=ast.Name(id='auto_hooked', ctx=ast.Load()),
                    args=[node],
                    keywords=[]
                )
            return node

    # Apply the transformer
    new_tree = ModuleWrapper().visit(tree)
    
    # Compile the modified AST
    compiled = compile(new_tree, '<string>', 'exec')
    
    # Create a new namespace and execute the compiled code
    namespace = {}
    exec(compiled, namespace)
    
    # Get the modified class from the namespace
    ModifiedClass = namespace[cls.__name__]
    
    # Create the wrapped class
    class Wrapped(AutoHookedRootModule, ModifiedClass):
        def __init__(self, *args, **kwargs):
            AutoHookedRootModule.__init__(self)
            ModifiedClass.__init__(self, *args, **kwargs)
            self.setup()

    return Wrapped """