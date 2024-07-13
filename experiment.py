from typing import Any, Callable, Generic, List, Optional, ParamSpec, Protocol, Tuple, TypeVar, Type, Union, cast, reveal_type, Concatenate
import torch
import torch.nn as nn
from transformer_lens.hook_points import HookPoint, HookedRootModule


P = ParamSpec('P')
R = TypeVar('R')

class ForwardProtocol(Protocol[P, R]):
    def forward(self, *args: P.args, **kwargs: P.kwargs) -> R:...

T = TypeVar('T', bound=nn.Module)

class HookedClass(Generic[T], HookedRootModule):...

def create_hooked_class(base_class: Type[T]) -> Type[T]:        
    class HookedModuleType(HookedRootModule):
        def __init__(self, *args, **kwargs):
            HookedRootModule.__init__(self)
            self.base_module: T
            self._add_hooks()

        def _add_hooks(self):
            for name, module in self.named_modules():
                if isinstance(module, nn.Module) and not isinstance(module, HookPoint):
                    setattr(module, 'hook_point', HookPoint())
                    original_forward = module.forward
                    def new_forward(self, *args, **kwargs):
                        return self.hook_point(original_forward(*args, **kwargs))
                    module.forward = new_forward.__get__(module, type(module))

        def forward(self, *args: P.args, **kwargs: P.kwargs) -> R:
            return self.base_module.forward(*args, **kwargs)

    return cast(Type[T], HookedModuleType)


# Example usage
class ComplexModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 30)
        self.nested = NestedModule()

    def forward(self, x: torch.Tensor, y : Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.linear1(x)
        x = self.linear2(x)
        return self.nested(x)

class NestedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(30, 40)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
def add_hooks(module: T) -> Union[T, HookedRootModule]:    
    hooked_class = create_hooked_class(type(module))
    new_module = hooked_class.__new__(hooked_class)
    new_module.__dict__ = module.__dict__
    new_module.__init__()
    return new_module


class WrappedComplexModule(ComplexModule, HookedRootModule):
    pass

Wrapped = create_hooked_class(ComplexModule)


Wrapped.forward()

model = ComplexModule()
hooked_model = add_hooks(model)

# Type checking
reveal_type(hooked_model.run_with_hooks)
reveal_type(hooked_model.forward)

hooked_model.run_with_hooks()

# Test if hook points are added
print(isinstance(hooked_model, HookedRootModule))
print(hasattr(hooked_model.linear1, 'hook_point'))
print(hasattr(hooked_model.linear2, 'hook_point'))
print(hasattr(hooked_model.nested, 'hook_point'))
print(hasattr(hooked_model.nested.linear, 'hook_point'))
print(hasattr(hooked_model, 'run_with_hooks'))

# Test forward pass
input_tensor = torch.randn(1, 10)
output = hooked_model.forward(input_tensor)
print(f"Output shape: {output.shape}")

P = ParamSpec('P')


