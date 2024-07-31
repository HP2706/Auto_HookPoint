from typing import Generic, TypeVar
from torch import nn
import functools
from transformer_lens.hook_points import HookPoint, HookedRootModule
from hook import HookedParameter
T = TypeVar('T')

## this is an experimental version of HookedModule 
class ExperimentalHookedModule(HookedRootModule, Generic[T]):
    def __init__(self, module: T):
        super().__init__()
        object.__setattr__(self, '_module', module)
        object.__setattr__(self, 'mod_dict', {})
        object.__setattr__(self, 'hook_dict', {})
        object.__setattr__(self, 'hook_point', HookPoint())

        self._wrap_submodules()
        self.setup()

        print("mod_dict", self.mod_dict)
        print("hook_dict", self.hook_dict)

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
        print("named_children", list(module.named_children()))
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            self.mod_dict[full_name] = child
            
            if isinstance(child, ExperimentalHookedModule):
                hook_point_name = f"{full_name}.hook_point"
                print("adding hook_point", hook_point_name)
                self.mod_dict[hook_point_name] = self.hook_dict[hook_point_name] = child.hook_point
                self._populate_dicts(child._module, full_name)
            elif isinstance(child, HookPoint):
                self.hook_dict[full_name] = child
                print("adding hook_point", full_name)
            else:
                self._populate_dicts(child, full_name)

    def print_mod_dict(self):
        print("mod_dict", self.mod_dict)
        print("hook_dict", self.hook_dict)

    def _wrap_submodules(self):
        for name, child in self._module.named_children():
            if isinstance(child, nn.Module):
                setattr(self._module, name, ExperimentalHookedModule(child))

    def _unwrap(self):
        return self._module

    @functools.wraps(nn.Module.forward)
    def forward(self, *args, **kwargs):
        print('calling forward')
        return self.hook_point(self._module.forward(*args, **kwargs))

    def __getattr__(self, name):
        if name == '_module':
            return object.__getattribute__(self, '_module')
        if name in ('mod_dict', 'hook_dict', 'hook_point'):
            return object.__getattribute__(self, name)
        return getattr(self._module, name)
