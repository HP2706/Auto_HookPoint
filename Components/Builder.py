from typing import List
from torch import nn
from transformer_lens.hook_points import HookPoint, HookedRootModule

class HookBuilder(HookedRootModule):
    def __init__(self):
        super().__init__()
        self.modules_to_hook = []

    def build_hooks(self, filter : List[str] = []):
        self._initialize_hooks(filter=filter)
        self._wrap_modules_with_hooks(filter=filter)
        self.setup()

    def _wrap_modules_with_hooks(self, filter : List[str] = []):
        for name, module in self.modules_to_hook:
            if name in filter:
                continue
            if not isinstance(module, HookPoint):
                hook = getattr(self, f"hook_{name}")
                wrapped_module = nn.Module()
                wrapped_module.forward = lambda x, module=module, hook=hook: hook(module(x))
                setattr(self, name, wrapped_module)

    def _initialize_hooks(self, filter : List[str] = []):
        for name, module in self.named_children():
            in_filtered = name in filter
            is_hook = isinstance(module, HookPoint)
            is_module = isinstance(module, nn.Module)
            hook_exists = hasattr(self, f"hook_{name}")
            if is_module and not is_hook and not in_filtered and not hook_exists:
                self.modules_to_hook.append((name, module))

        for name, module in self.modules_to_hook:

            setattr(self, f"hook_{name}", HookPoint())