
import sys
import os
# add parent directory to sys path to import auto_hook
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Auto_HookPoint import auto_hook
import torch.nn as nn

@auto_hook
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc1(x)
    
model = MyModel()
print(type(model)) #HookedModule[MyModel]
print(model.hook_dict.items())  #dict_items([('hook_point', HookPoint()), ('fc1.hook_point', HookPoint()), ('fc2.hook_point', HookPoint())])