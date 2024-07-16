from torch import nn


class SimpleNestedModuleList(nn.Module):
    def __init__(self):
        super().__init__()
        self.bla = nn.ModuleList([nn.Linear(10, 10), nn.Linear(10, 10)])

    def get_forward_shape(self):
        return self.bla[0].get_forward_shape()

    def forward(self, x):
        for module in self.bla:
            x = module(x)
        return x