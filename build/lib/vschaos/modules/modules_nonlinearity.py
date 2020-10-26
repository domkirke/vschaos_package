import torch
import torch.nn as nn


class ScaledSoftsign(nn.Module):
    def __init__(self, device=None):
        super(ScaledSoftsign, self).__init__()
        self.params = nn.ParameterDict({'a': nn.Parameter(torch.tensor(2., requires_grad=True)),
                                        'b': nn.Parameter(torch.tensor(1., requires_grad=True))})
        nn.init.normal_(self.params['a'])
        nn.init.normal_(self.params['b'])

    def forward(self, x):
        return (self.params['a'] * x)/(1 + torch.abs( self.params['b'] * x) )


class Swish(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.reciprocal(1 + torch.exp(-x))

class LipSwish(Swish):
    def forward(self, x):
        return super().forward(x) / 1.1

class Siren(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.sin(x)

