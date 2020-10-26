import torch
import torch.nn as nn
from functools import reduce
from . import flatten_seq_method


class Identity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

class Flatten(nn.Module):

    def forward(self, x, dim=1, *args, **kwargs):
        return x.reshape(*x.shape[:-dim], -1)

class Squeeze(nn.Module):
    def __repr__(self):
        return "Squeeze()"

    def forward(self, x, *args, **kwargs):
        if isinstance(x, list):
            return [self(x_i) for x_i in x]
        return torch.squeeze(x)

class Unsqueeze(nn.Module):
    def __repr__(self):
        return "Unsqueeze(dim=%s)"%self.dim
    def __init__(self, dim=1):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x, *args, **kwargs):
        if isinstance(x, list):
            return [self(x_i) for x_i in x]
        return torch.unsqueeze(x, self.dim)


class Reshape(nn.Module):
    def __repr__(self):
        return "Reshape(shape=%s)"%str(self.shape)

    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = tuple([int(s) for s in shape])

    def forward(self, x, *args, **kwargs):
        shape = (*x.shape[0:-1], *self.shape)
        return torch.reshape(x, shape)
