import torch.nn as nn
from ..distributions import Categorical
from ..utils import oneHot


DEFAULT_INIT = nn.init.xavier_normal_
CONDITIONING_HASH = ['concat']

def get_init(nn_lin):
    if nn_lin=="ReLU":
        return 'relu'
    elif nn_lin=="TanH":
        return 'tanh'
    elif nn_lin=="LeakyReLU":
        return 'leaky_relu'
    elif nn_lin=="conv1d":
        return "conv1d"
    elif nn_lin=="cov2d":
        return "conv2d"
    elif nn_lin=="conv3d":
        return "conv3d"
    elif nn_lin=="Sigmoid":
        return "sigmoid"
    else:
        raise NotImplementedError
    
def init_module(module, nn_lin=None, method=DEFAULT_INIT):
    if type(module)==nn.Sequential:
        for m in module:
            init_module(m, nn_lin=nn_lin, method=method)

    if nn_lin is not None:
        try:
            gain = nn.init.calculate_gain(get_init(nn_lin))
        except NotImplementedError:
            if nn_lin == "Siren":
                if type(module)==nn.Linear:
                    nn.init.uniform_(module.weight, -torch.sqrt(torch.tensor(6/float(module.weight.shape[1]))),
                                 torch.sqrt(torch.tensor(6/float(module.weight.shape[1]))))
                    nn.init.zeros_(module.bias)
                else:
                    raise NotImplementedError
            else:
                gain = 1.0
    else:
        gain = 1.0

    if type(module)==nn.Linear:
        method(module.weight.data, gain)
        nn.init.zeros_(module.bias)

class Identity(nn.Module):
    def __call__(self, *args, **kwargs):
        return args

class Sequential(nn.Sequential):
    def forward(self, input, *args, **kwargs):
        for module in self._modules.values():
            input = module(input, *args, **kwargs)
        return input

def flatten(x, dim=1):
    if len(x.shape[dim:]) != 1:
        if not x.is_contiguous():
            x = x.contiguous()
        return x.view(*tuple(x.shape[:dim]), np.cumprod(x.shape[dim:])[-1])
    else:
        return x

def format_label_data(self, y, phidden):
        def process(y, plabel):
            if plabel['dist'] == Categorical:
                y = oneHot(y, plabel['dim'])
            device = next(self.parameters()).device
            if not torch.is_tensor(y):
                y = torch.from_numpy(y)
            y = y.to(device, dtype=torch.float32)
            return y
        if not self.phidden.get('label_params') or y is None:
            return
        cond_data = []
        for t in phidden['label_params'].keys():
            y_tmp = process(y[t], phidden['label_params'][t])
            if phidden['label_params'][t].get('embedding') is not None:
                y_tmp = self.embeddings[t](y_tmp)
            cond_data.append(y_tmp)
        ys = torch.cat(cond_data, dim=-1)
        return ys

from . import flow
from .modules_nonlinearity import *
MLP_DEFAULT_NNLIN = Swish
CONV_DEFAULT_NNLIN = Swish

from .modules_normalization import *
from .modules_bottleneck import *
from .modules_convolution import *
from .modules_embeddings import *
from .modules_distribution import BernoulliLayer, GaussianLayer, get_module_from_density, CategoricalLayer
from .modules_hidden import *

