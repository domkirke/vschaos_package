#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 19:44:23 2018

@author: chemla
"""

from . import MLP
from . import CONV_DEFAULT_NNLIN, ActNorm1d, ActNorm2d, ActNorm3d
import torch
import torch.nn as nn
import numpy as np
from ..utils import checklist, checktuple, Unsqueeze, Reshape, Flatten, split_select, flatten_seq_method
from . import Sequential, format_label_data
from copy import deepcopy

DEFAULT_GATED_NNLIN = "ReLU"

conv_func_hash = {nn.Conv1d:nn.functional.conv1d, nn.Conv2d:nn.functional.conv2d, nn.Conv3d:nn.functional.conv3d}
def perform_conv(x, conv_module, windowed=None):
    # convolution function implementing windowed convolution
    if windowed:
        if len(conv_module.weight.shape) > 3:
            raise NotImplementedError
        weight = conv_module.weight
        window = getattr(torch, windowed+'_window')(weight.shape[-1], periodic=False, device = weight.device) 
        return conv_func_hash[type(conv_module)](x, weight*window, bias=conv_module.bias, stride=conv_module.stride, padding = conv_module.padding, dilation = conv_module.dilation)
    else:
        return conv_module(x)

#%% Individual convolutional / deconvolutional layers

class ConvLayer(nn.Module):
    """
    ConvLayer implements single convolutional layers with several additional features such as dropout or batch normalization.
    :param in_channels: input channels
    :type in_channels: int
    :param out_channels: output channels
    :type out_channels: int
    :param conv_dim: convolution dimensions (1, 2 or 3)
    :type conv_dim: int
    :param kernel_size: kernel size
    :type kernel_size: int
    :param pool: pooling size (default : None)
    :type pool: int or None
    :param dilation: convolution dilation (default : 1)
    :type dilation: int
    :param stride: convolution stride (default : 1)
    :type stride: int
    :param windowed: window convolution (default : False)
    :type windowed: bool or None
    :param normalization: has batch normalization (None, 'batch' or 'instance', default : 'batch')
    :param nn_lin: used non-linearity (default : %s, has to be a `torch.nn` item)
    :type nn_lin: None or str
    """%CONV_DEFAULT_NNLIN
    dump_patches = True
    conv_modules = {1: torch.nn.Conv1d, 2:torch.nn.Conv2d, 3:torch.nn.Conv3d}
    dropout_modules = {1: nn.Dropout, 2:nn.Dropout2d, 3:nn.Dropout3d}
    bn_modules = {1: nn.BatchNorm1d, 2:nn.BatchNorm2d, 3:nn.BatchNorm3d}
    in_modules = {1: nn.InstanceNorm1d, 2:nn.InstanceNorm2d, 3:nn.InstanceNorm3d}

    ac_modules = {1: ActNorm1d, 2: ActNorm2d, 3: ActNorm3d}
    pool_modules = {1:nn.MaxPool1d, 2:nn.MaxPool2d, 3:nn.MaxPool3d}

    def __init__(self, *args, **kwargs):
        super(ConvLayer, self).__init__()
        self.build_module(*args, **kwargs)
        self.init_modules(*args, **kwargs)


    def build_module(self, in_channels, out_channels, conv_dim, kernel_size, pool=None, dilation=1, dropout=0.5, padding=0, stride=1, windowed=None, nn_lin=CONV_DEFAULT_NNLIN, normalization=None, *args, **kwargs):
        self.pool_indices = None
        # Convolutional modules
        self.conv_dim = conv_dim
        self.windowed = windowed
        additional_conv_args = {}
        if kwargs.get('output_padding') is not None:
            output_padding = kwargs['output_padding']
            if hasattr(output_padding, "__iter__"):
                output_padding = tuple([int(x) for x in kwargs['output_padding']])
            additional_conv_args['output_padding'] = output_padding

        self.in_channels = in_channels; self.out_channels = out_channels
        self.stride = checktuple(stride, conv_dim)
        self.kernel_size = checktuple(kernel_size, conv_dim)
        self.dilation = checktuple(dilation, conv_dim)
        self.padding = checktuple(padding)
        self.add_module('conv_module', self.conv_modules[conv_dim](self.in_channels, self.out_channels, self.kernel_size, padding=self.padding, dilation=self.dilation, stride=self.stride, **additional_conv_args))
        # Dropout
        self.dropout = dropout
        if self.dropout:
            self.add_module('dropout_module', self.dropout_modules[conv_dim](self.dropout))
        # Batch/Instance Normalization
        self.normalization =normalization
        if self.normalization:
            if self.normalization == 'batch':
                self.add_module('normalization', self.bn_modules[conv_dim](out_channels))
            elif self.normalization == 'instance':
                self.add_module('normalization', self.in_modules[conv_dim](out_channels))
            elif self.normalization == "activation":
                self.add_module('normalization', self.ac_modules[conv_dim](out_channels))

        # Non-linearity
        self.nn_lin = nn_lin
        if self.nn_lin is not None:
            if isinstance(self.nn_lin, str):
                nn_lin = getattr(nn, nn_lin)
            if isinstance(self.nn_lin, type):
                nn_lin = nn_lin()
            self.add_module('nnlin_module', nn_lin)

        # Pooling layer
        self.pooling = pool
        if not pool is None:
            self.init_pooling(conv_dim, pool)

    def init_pooling(self, dim, kernel_size):
        self.add_module('pool_module', self.pool_modules[dim](kernel_size, return_indices=True))

    def init_modules(self, *args, **kwargs):
        torch.nn.init.xavier_normal_(self._modules['conv_module'].weight)
        torch.nn.init.zeros_(self._modules['conv_module'].bias)

    def forward(self, x):
        """ performs convolution on input x.
        :param torch.Tensor x: input tensor
        :return: convoluted tensor
        :rtype: torch.Tensor"""
        # perform convolution
        current_out = perform_conv(x, self._modules['conv_module'], windowed = self.windowed)
        if self.dropout:
            current_out = self._modules['dropout_module'](current_out)
        if self.normalization:
            current_out = self._modules['normalization'](current_out)
        if self.nn_lin:
            current_out = self._modules['nnlin_module'](current_out)
        if self.pooling:
            current_out = self._modules['pool_module'](current_out)
            try:
                current_out, self.pool_indices = current_out
            except Exception as e:
                raise Warning(str(e))
            pass
        return current_out

    def get_pooling_indices(self):
        return self.pool_indices

    def get_output_size(self, input_size):
        """get_output_size returns the post-convolution shape of an input shape input_size. input_size must be a tuple
        whose shape is the dimension of the layer.
        :param np.ndarray input_size: shape of the input tensor
        :return: shape of convoluted input
        :rtype: np.ndarray"""
        # conv layers
        current_output = input_size + 2*np.array(self._modules['conv_module'].padding) - (self._modules['conv_module'].kernel_size - 1) - 1
        pre_pooling = np.floor(current_output/np.array(self._modules['conv_module'].stride) + 1)
        # pooling layers
        post_pooling = None
        if not self._modules.hasattr('pool_module') is None:
            post_pooling = pre_pooling - (self._modules['pool_module'].kernel_size-1) - 1
            post_pooling = np.floor(post_pooling/np.array(self._modules['pool_module'].kernel_size) + 1)
        return pre_pooling, post_pooling


class GatedConvLayer(ConvLayer):
    """
    GatedConvLayer implements a simple gated convolutional layer, inheriting ConvLayer but performing gated convolution.
    :param in_channels: input channels
    :type in_channels: int
    :param out_channels: output channels
    :type out_channels: int
    :param conv_dim: convolution dimensions (1, 2 or 3)
    :type conv_dim: int
    :param kernel_size: kernel size
    :type kernel_size: int
    :param pool: pooling size (default : None)
    :type pool: int or None
    :param dilation: convolution dilation (default : 1)
    :type dilation: int
    :param stride: convolution stride (default : 1)
    :type stride: int
    :param windowed: window convolution (default : False)
    :type windowed: bool or None
    :param normalization: has batch normalization (None, 'batch' or 'instance', default : 'batch')
    :param nn_lin: used non-linearity (default : %s, has to be a `torch.nn` item)
    :type nn_lin: None or str
    """%CONV_DEFAULT_NNLIN
    dump_patches = True
    def build_module(self, in_channels, out_channels, conv_dim, kernel_size,  dilation=1, padding=0, stride=1, nn_lin=None, *args, **kwargs):
        super(GatedConvLayer, self).build_module(in_channels, out_channels, conv_dim, kernel_size, dilation=dilation, padding=padding, nn_lin=None, stride=stride, *args, **kwargs)
        self.pool_indices = None
        self.nn_lin = nn_lin
        self.residual = kwargs.get('residual', True)
        # Convolutional modules
        additional_args = {'output_padding': kwargs['output_padding']} if kwargs.get('output_padding') else {}
        self.add_module('conv_module_sig', self.conv_modules[conv_dim](self.in_channels, self.out_channels, self.kernel_size, padding=self.padding, dilation=self.dilation, stride=self.stride, **additional_args))
        self.add_module('conv_module_residual', self.conv_modules[conv_dim](self.in_channels, self.out_channels, self.kernel_size, padding=self.padding, dilation=self.dilation, stride=self.stride, **additional_args))
        self.add_module('conv_module_1x1', self.conv_modules[conv_dim](self.out_channels, self.out_channels, 1, padding=0, dilation=1, stride=1, **additional_args))

    def init_modules(self, *args, **kwargs):
        torch.nn.init.xavier_normal_(self._modules['conv_module_sig'].weight, 5.0/3)
        torch.nn.init.zeros_(self._modules['conv_module_sig'].bias)
        torch.nn.init.constant_(self._modules['conv_module_residual'].weight, 1./(np.prod(self.kernel_size)*self.in_channels))
        torch.nn.init.zeros_(self._modules['conv_module_residual'].bias)
        torch.nn.init.xavier_normal_(self._modules['conv_module_1x1'].weight)
        torch.nn.init.zeros_(self._modules['conv_module_1x1'].bias)

        self._modules['conv_module_residual'].weight.data.requires_grad = False
        self._modules['conv_module_residual'].bias.data.requires_grad = False

    def forward(self, x):
        # Main convolution
        current_out_sig = torch.tanh(self._modules['conv_module'](x))
        current_out_tanh = torch.nn.functional.sigmoid(perform_conv(x, self._modules['conv_module_sig'], windowed=self.windowed))
        current_out = self.conv_module_1x1(current_out_sig * current_out_tanh)
        if self.dropout:
            current_out = self._modules['dropout_module'](current_out)
        # Residual path
        if self.residual:
            current_out_residual = self.conv_module_residual(x)
            current_out = current_out + current_out_residual
        if self.normalization:
            current_out = self._modules['normalization'](current_out)

        if self.nn_lin is not None:
            current_out = getattr(torch.nn, self.nn_lin)()(current_out)
        # Pooling
        if self.pooling:
            current_out = self._modules['pool_module'](current_out)
            try:
                current_out, self.pool_indices = current_out
            except Exception as e:
                raise Warning(str(e))
            pass
        return current_out


#%% Individual deconvolution layers


def perform_deconv(x, conv_module, windowed=None):

    if windowed:
        if len(conv_module.weight.shape) > 3:
            raise NotImplementedError
        weight = conv_module.weight
        window = getattr(torch, windowed+'_window')(weight.shape[-1], periodic=False, device = weight.device)
        return nn.functional.conv_transpose1d(x, weight*window, bias=conv_module.bias, stride=conv_module.stride, padding = conv_module.padding, dilation = conv_module.dilation, output_padding=conv_module.output_padding)
    else:
        return conv_module(x)

    
class DeconvLayer(ConvLayer):
    """
    DeconvLayer implements single deconvolutional layers with several additional features such as dropout or batch normalization.
    :param in_channels: input channels
    :type in_channels: int
    :param out_channels: output channels
    :type out_channels: int
    :param conv_dim: convolution dimensions (1, 2 or 3)
    :type conv_dim: int
    :param kernel_size: kernel size
    :type kernel_size: int
    :param pool: pooling size (default : None)
    :type pool: int or None
    :param dilation: convolution dilation (default : 1)
    :type dilation: int
    :param stride: convolution stride (default : 1)
    :type stride: int
    :param windowed: window convolution (default : False)
    :type windowed: bool or None
    :param normalization: has batch normalization (None, 'batch' or 'instance', default : 'batch')
    :param nn_lin: used non-linearity (default : %s, has to be a `torch.nn` item)
    :type nn_lin: None or str
    """%CONV_DEFAULT_NNLIN
    pool_modules = {1:nn.MaxUnpool1d, 2:nn.MaxUnpool2d, 3:nn.MaxUnpool3d}
    conv_modules = {1: torch.nn.ConvTranspose1d, 2:torch.nn.ConvTranspose2d, 3:torch.nn.ConvTranspose3d}
    
    def forward(self, x, *args, indices=None, output_size=None, **kwargs):
        batch_shape = checktuple(x.shape[0])
        if len(x.shape) > self.conv_dim + 2:
            batch_shape = x.shape[:2]
            x = x.view(np.cumprod(x.shape[:-(self.conv_dim+1)])[-1], *x.shape[-(self.conv_dim+1):])

        if not output_size is None:
            output_size = [int(o) for o in output_size]
        if self.pooling:
            add_args = {'output_size':output_size} if output_size is not None else {}
            x = self._modules['pool_module'](x, indices=indices, **add_args)
        current_out = perform_deconv(x, self._modules['conv_module'], windowed=self.windowed)
        if self.dropout:
            current_out = self._modules['dropout_module'](current_out)
        if self.normalization:
            current_out = self._modules['normalization'](current_out)
        if self.nn_lin:
            current_out = self._modules['nnlin_module'](current_out)
        if len(batch_shape) > 1:
            current_out = current_out.view(*batch_shape, *current_out.shape[-(self.conv_dim+1):])

        return current_out

    def init_pooling(self, dim, kernel_size):
        self.add_module('pool_module', self.pool_modules[dim](kernel_size))

    def init_modules(self, *args, **kwargs):
        torch.nn.init.xavier_normal_(self._modules['conv_module'].weight)
        torch.nn.init.zeros_(self._modules['conv_module'].bias)
    
    def get_output_size(self, input_size):
        # conv layers
        current_output = (input_size-1)*self._modules['conv_module'].stride - 2*np.array(self._modules['conv_module'].padding) - self._modules['conv_module'].dilation*(self._modules['conv_module'].kernel_size - 1) + self._modules['conv_module'].output_padding
        pre_pooling = np.floor(current_output/np.array(self._modules['conv_module'].stride) + 1)
    
        # pooling layers
        post_pooling = None
        if not self._modules.hasattr('pool_module') is None:
            post_pooling = (pre_pooling -1) * self._modules['pool_module'].kernel_size + self._modules['pool_module'].kernel_size
        return pre_pooling, post_pooling


class GatedDeconvLayer(DeconvLayer):
    """
    GatedConvLayer implements single convolutional layers with several additional features such as dropout or batch normalization.
    :param in_channels: input channels
    :type in_channels: int
    :param out_channels: output channels
    :type out_channels: int
    :param conv_dim: convolution dimensions (1, 2 or 3)
    :type conv_dim: int
    :param kernel_size: kernel size
    :type kernel_size: int
    :param pool: pooling size (default : None)
    :type pool: int or None
    :param dilation: convolution dilation (default : 1)
    :type dilation: int
    :param stride: convolution stride (default : 1)
    :type stride: int
    :param windowed: window convolution (default : False)
    :type windowed: bool or None
    :param normalization: has batch normalization (None, 'batch' or 'instance', default : 'batch')
    :param nn_lin: used non-linearity (default : %s, has to be a `torch.nn` item)
    :type nn_lin: None or str
    """%CONV_DEFAULT_NNLIN
    def build_module(self, in_channels, out_channels, conv_dim, kernel_size,  dilation=1, padding=0, stride=1, *args, **kwargs):
        self.pool_indices = None
        super(GatedDeconvLayer, self).build_module(in_channels, out_channels, conv_dim, kernel_size, dilation=dilation, padding=padding, stride=stride, *args, **kwargs)
        # Convolutional modules
        add_args = {'output_padding':kwargs['output_padding']} if kwargs.get('output_padding') is not None else {}
        self.add_module('conv_module_sig', self.conv_modules[conv_dim](in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, stride=stride,**add_args))
        self.add_module('conv_module_residual', self.conv_modules[conv_dim](in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, stride=stride, **add_args))
        self.add_module('conv_module_1x1', self.conv_modules[conv_dim](out_channels, out_channels, 1, dilation=1, stride=1))
        self.nn_lin = kwargs.get('nn_lin', DEFAULT_GATED_NNLIN)

    def init_modules(self, *args, **kwargs):
        torch.nn.init.xavier_normal_(self._modules['conv_module_sig'].weight, 5.0/3)
        torch.nn.init.zeros_(self._modules['conv_module_sig'].bias)
        torch.nn.init.constant_(self._modules['conv_module_residual'].weight, 1./(np.cumprod(self.conv_module_residual.kernel_size)[-1]*self.conv_module_residual.in_channels))
        torch.nn.init.zeros_(self._modules['conv_module_residual'].bias)
        torch.nn.init.xavier_normal_(self._modules['conv_module_1x1'].weight)
        torch.nn.init.zeros_(self._modules['conv_module_1x1'].bias)
        self._modules['conv_module_residual'].weight.data.requires_grad = False
        self._modules['conv_module_residual'].bias.data.requires_grad = False

    def forward(self, x,  *args, indices=None, output_size=None, **kwargs):
        if not output_size is None:
            output_size = [int(o) for o in output_size]

        current_out = x
        if self.pooling:
            current_out = self._modules['pool_module'](current_out)
            try:
                current_out, self.pool_indices = current_out
            except Exception as e:
                raise Warning(str(e))
                pass

        # Main convolution
        current_out_residual = self._modules['conv_module_residual'](current_out)
        current_out_sig = torch.tanh(self._modules['conv_module'](current_out))
        current_out_tanh = torch.sigmoid(perform_deconv(current_out, self._modules['conv_module_sig'], windowed=self.windowed))
        current_out = self._modules['conv_module_1x1'](current_out_sig * current_out_tanh)
        if self.dropout:
            current_out = self._modules['dropout_module'](current_out)

        current_out = current_out + current_out_residual

        if self.nn_lin is not None:
            current_out = getattr(torch.nn, self.nn_lin)()(current_out)

        return current_out 

#%% Convolution & Deconvolution containers

def get_label_channel(x, y, device="cpu"):
    return y.reshape(*y.shape, *tuple((1,)*(len(x.shape)-2))).repeat(1, 1, *x.shape[2:]).float().to(device)



class Convolutional(nn.Module):
    """
    The Convolutional class gathers multiple ConvLayer (and sub-classes) to create multi-layer convolutional modules.
    This class also supports additional features such as multi-input, conditioning, post-convolution shape, and adaptive padding.

    :py:class:`Convolutional` takes the following arguments :
    * *conv_dim* (required) convolution dimension (1, 2 or 3)
    * *channels* (required) list of channels
    * *kernel_size* (required) list of kernel sizes
    * *stride* list of strides (default : 1)
    * *dilation* list of dilations (default : 1)
    * *pool* list of poolings (default : None)
    * *conv_layer* layer class (default : :py:class:`ConvLayer`)
    * *norm_conv* batch normalization (can be : None, 'batch', 'instance', default : 'batch')
    * *dropout_conv* dropout (default : 0)

    :param pins: input parameters, formatted with the `vschaos` format (with additional *channels* keyword)
    :type pins: list or dict
    :param phidden: hidden parameters of current module, specified as above
    :type phidden: dict
    :param nn.Module conv_layer: type of convolution layer (default : :py:class:`ConvLayer`).
    :param bool return_indices: returns pooling indices (default : True)
    """
    conv_layer = ConvLayer
    dump_patches = True
    in_channels = True

    def __init__(self, pins, phidden, conv_layer=None, return_indices=True, *args, **kwargs):
        super(Convolutional, self).__init__()

        # parse hidden parameters
        self.depth = len(phidden['channels'])
        self.pins = pins
        phidden['conv_dim'] = phidden.get('conv_dim', len(checktuple(pins['dim'])))
        # phidden['conv_dim'] = len(checktuple(pins.get('dim')))
        self.conv_dim = phidden['conv_dim']
        phidden = dict(phidden)
        phidden['stride'] = checklist(phidden.get('stride', 1), len(phidden['channels']))
        phidden['dilation'] = checklist(phidden.get('dilation', 1), len(phidden['channels']))
        phidden['pool'] = checklist(phidden.get('pool'), len(phidden['channels']))
        phidden['kernel_size'] = [checktuple(ks, self.conv_dim) for ks in phidden['kernel_size']]
        phidden['padding'] = [tuple(np.ceil(np.array(ks)/2).astype('int').tolist()) for ks in phidden['kernel_size']]
        phidden['conv_layer'] = checklist(phidden.get('layer', self.conv_layer), len(phidden['channels']))
        phidden['output_padding'] = checklist(phidden.get('output_padding', None), len(phidden['channels']))
        phidden['norm_conv'] = checklist(phidden.get('norm_conv'), self.depth)
        phidden['dropout_conv'] = checklist(phidden.get('dropout_conv', 0.0), self.depth)

        self.return_indices = return_indices
        self.phidden = phidden
        #TODO specification of non linearity            
        self.dim_history = []
        if conv_layer:
            self.conv_layer = conv_layer
        self.conv_encoders = self.init_modules(self.pins, self.phidden, self.return_indices, self.conv_layer)
        self.embeddings = self.get_embeddings(self.phidden)


    @property
    def input_ndim(self):
        return self.conv_dim

    @staticmethod
    def init_modules(pins, phidden, return_indices=False, conv_layer=ConvLayer):
        conv_modules = []
        # conditioning
        pins = dict(pins)
        pins['channels'] = pins.get('channels', 1)
        if phidden.get('label_params'):
            phidden['conditioning'] = phidden.get('conditioning', 'concat')
            if phidden['conditioning'] == 'concat':
                pins['channels'] += sum([p['dim'] for p in phidden['label_params'].values()])
        channels = [pins['channels']] + phidden['channels']

        for l in range(len(channels)-1):
            current_dict = {'kernel_size':phidden['kernel_size'][l], 'stride':phidden['stride'][l], 'output_padding':phidden['output_padding'][l], 'pool':phidden['pool'][l],
                            'padding':phidden['padding'][l], 'conv_dim':phidden['conv_dim'], 'normalization':phidden['norm_conv'][l], 'dropout':phidden['dropout_conv'][l], 'dilation':phidden['dilation'][l]}
            conv_modules.append(conv_layer(channels[l], channels[l+1], return_indices=return_indices, **current_dict))
        return nn.ModuleList(conv_modules)

    @staticmethod
    def get_embeddings(phidden):
        embeddings = nn.ModuleDict()
        if phidden.get('label_params') is None:
            return None
        for k, pl in phidden.get('label_params', {}).items():
            if pl.get('embedding') is not None:
                embeddings[k] = pl['embedding']()
        return embeddings

    @flatten_seq_method
    def forward(self, x, y=None, *args, **kwargs):
        """
        forward method of the Convolution module. can take a list of input, and addditional label information in case
        of conditioning.
        :param x: input tensors of module
        :type x: list or torch.Tensor
        :param dict y: label input (default : None)
        :return: convoluted tensor
        :rtype:  torch.Tensor
        """
        if issubclass(type(x), list):
            x = torch.cat(x,)

        # conditioning
        if self.phidden.get('label_params'):
            assert y is not None, "Convolutional module need following labels : %s"%list(self.phidden['label_params'].keys())
            y = format_label_data(self, y, self.phidden)
            current_device = next(self.parameters()).device
            x = torch.cat((x, get_label_channel(x, y, device=current_device)), dim=1)

        # successively pass into convolutional modules
        out = x.clone()
        for l in range(self.depth):
            out = self.conv_encoders[l](out)
        return out
    
    def get_output_conv_length(self, params=None, input_dim=None):
        """
        returns layerwise output shapes of the Convolutional module, before and after pooling
        :param params: hidden parameters (default : instance's hidden parameters)
        :type params: dict or None
        :param input_dim: input parameters (default : instances's input parameters)
        :type input_dim: list, dict or None
        :return: pre-pooling shapes, post-pooling shapes
        :rtype: (tuple, tuple)
        """
        params = params or self.phidden
        current_length = input_dim or sum([p['dim'] for p in checklist(self.pins)])
        if not issubclass(type(current_length), (list, tuple)):
            current_length = [current_length]
        current_output = np.array(current_length)
        
        pre_pooling_dims = []; post_pooling_dims = []
        for l in range(len(params['channels'])):
            # conv layers
            current_output = current_output + 2*np.array(params['padding'][l]) - params['dilation'][l] * (np.array(params['kernel_size'][l])-1) - 1
            current_output = np.floor(current_output/np.array(params['stride'][l]) + 1)
            pre_pooling_dims.append(current_output)

            # pooling layers
            if not params['pool'][l] is None:
                current_output = current_output - (np.array(params['pool'][l])-1) - 1
                current_output = np.floor(current_output/np.array(params['pool'][l]) + 1)
            post_pooling_dims.append(deepcopy(current_output))

        return pre_pooling_dims, post_pooling_dims

    def get_pooling_indices(self):
        """
        return pooling indices of the last processed input
        :return: layerwise pooling indices
        :rtype: list
        """
        pooling_indices = []
        for l in range(self.depth):
            pooling_indices.append(self.conv_encoders[l].get_pooling_indices())
        # for p in pooling_indices:
        #     if p is not None:
        #         print(p.shape)
        return pooling_indices


class Deconvolutional(Convolutional):
    """
    The Convolutional class gathers multiple ConvLayer (and sub-classes) to create multi-layer convolutional modules.
    This class also supports additional features such as multi-input, conditioning, post-convolution shape, and adaptive padding.

    :py:class:`Convolutional` takes the following arguments :
    * *conv_dim* (required) convolution dimension (1, 2 or 3)
    * *channels* (required) list of channels
    * *kernel_size* (required) list of kernel sizes
    * *stride* list of strides (default : 1)
    * *dilation* list of dilations (default : 1)
    * *pool* list of poolings (default : None)
    * *conv_layer* layer class (default : :py:class:`ConvLayer`)
    * *norm_conv* batch normalization (can be : None, 'batch', 'instance', default : 'batch')
    * *dropout_conv* dropout (default : 0)
    * *output_padding* : output paddding (default : 0)

    :param pins: input parameters, formatted with the `vschaos` format (with additional *channels* keyword)
    :type pins: list or dict
    :param phidden: hidden parameters of current module, specified as above
    :type phidden: dict
    :param nn.Module conv_layer: type of convolution layer (default : :py:class:`ConvLayer`).
    :param bool return_indices: returns pooling indices (default : True)
    """
    conv_layer = DeconvLayer
    separate_heads = False

    @staticmethod
    def init_modules(pins, phidden, return_indices=False, conv_layer=ConvLayer):
        # conditioning
        if phidden.get('label_params'):
            phidden['conditioning'] = phidden.get('conditioning', 'concat')
            if phidden['conditioning'] == 'concat':
                phidden['channels'] = list(phidden['channels'])
                phidden['channels'][0] = phidden['channels'][0] + sum([p['dim'] for p in phidden['label_params'].values()])
        channels = phidden['channels']
        conv_modules = []
        # conv_dim = len(checktuple())
        for l in range(len(phidden['channels'])-1):
            current_dict = {'kernel_size':phidden['kernel_size'][l], 'stride':phidden['stride'][l], 'output_padding':phidden['output_padding'][l], 'pool':phidden['pool'][l],
                            'padding':phidden['padding'][l], 'conv_dim':phidden['conv_dim'], 'normalization':phidden['norm_conv'][l], 'dropout':phidden['dropout_conv'][l], 'dilation':phidden['dilation'][l]}
            conv_modules.append(conv_layer(channels[l], channels[l+1], return_indices=return_indices, **current_dict))
        return nn.ModuleList(conv_modules)

    @property
    def input_ndim(self):
        return self.conv_dim

    @flatten_seq_method
    def forward(self, x, y=None, indices=None, output_size=None):
        """
        forward method of the Deconvolution module. can take a list of input, and additional label information in case.
        Also requires indices in case of unpooling operations, and a specific output shape can be precised if needed
        using output padding.
        :param x: input tensors of module
        :type x: list or torch.Tensor
        :param dict y: label input (default : None)
        :param list indices: corresponding pooling indices (default : None. required if unpooling is processed).
        :param tuple output_size: desired output shape
        :return: convoluted tensor
        :rtype: torch.Tensor
        """
        if issubclass(type(x), list):
            x = torch.cat(x, )

        if self.phidden.get('label_params'):
            assert y is not None, "Convolutional module need following labels : %s"%list(self.phidden['label_params'].keys())
            y = format_label_data(self, y, self.phidden)
            current_device = next(self.parameters()).device
            x = torch.cat((x, get_label_channel(x, y, device=current_device)), dim=1)

        out = x.clone()
        # successively pass into convolutional modules
        for l in range(self.depth-1):
            #print('deconv : ', l, out.shape)
            out_size = None; ind = None
            if indices is not None:
                ind = indices[-(l+1)],
            if output_size is not None:
                out_size = output_size[l]
            out = self.conv_encoders[l](out, indices=ind, output_size=out_size)
        return out



# Multi-head convolutional containers

class MultiHeadConvolutional(Convolutional):
    """
    MultiHeadConvolutional implements multiple py:class:`Convolutional` modules, called *heads*, that are finally summed
    using learnable weights.

    arguments : see :py:class:`Convolutional`
    """
    separate_heads = True
    def __init__(self, pins, phidden, *args, **kwargs):
        super(MultiHeadConvolutional, self).__init__(pins, phidden, *args, **kwargs)
        if not self.separate_heads:
            self.head_weights = nn.ParameterList([nn.Parameter(torch.randn(1)) for i in range(len(self.conv_encoders))])

    @staticmethod
    def init_modules(pins, phidden, return_indices=False, conv_module=ConvLayer):
        num_heads = phidden.get('heads', 1)
        heads = [None]*num_heads
        for i in range(num_heads):
            heads[i] = Convolutional(pins, phidden, return_indices=return_indices, conv_layer=conv_module)
        return nn.ModuleList(heads)

    def get_output_conv_length(self, params=None, input_dim=None):
        pre_dims, post_dims = super(MultiHeadConvolutional, self).get_output_conv_length(params=params, input_dim=input_dim)
        post_dims[-1] *= self.phidden['heads']
        # pre_dims[-1] *= self.phidden['heads']
        return pre_dims, post_dims


    def forward(self, *args, **kwargs):
        individual_outs = []
        for head in self.conv_encoders:
            individual_outs.append(head.forward(*args, **kwargs))
        if self.separate_heads:
            return torch.cat([i.unsqueeze(1) for i in individual_outs], dim=1)
        else:
            summed_out = sum([self.head_weights[i] * out for i, out in enumerate(individual_outs)])
            return summed_out


class MultiHeadDeconvolutional(Deconvolutional):
    """
    MultiHeadDeconvolutional implements multiple py:class:`Deconvolutional` modules, called *heads*, that are finally summed
    using learnable weights.

    arguments : see :py:class:`Deconvolutional`
    """
    conv_module = DeconvLayer
    separate_heads = True
    def __init__(self, pins, phidden, *args, **kwargs):
        super(MultiHeadDeconvolutional, self).__init__(pins, phidden, *args, **kwargs)
        self.head_weights = nn.ParameterList([nn.Parameter(torch.randn(1)) for i in range(len(self.conv_encoders))])

    @staticmethod
    def init_modules(pins, phidden, return_indices=False, conv_module=ConvLayer):
        num_heads = phidden.get('heads', 1)
        heads = [None]*num_heads
        for i in range(num_heads):
            heads[i] = Deconvolutional(pins, phidden, return_indices=return_indices, conv_layer=conv_module)
        return nn.ModuleList(heads)

    def forward(self, *args, **kwargs):
        individual_outs = []
        for i, head in enumerate(self.conv_encoders):
            individual_outs.append(self.head_weights[i] * head.forward(*args, **kwargs))
        if self.separate_heads:
            return torch.cat([i.unsqueeze(1) for i in individual_outs], dim=1)
        else:
            summed_out = sum(individual_outs)
            return summed_out


###
# Convolution + MLP containers for variational auto-encoders

class ConvolutionalLatent(nn.Module):
    take_sequence = False
    conv_class = Convolutional
    conv_layer = ConvLayer
    dump_patches = True
    in_channels = True

    def __init__(self, pins, phidden, *args, **kwargs):
        """
        The ConvolutionalLatent class is a container composed by a :py:class:`Convolutional` module, performing
        multi-layer convolution of an input, and a :py:class:`vschaos.modules.MLP` module, flattening channels and
        targetting a given latent dimension. Flattening can be done in two ways : the *unflatten* mode, that flattens
        the channels, and the *conv1x1* mode, that adds a convolution layer with 1 output channel.

        In addition to the parameters taken by the :py:class:`Convolutional` module, :py:class:`ConvolutionalLatent` takes
        * *conv2mlp* (str) : flattening strategy (choices : *flatten* or *conv1x1*)
        * *conv_class* (nn.Module) : Convolutional module class
        * *conv_layer* (nn.Module) : Convolutional layer class

        :param pins: input paramters. can be multi-input
        :type pins: list or dict
        :param dict phidden: hidden parameters. See :py:class:`Convolutional` for valid keywords

        """
        super(ConvolutionalLatent, self).__init__()

        # init convolutional module
        self.pins = checklist(pins); self.phidden = checklist(phidden, n=len(self.pins))

        # make a Convolutional module for each input
        conv_modules = []; self.return_indices = False
        for i in range(len(self.pins)):
            conv_class = self.phidden[i].get('conv_class', self.conv_class)
            conv_layer = self.phidden[i].get('conv_layer', self.conv_layer)
            conv_modules.append(conv_class(self.pins[i], self.phidden[i], conv_layer=conv_layer, *args, **kwargs))
            self.return_indices = self.return_indices or conv_modules[-1].return_indices
        self.conv_modules = nn.ModuleList(conv_modules)

        # make flattening modules
        transfer_modules = [None]*len(self.phidden); transfered_sizes = [None]*len(self.phidden)
        output_size = [p.get_output_conv_length(input_dim=self.pins[i]['dim'])[1][-1] for i, p in enumerate(self.conv_modules)]
        for i in range(len(self.phidden)):
            transfer_modules[i], transfered_sizes[i] = self.get_flattening_module(output_size[i], self.phidden[i])
        self.transfer_modules = nn.ModuleList(transfer_modules)
        mlp_input_size = sum(transfered_sizes)

        # make post encoders
        post_encoder_params = {'dim':phidden.get('dim'), 'nlayers':phidden.get('nlayers')}
        self.post_encoder = self.get_post_mlp(mlp_input_size, post_encoder_params,  *args, **kwargs)


    @staticmethod
    def get_flattening_module(input_dim, phidden, *args, **kwargs):
        transfer_mode = phidden.get('transfer', 'unflatten')
        if transfer_mode == "unflatten":
            return Flatten(),  int(np.cumprod(input_dim)[-1])*phidden['channels'][-1]
        elif transfer_mode == "conv1x1":
            n_channels = phidden['channels'][-1]
            return Sequential(ConvLayer.conv_modules[phidden['conv_dim']](n_channels, 1, kernel_size=1), Flatten()), input_dim

    @staticmethod
    def get_post_mlp(input_dim, phidden, nn_lin="ReLU",  *args, **kwargs):
        return MLP({'dim':input_dim}, phidden, nn_lin=nn_lin, *args, **kwargs)

    def get_output_conv_length(self, *args, **kwargs):
        """
        returns layerwise output shapes of the Convolutional module, before and after pooling
        :param params: hidden parameters (default : instance's hidden parameters)
        :type params: dict or None
        :param input_dim: input parameters (default : instances's input parameters)
        :type input_dim: list, dict or None
        :return: pre-pooling shapes, post-pooling shapes
        :rtype: (tuple, tuple)
        """
        return [conv_module.get_output_conv_length(*args, **kwargs) for conv_module in self.conv_modules]

    def get_pooling_indices(self, layer=None):
        """
        get pooling indices of the convolutional layers. If not any layer is precised, the pooling indices of every
        layers are returned.
        :param int layer: precise layer (default : None)
        :return: pooling indices
        :rtype: list or torch.Tensor
        """
        if layer is None:
            return [conv_module.get_pooling_indices() for conv_module in self.conv_modules]
        else:
            return [conv_module.get_pooling_indices()[layer] for conv_module in self.conv_modules]

    def forward(self, x, *args, y=None, **kwargs):
        """
        forward data in the convolution process
        :param x: data
        :type x: list or torch.Tensor
        :param dict y: label information
        :return: hidden representation
        :rtype: torch.Tensor
        """
        x = checklist(x)
        conv_outs = []
        for i, inp in enumerate(x):
            if len(inp.shape) == 2:
                inp = inp.unsqueeze(1)
            # forward!
            conv_out = self.conv_modules[i](inp, y=y, *args, **kwargs)
            conv_outs.append(self.transfer_modules[i](conv_out, dim=self.conv_modules[i].conv_dim+1))
        conv_out = torch.cat(conv_outs, dim=-1)
        out = self.post_encoder(conv_out, y=y)
        return out


class DeconvolutionalLatent(ConvolutionalLatent):
    """
    The DeconvolutionalLatent class is a container composed by a :py:class:`Deconvolutional` module, performing
    multi-layer convolution of an input, and a :py:class:`vschaos.modules.MLP` module, flattening channels and
    targetting a given latent dimension. Flattening can be done in two ways : the *unflatten* mode, that flattens
    the channels, and the *conv1x1* mode, that adds a convolution layer with 1 output channel.

    The hidden parameters of the module are reversed at initialization, such that the DeconvolutionalLatent arguments
    are the same than a corresponding ConvolutionalLatent parametrization.

    In addition to the parameters taken by the :py:class:`Convolutional` module, :py:class:`ConvolutionalLatent` takes
    * *mlp2conv* (str) : flattening strategy (choices : *flatten* or *conv1x1*)
    * *conv_class* (nn.Module) : Convolutional module class
    * *conv_layer* (nn.Module) : Convolutional layer class

    :param pins: input paramters. can be multi-input
    :type pins: list or dict
    :param dict phidden: hidden parameters. See :py:class:`Convolutional` for valid keywords
    :param nn.Module encoder: mirrored encoder, that is needed if the module has a pooling procedure.
    :param dict pout: (optional) output parameters, allowing to speciify a target output dimension with the *dim* keyword.
    """
    conv_class = Deconvolutional
    conv_layer = DeconvLayer

    @property
    def in_channels(self):
        return self.conv_dim

    def __init__(self, pins, phidden, encoder=None, pouts=None,  *args, **kwargs):
        nn.Module.__init__(self)
        self.pins = pins; self.phidden = checklist(phidden, copy=True)
        self.has_pooling = False

        self.pre_pooling_sizes_dec = [None] * len(pins); self.post_pooling_sizes_dec = [None] * len(pins)
        self.output_padding = [None] * len(pins); self.pool_target_sizes = [None] * len(pins)
        self.output_size = [None] * len(pins)
        conv_modules = []
        transfer_modules = [None]*len(self.phidden); transfered_sizes = [None]*len(self.phidden)
        for i, ph in enumerate(self.phidden):
            has_pooling = ph.get('pool') is not None and not set(ph.get('pool')) == {None}
            self.has_pooling |= has_pooling
            # if an encoder is given, mirror the encoding process
            # ph = dict(encoder.phidden) if encoder and self.has_pooling else dict(phidden)

            # reverse parameters to mirror the given encoding process
            if self.has_pooling:
                assert encoder is not None, "encoder is required to get pack indices during unpooling"
                current_ins = encoder.pins
                if isinstance(current_ins, list):
                    current_ins = current_ins[0]
                conv_dim = len(checktuple(current_ins['dim']))
            else:
                conv_dim = 1 if pouts is None else len(checktuple(checklist(pouts)[i]['dim']))
            self.conv_dim = conv_dim
            conv_class = ph.get('conv_class', self.conv_class)
            conv_layer = ph.get('conv_layer', self.conv_layer)
            self.phidden[i]['channels'] = checklist(ph['channels'], rev=True)
            self.phidden[i]['stride'] = checklist(ph.get('stride', 1), len(ph['channels']), rev=True)
            self.phidden[i]['dilation'] = checklist(ph.get('dilation', 1), len(ph['channels']), rev=True)
            self.phidden[i]['pool'] = checklist(ph.get('pool'), len(ph['channels']), rev=True)
            self.phidden[i]['kernel_size'] = list(reversed([tuple(checklist(ks, conv_dim))for ks in self.phidden[i]['kernel_size']]))
            self.phidden[i]['padding'] = [tuple(np.ceil(np.array(ks)/2).astype('int').tolist()) for ks in self.phidden[i]['kernel_size']]
            self.phidden[i]['output_padding'] = checklist(ph.get('output_padding', 0), len(ph['channels']), rev=True)
            self.phidden[i]['conv_dim'] = ph.get('conv_dim') or len(checktuple(checklist(pouts)[i]['dim']))

            # fit layers' output size to fit the incoming unpooling indices in case
            if has_pooling:
                if encoder is None:
                    raise Warning('DeconvolutionalLatent module has to be initialized with a valid Convolutional module')
                pre_pooling_sizes_enc, post_pooling_sizes_enc = encoder.conv_modules[0].get_output_conv_length()
                self.pre_pooling_sizes_dec[i], self.post_pooling_sizes_dec[i], self.output_padding[i], self.pool_target_sizes[i] = self.get_output_conv_length(self.phidden[i], post_pooling_sizes_enc[-1], pre_pooling_sizes_enc, post_pooling_sizes_enc)
                self.phidden[i]['output_padding'] = self.output_padding[i]
                self.output_size[i] = post_pooling_sizes_enc[-1]
            else:
                self.output_size[i], self.output_padding[i] = self.get_pre_mlp_dimension(pouts or {'dim':self.phidden['dim']}, self.phidden[i])
                self.phidden[i]['output_padding'] = self.output_padding[i]
                self.pre_pooling_sizes_dec[i], self.post_pooling_sizes_dec[i], _,  self.pool_target_sizes = self.get_output_conv_length(self.phidden[i], input_dim=self.output_size[i])

            # init module
            conv_modules.append(conv_class(checklist(self.pins, n=len(self.phidden))[i], self.phidden[i], conv_layer=conv_layer, *args, **kwargs))
            self.conv_modules = nn.ModuleList(conv_modules)
            input_size = tuple(self.post_pooling_sizes_dec[i][0].tolist())
            transfer_modules[i], transfered_sizes[i] = self.get_flattening_module(input_size, self.phidden[i])
           # self.separate_heads = self.conv_module.separate_heads
            # self.return_indices = self.conv_module.return_indices

        # make pre encoder
        self.transfer_modules = nn.ModuleList(transfer_modules)
        self.transferred_size = transfered_sizes
        if ph.get('requires_indices'):
            self.encoder = encoder
        pre_mlp_dims = checklist(self.phidden[0]['dim'], n=self.phidden[0].get('nlayers', len(checklist(self.phidden[0]['dim']))))
        pre_mlp_dims[-1] = sum(transfered_sizes)
        pre_mlp_params = {'dim':pre_mlp_dims, 'nlayers':len(pre_mlp_dims), 'layer':self.phidden[0].get('layer'), 'normalization':self.phidden[0].get('normalization'), 'dropout':self.phidden[0].get('dropout')}
        self.pre_mlp = self.get_pre_mlp(self.pins, pre_mlp_params, *args, **kwargs)
        


    def get_output_conv_length(self, params=None, input_dim=None, target_pre=None, target_post=None):
        """
        returns layer-wise output shapes of the Convolutional module, before and after pooling.
        Can also be used to compute adequate output paddings to fit a mirrored encoding process.
        :param params: hidden parameters (default : instance's hidden parameters)
        :type params: dict or None
        :param input_dim: input parameters (default : instances's input parameters)
        :type input_dim: list, dict or None
        :param list target_pre: target pre-pooling shapes (used to compute output paddings)
        :param target_post: taret post-pooling shapes (used to copuute output paddings)

        :return: pre-pooling shapes, post-pooling shapes, output paddings, pool target sizes
        :rtype: (tuple, tuple, tuple, tuple)
        """
        current_length = input_dim if not input_dim is None else self.output_size
        # if issubclass(type(current_length), list):
        #     params = checklist(params, n=len(current_length))
        #     outs = [self.get_output_conv_length(params[i], input_dim = self.output_size[i], target_pre=target_pre, target_post=target_post) for i in range(len(current_length))]
        #     return tuple(zip(outs))
        params = params or self.phidden
        current_output = current_length.copy()
        pre_pooling_dims = []; post_pooling_dims = []
        output_paddings = []; pool_target_sizes = []
        
        for l in range(len(params['kernel_size'])):
            # pooling layers
            pool_target_sizes.append(None)
            if not params['pool'][l] is None:
                current_output = (current_output-1)*np.array(params['pool'][l]).astype('int') + np.array(params['pool'][l]).astype('int')
                if target_pre:
                    if (current_output != target_pre[-(l+1)]).any():
                        pool_target_sizes[-1] = target_pre[-(l+1)].astype('int')
                        current_output = target_pre[-(l+1)].astype('int')
            post_pooling_dims.append(current_output)

            # conv layers
            current_output = (current_output-1)*params['stride'][l] - 2*np.array(params['padding'][l]) + np.array(params['dilation'][l]).astype('int') * (np.array(params['kernel_size'][l]) - 1).astype('int') + 1
            if params.get('output_padding'):
                current_output = current_output + params['output_padding'][l]

            if target_post:
                if l <= len(params['kernel_size'])-2:
                    output_paddings.append(tuple((target_post[-(l+2)].astype('int') - current_output.astype('int')).tolist()))
                    current_output += output_paddings[-1]
                else:
                    output_paddings.append(tuple(np.zeros_like(current_output, dtype='int').tolist()))
            pre_pooling_dims.append(current_output.astype('int'))
 
        return pre_pooling_dims, post_pooling_dims, output_paddings, pool_target_sizes

    def get_pre_mlp_dimension(self, pins, phidden):
        """
        returns the adequate MLP output dimensions to fit given input parameters.?
        :param pins: input parameters
        :param phidden: hidden parameters
        :return: output_dims, output_paddings
        :rtype: list, list
        """
        input_dim = sum([np.array(p['dim']) for p in checklist(pins)])
        output_dim = np.array(input_dim)
        n_layers = len(phidden['channels'])
        output_paddings = []
        for n in reversed(range(n_layers)):
            new_dim = np.floor((output_dim + 2*np.array(phidden['padding'][n]) - np.array(phidden['dilation'][n]) * (np.array(phidden['kernel_size'][n])-1) - 1) / np.array(phidden['stride'][n]) + 1)
            predicted_dim = (new_dim - 1) * np.array(phidden['stride'][n]) - 2 * np.array(phidden['padding'][n]) + np.array(phidden['dilation'][n]) * (np.array(phidden['kernel_size'][n]) - 1) + 1
            output_paddings.append(tuple((output_dim - predicted_dim).astype('int').tolist()))
            output_dim = new_dim
        if output_dim.shape == tuple():
            output_dim = np.expand_dims(output_dim, 0)
        output_paddings = list(reversed(output_paddings))
        return output_dim, output_paddings

    @staticmethod
    def get_pre_mlp(pin, phidden, transfer_mode="unflatten", n_channels=None, reshape=None, *args, **kwargs):
        if transfer_mode == "unflatten":
            return MLP(pin, phidden, *args, **kwargs)
        elif transfer_mode == "conv1x1":
            conv_module = ConvLayer.conv_modules[phidden['conv_dim']](1, n_channels, kernel_size = 1, bias=False)
            nn.init.xavier_normal_(conv_module.weight); #nn.init.zeros_(conv_module.bias)
            if reshape is None:
                return Sequential(MLP(pin, phidden, *args, **kwargs), Unsqueeze(1), conv_module)
            else:
                return Sequential(MLP(pin, phidden, *args, **kwargs), Reshape(reshape), conv_module)

    @staticmethod
    def get_flattening_module(input_dim, phidden, *args, **kwargs):
        transfer_mode = phidden.get('transfer', 'unflatten')
        n_channels = phidden['channels'][0]
        if transfer_mode == "unflatten":
            return Reshape([n_channels, *input_dim]), np.cumprod(input_dim)[-1] * n_channels
        elif transfer_mode == "conv1x1":
            return Sequential(Unsqueeze(1), ConvLayer.conv_modules[phidden['conv_dim']](1, n_channels, kernel_size=1)), input_dim


    def get_pooling_indices(self, dim=0):
        """
        get pooling indices of the convolutional layers. If not any layer is precised, the pooling indices of every
        layers are returned.
        :param int layer: precise layer (default : None)
        :return: pooling indices
        :rtype: list or torch.Tensor
        """
        indices = None
        if self.has_pooling:
            if isinstance(self.encoder, nn.ModuleList):
                indices = [enc.get_pooling_indices(dim) for enc in self.encoder]
            else:
                indices = self.encoder.get_pooling_indices(dim)
        return indices

    def forward(self, x, *args, y=None, **kwargs):
        """
        forward function of the DeconvolutionalLatent module.
        :param x:
        :param args:
        :param y:
        :param kwargs:
        :return:
        """
        # post_mlp_size = [int(o) for o in self.output_size]

        original_shape = None
        # if issubclass(type(x), list):
        #     x = torch.cat(x, dim=-1)
        pre_output = self.pre_mlp(x, y=y)
        pre_output = split_select(pre_output, self.transferred_size)
        outs = []
        for i in range(len(pre_output)):
            pre_output[i] = self.transfer_modules[i](pre_output[i])
            pooling_indices = None
            if self.has_pooling:
                pooling_indices = self.encoder.get_pooling_indices()[0]
            outs.append(self.conv_modules[i](pre_output[i], y=y, indices=pooling_indices, output_size=self.pool_target_sizes[i]))
        if not issubclass(type(self.pins), list):
            outs = outs[0]
        return outs


class MultiHeadConvolutionalLatent(ConvolutionalLatent):
    """
    MultiHeadConvolutionalLatent is a :py:class:`ConvolutionalLatent` using multi-head convolution.
    Is similar than specifying conv_class = :py:class:`MultiHeadConvolutional` in module's parameters.
    """
    conv_class = MultiHeadConvolutional

class MultiHeadDeconvolutionalLatent(DeconvolutionalLatent):
    """
    MultiHeadDeconvolutionalLatent is a :py:class:`DeconvolutionalLatent` using multi-head convolution.
    Is similar than specifying conv_class = :py:class:`MultiHeadDeconvolutional` in module's parameters.
    """
    conv_class = MultiHeadDeconvolutional


class GatedConvolutionalLatent(ConvolutionalLatent):
    """
    GatedConvolutionalLatent is a :py:class:`ConvolutionalLatent` using multi-head convolution.
    Is similar than specifying conv_layer = :py:class:`GatedConvLayer` in module's parameters.
    """
    conv_layer = GatedConvLayer

class GatedDeconvolutionalLatent(DeconvolutionalLatent):
    """
    GatedDeconvolutionalLatent is a :py:class:`DeconvolutionalLatent` using multi-head convolution.
    Is similar than specifying conv_layer = :py:class:`GatedDeconvLayer` in module's parameters.
    """
    conv_layer = GatedDeconvLayer


class MultiHeadGatedConvolutionalLatent(MultiHeadConvolutionalLatent):
    """
    MultiHeadGatedConvolutionalLatent is a :py:class:`ConvolutionalLatent` using multi-head convolution and gated layers.
    Is similar than specifying conv_layer = :py:class:`GatedConvLayer` and conv_class = :py:class:`MultiHeadConvolutional`
    in module's parameters.
    """
    conv_layer = GatedConvLayer

class MultiHeadGatedDeconvolutionalLatent(MultiHeadDeconvolutionalLatent):
    """
    MultiHeadGatedDeconvolutionalLatent is a :py:class:`DeconvolutionalLatent` using multi-head convolution and gated layers.
    Is similar than specifying conv_layer = :py:class:`GatedDeconvLayer` and conv_class = :py:class:`MultiHeadDeconvolutional`
    in module's parameters.
    """
    conv_layer = GatedDeconvLayer

