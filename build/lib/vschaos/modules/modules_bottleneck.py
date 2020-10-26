#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:49:58 2017

@author: chemla
"""
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from . import format_label_data
from ..utils import checklist, checktuple, flatten_seq_method
from . import init_module, MLP_DEFAULT_NNLIN, CONDITIONING_HASH, flatten, modules_nonlinearity as vsnl, Sequential
# Full modules for variational algorithms


class MLPLayer(nn.Module):
    """
    Base layer for a fully-connected network.

    Parameters
    ----------
    input_dim : input dimensions (int)
    options : dict
        native options:
            dataPrefix (str) : data root of dataset
            dataDirectory (str) : audio directory of dataset (default: dataPrefix + '/data')
            analysisDirectory (str) : transform directory of dataset (default: dataPrefix + '/analysis')
            metadataDirectory (str) : metadata directory of dataset (default: dataPrefix + '/metadata')
            tasks [list(str)] : tasks loaded from the dataset
            taskCallback[list(callback)] : callback used to load metadata (defined in data_metadata)
            verbose (bool) : activates verbose mode
            forceUpdate (bool) : forces updates of imported transforms
            checkIntegrity (bool) : check integrity of  files

    Returns
    -------
    A new dataset object

    Example
    -------
    """
    dump_patches = True
    nn_lin = "ELU"
    def __init__(self, input_dim, output_dim, nn_lin=None, nn_lin_args={}, normalization='batch', dropout=None, name_suffix="", bias=True, *args, **kwargs):
        """
        :param input_dim: input dimension
        :type input_dim: int
        :param output_dim: output dimension
        :type output_dim: int
        :param nn_lin: non-linearity
        :type nn_lin: `str` or `nn.Module`
        :param batch_norm: batch normalization (batch or instance)
        :type batch_norm: str
        :param dropout: dropout probability (0.0: no dropout 1.0: full dropout)
        :type dropout: float
        :param name_suffix: specific layer name
        :type name_suffix: str
        :param bias: does module have bias
        :type bias: bool
        :returns: MLPLayer
        """
        super(MLPLayer, self).__init__()
        self.input_dim = input_dim; self.output_dim = output_dim
        self.name_suffix = name_suffix; self.normalization = normalization

        modules = OrderedDict()
        # Linear module
        modules["hidden"+name_suffix] =  nn.Linear(input_dim, output_dim, bias=bias)

        # nn.init.xavier_normal_(modules["hidden"+name_suffix].weight)
        # if bias:
        #     nn.init.zeros_(modules["hidden"+name_suffix].bias)
        # Batch / Instance Normalization
        if normalization == 'batch':
            modules["batch_norm_"+name_suffix]= nn.BatchNorm1d(output_dim)
        if normalization == 'instance':
            modules["instance_norm_"+name_suffix]= nn.InstanceNorm1d(1)
        # Dropout
        if not dropout is None:
            modules['dropout_'+name_suffix] = nn.Dropout(dropout)
        # Non Linearity
        self.nn_lin = nn_lin
        if nn_lin:
            if hasattr(vsnl, nn_lin):
                modules["nnlin" + name_suffix] = getattr(vsnl, nn_lin)(**nn_lin_args)
            elif hasattr(nn, nn_lin):
                modules["nnlin"+name_suffix] = getattr(nn, nn_lin)(**nn_lin_args)
            else:
                raise AttributeError('non linearity %s not found'%nn_lin)
        init_module(modules["hidden" + name_suffix], self.nn_lin)
        self.module = nn.Sequential(modules)

    @flatten_seq_method
    def forward(self, x):
        """
        processes the input.
        if the input tensor has more than two dimensions, the tensor is flattened over the first dim.
        :param x: input to process
        :type x: torch.Tensor
        :return torch.Tensor:
        """
        out = self.module._modules['hidden'+self.name_suffix](x)
        if self.normalization == 'batch':
            out = self.module._modules['batch_norm_'+self.name_suffix](out)
        elif self.normalization == 'instance':
            out = self.module._modules['instance_norm_'+self.name_suffix](out.unsqueeze(1))
            out = out.squeeze()
        if self.nn_lin:
            out = self.module._modules['nnlin'+self.name_suffix](out)
        return out



class MLPResidualLayer(MLPLayer):
    """
    Residual version of :class:`MLPLayer`. Input dimension must equal output dimension
    if the input tensor has more than two dimensions, the tensor is flattened over the first dim.
    """
    def forward(self, x):
        """
        processes the input
        :param x: input to process
        :type x: torch.Tensor
        :return torch.Tensor:
        """
        if len(x.shape) > 2:
            x = flatten(x)
        out = super(MLPResidualLayer, self).forward(x)
        if self.input_dim == self.output_dim:
            out = nn.functional.relu(out + x)
        return out


class MLPGatedLayer(MLPLayer):
    """
    Gated version of :class:`MLPLayer`.
    if the input tensor has more than two dimensions, the tensor is flattened over the first dim.
    """
    def __init__(self, input_dim, output_dim, normalization='batch', *args, **kwargs):
        super(MLPGatedLayer, self).__init__(input_dim, output_dim, normalization=None, *args, **kwargs)
        self.gate_module = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Sigmoid())
        if normalization == 'batch':
            self.batch_module = nn.BatchNorm1d(output_dim)
        if normalization == 'instance':
            self.batch_module = nn.InstanceNorm1d(1)
        self.nn_lin = kwargs.get('nn_lin', MLP_DEFAULT_NNLIN)

    @flatten_seq_method
    def forward(self, x):
        """
        processes the input
        :param x: input to process
        :type x: torch.Tensor
        :return torch.Tensor:
        """
        out_module = self.module(x)
        out_gated = self.gate_module(x)
        out_module = out_module * out_gated
        if hasattr(self, "batch_module"):
            out_module = self.batch_module(out_module)
        if self.nn_lin:
            out_module = getattr(nn, self.nn_lin)()(out_module)
        return out_module

class MLPGatedResidualLayer(MLPGatedLayer):
    """
    Gated version of :class:`MLPLayer`. Input dimension must equal output dimension
    if the input tensor has more than two dimensions, the tensor is flattened over the first dim.
    """
    def forward(self, x):
        """
        processes the input
        :param x: input to process
        :type x: torch.Tensor
        :return torch.Tensor:
        """
        out = super(MLPGatedResidualLayer, self).forward(x)
        if self.input_dim == self.output_dim:
            out = nn.functional.relu(out + x)
        return out


class MLP(nn.Module):
    DefaultLayer = MLPLayer
    dump_patches = True
    take_sequences = False
    ''' Generic layer that is used by generative variational models as encoders, decoders or only hidden layers.'''
    def __init__(self, pins, phidden, linked=False, name="", *args, **kwargs):
        ''':param pins: Input properties.
        :type pins: dict or [dict]
        :param phidden: properties of hidden layers.
        :type phidden: dict
        :param name: name of module
        :type name: string'''
        
        super(MLP, self).__init__()

        # set input parameters
        self.pins = pins
        self.phidden = phidden

        # get hidden layers
        self.linked = linked
        self.hidden_module = self.get_hidden_layers(pins, phidden)
        self.embeddings = self.get_embeddings(phidden)

    @staticmethod
    def get_embeddings(phidden):
        if isinstance(phidden, list):
            embeddings_list = [MLP.get_embeddings(ph) for ph in phidden]
            print([list(k.keys()) for k in embeddings_list])
            keys = set(sum([list(k.keys()) for k in embeddings_list], []))
            embeddings = {k:[None]*len(phidden) for k in keys}
            for i, em in enumerate(embeddings_list):
                for k in em.keys():
                    embeddings[k][i] = em[k]
            return embeddings
        else:
            embeddings = nn.ModuleDict()
            if phidden.get('label_params') is None:
                return embeddings
            for k, pl in phidden.get('label_params', {}).items():
                if pl.get('embedding') is not None:
                    embeddings[k] = pl['embedding']
            return embeddings


    def get_input_dimension(self, pins, phidden):
        # accumulate dimensions in input parameters
        input_dim = sum([np.cumprod(checktuple(p['dim']))[-1] for p in checklist(pins)])
        # add dimensions in  case of concatenative conditioning
        is_conditioned = phidden.get('label_params')
        conditioning = phidden.get('conditioning', 'concat')
        dims = []
        if is_conditioned and conditioning == 'concat':
            for k, v in is_conditioned.items():
                if v.get('embedding') is None:
                    dims.append(v['dim'])
                else:
                    dims.append(v['embedding'].out_dim)
        input_dim += sum(dims)
        return input_dim

    def get_hidden_layers(self, pins, phidden={"dim":800, "nlayers":2, 'label':None, 'nn_lin':MLP_DEFAULT_NNLIN, 'conditioning':'concat', "linked":False}):
        ''' outputs the hidden module of the layer.
        :param input_dim: dimension of the input
        :type input_dim: int
        :param phidden: parameters of hidden layers
        :type phidden: dict
        :param nn_lin: non-linearity name
        :type nn_lin: str
        :param name: name of module
        :type name: str
        :returns: nn.Sequential '''

        # recursive if phidden is a list
        if issubclass(type(phidden), list):
            if self.linked:
                return nn.ModuleList([self.get_hidden_layers(pins, ph) for ph in phidden])
            else:
                assert len(pins) == len(phidden)
                return nn.ModuleList([self.get_hidden_layers(pins[i], phidden[i]) for i in range(len(pins))])


        # get input dimension
        input_dim = self.get_input_dimension(pins, phidden)

        # check number of layers
        phidden = dict(phidden)
        nlayers = phidden.get('nlayers', 2) or 2
        if issubclass(type(phidden['dim']), list):
            assert nlayers <= len(phidden['dim']), '%d dimensions are given but %d layers are specified'%(len(phidden['dim']), phidden['nlayers'])

        # get layer parameters
        LayerModule = checklist(phidden.get('layer', self.DefaultLayer) or MLPLayer, n=nlayers)
        phidden['dim'] = checklist(phidden['dim'], n=nlayers)
        phidden['nn_lin'] = checklist(phidden.get('nn_lin', [n.nn_lin for n in LayerModule]), n=nlayers)
        phidden['normalization'] = checklist(phidden.get('normalization'), n=nlayers)
        phidden['dropout'] = checklist(phidden.get('dropout'), n=nlayers)
        phidden['bias'] = checklist(phidden.get('bias', True), n=nlayers)
        self.flatten = phidden.get('flatten', False)

        # make modules
        modules = OrderedDict()
        for i in range(nlayers):
            hidden_dim = int(phidden['dim'][i])
            n_in = int(input_dim) if i==0 else int(phidden['dim'][i-1])
            modules['layer_%d'%i] = LayerModule[i](n_in, hidden_dim, nn_lin=phidden['nn_lin'][i],
                                                   normalization=phidden['normalization'][i], dropout=phidden['dropout'][i],
                                                   bias=phidden['bias'][i], name_suffix="_%d"%i)
        hidden_module = Sequential(modules)
        return hidden_module


    def forward(self, x, y=None, sample=True, *args, **kwargs):
        '''outputs parameters of corresponding output distributions
        :param x: input or vector of inputs.
        :type x: torch.Tensor or [torch.Tensor ... torch.Tensor]
        :param outputHidden: also outputs hidden vector
        :type outputHidden: True
        :returns: (torch.Tensor..torch.Tensor)[, torch.Tensor]'''

        # Concatenate input if module is a list
        if type(self.phidden) == list:
            if type(self.pins) == list:
                x = checklist(x, n=len(self.pins))
                for i, ph in enumerate(self.phidden):
                    x[i] = flatten(x[i], len(x[i].shape) - len(checktuple(self.pins[i]['dim'])))
                    if ph.get('label_params') and ph.get('conditioning', 'concat'):
                        assert y, 'need label if hidden module is conditioned'
                        y_tmp = format_label_data(self, y, ph)
                        x[i] = torch.cat((x[i], y_tmp), -1)
                out = [self.hidden_module[i](x[i]) for i in range(len(x))]
            else:
                x = flatten(x, len(x.shape) - len(checktuple(self.pins['dim'])))
                x_tmp = [x]*len(self.phidden)
                for i, ph in enumerate(self.phidden):
                    if ph.get('label_params') and ph.get('conditioning', 'concat'):
                        assert y, 'need label if hidden module is conditioned'
                        y_tmp = format_label_data(self, y, ph)
                        x_tmp[i] = torch.cat((x, y_tmp), -1)
                out = [self.hidden_module[i](x_tmp[i]) for i in range(len(self.phidden))]
        else:
            if type(self.pins) == list:
                x = checklist(x, n=len(self.pins))
                # for i in range(len(x)):
                #     x[i] = flatten(x[i], len(x[i].shape) - len(checktuple(self.pins[i]['dim'])))
                x = torch.cat(checklist(x, n=len(self.pins)), dim=-1)
            if self.phidden.get('label_params') and self.phidden.get('conditioning', 'concat'):
                y_tmp = {k:y[k] for k in self.phidden['label_params'].keys()}
                y_tmp = format_label_data(self, y_tmp, self.phidden)
                x = torch.cat([x, y_tmp], dim=-1)
            out = self.hidden_module(x, **kwargs)

        return out


"""(deprecated)
class DLGMLayer(nn.Module):
    dump_patches = True
    ''' Specific decoding module for Deep Latent Gaussian Models'''
    def __init__(self, pins, pouts, nn_lin=MLP_DEFAULT_NNLIN, name="", **kwargs):
        '''
        :param pins: parameters of the above layer
        :type pins: dict
        :param pouts: parameters of the ouput distribution
        :type pouts: dict
        :param phidden: parameters of the hidden layer(s)
        :type phidden: dict
        :param nn_lin: non-linearity name
        :type nn_lin: str
        :param name: name of module
        :type name: str'''
        
        super(DLGMLayer, self).__init__()
                
        if issubclass(type(pouts), list):
            self.out_module = nn.ModuleList()
            self.cov_module = nn.ModuleList()
            for pout in pouts:
                self.out_module.append(nn.Linear(pins['dim'], pout['dim']))
                init_module(self.out_module, 'Linear')
                self.cov_module.append(nn.Sequential(nn.Linear(pout['dim'], pout['dim']), nn.Sigmoid()))
                init_module(self.cov_module, 'Sigmoid')
        else:
            self.out_module = nn.Linear(pins['dim'], pouts['dim'])
            init_module(self.out_module, 'Linear')
            self.cov_module = nn.Sequential(nn.Linear(pouts['dim'], pouts['dim']), nn.Sigmoid())
            init_module(self.cov_module, 'Sigmoid')

    def forward(self, h, eps, sample=True):
        '''outputs the latent vector of the corresponding layer
        :param z: latent vector of the above layer
        :type z: torch.Tensor
        :param eps: latent stochastic variables
        :type z: torch.Tensor
        :returns:torch.Tensor'''
            
        if issubclass(type(h), list):
            h = torch.cat(tuple(h), 1)
        
        if issubclass(type(self.out_module), nn.ModuleList):
            params = []
            for i, module in enumerate(self.out_module):
                params.append(self.out_module[i](h), self.cov_module[i](eps))
        else:
            params = (self.out_module(h), self.cov_module(eps))
            
        return params
"""
