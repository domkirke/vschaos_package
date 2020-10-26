#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 18:26:20 2018

@author: chemla
"""
import pdb, dill
import numpy as np
import torch
import torch.nn as nn
from ..distributions import Categorical
from collections import OrderedDict
from . import utils
from ..monitor import visualize_dimred as dr

    #############################################
    ###  Helper functions

def disentangle_hidden_params(hidden_params):
    def dis_tool(single_param):
        enc = single_param.get('encoder', single_param); dec = single_param.get('decoder', single_param)
        if issubclass(type(enc), dict):
            enc = dict(enc)
        elif issubclass(type(enc), list):
            enc = [dict(k) for k in enc]
        if issubclass(type(dec), dict):
            dec = dict(dec)
        elif issubclass(type(dec), list):
            dec = [dict(k) for k in dec]
        if 'decoder' in enc:
            del enc['decoder']
        if 'encoder' in dec:
            del dec['decoder']
        return enc, dec
    enc_hidden_params = [None]*len(hidden_params)
    dec_hidden_params = [None]*len(hidden_params)
    for i, hidden_layer in enumerate(hidden_params):
        if hidden_layer is None:
            continue
        if issubclass(type(hidden_layer), list):
            current_params = [dis_tool(x) for x in hidden_layer]
            enc_hidden_params[i], dec_hidden_params[i] =  [c[0] for c in current_params],  [c[1] for c in current_params]
        else:
            enc_hidden_params[i], dec_hidden_params[i] = dis_tool(hidden_layer)
    return enc_hidden_params, dec_hidden_params


class AbstractVAE(nn.Module):
    dump_patches = True
    take_sequences = False
    #############################################
    ###  Architecture methods
    
    def __len__(self):
        return len(self.platent)
    
    def __init__(self, input_params, latent_params, hidden_params=[{"dim":800, "layers":2}], device=-1, *args, **kwargs):
        nn.Module.__init__(self)
        
        # retaining constructor for save & load routines (stupid with dump_patches?)
        if not hasattr(self, 'constructor'):
            self.constructor = {'input_params':input_params, 'latent_params':latent_params, 'hidden_params':hidden_params, 'args':args, 'kwargs':kwargs} # remind construction arguments for easy load/save

        # turn singleton specifications into lists
        if not issubclass(type(latent_params), list):
            latent_params = [latent_params]
        if not issubclass(type(hidden_params), list):
            hidden_params = [hidden_params]
        self.pinput = input_params; self.phidden = hidden_params; self.platent = latent_params

        # check that hidden layers' specifications are well-numbered
        if len(hidden_params) < len(latent_params):
            print("[Warning] hidden layers specifcations is under-complete. Copying last configurations for missing layers")
            last_layer = hidden_params[-1]
            while len(hidden_params) < len(latent_params):
                hidden_params.append(last_layer)

        # init modules
        self.init_modules(self.pinput, self.platent, self.phidden, *args, **kwargs)

        # GPU handling
        self.device = 'cpu' if device < 0 else 'cuda:%d'%device
        self.is_cuda = self.device != 'cpu'
        if self.is_cuda:
            self.cuda(torch.device(self.device).index)

        self.optimizers = None
        self.schedulers = None
        self.manifolds = {None:dr.LatentManifold(None, self, None)}
        self.sample_norm = kwargs.get('sample_norm', False)


    # dynamical attributes
    @property
    def input_params(self):
        return self.pinput
    @property
    def latent_params(self):
        return self.platent
    @property
    def hidden_params(self):
        return self.phidden


    # architecture methods
    def init_modules(self, input_params, latent_params, hidden_params, *args, **kwargs):
        # Disentangle encoders and decoders specifications in case
        hidden_enc_params, hidden_dec_params = disentangle_hidden_params(hidden_params)
        self.encoders = self.make_encoders(input_params, latent_params, hidden_enc_params, *args, **kwargs)
        self.decoders = self.make_decoders(input_params, latent_params, hidden_dec_params, *args, **kwargs)

    def make_encoders(self, input_params, latent_params, hidden_params, *args, **kwargs):
        return nn.ModuleList()

    def make_decoders(self, input_params, latent_params, hidden_params, *args, **kwargs):
        return nn.ModuleList()


    #############################################
    ###  Processing methods

    def encode(self, x, options={}, *args, **kwargs):
        return None

    def decode(self, z, options={}, *args, **kwargs):
        return None

    def forward(self, x, y=None, options={}, *args, **kwargs):
        return {}


    #############################################
    ###  Loss methods & optimization schemes

    def get_loss(self, *args):
        return 0.

    def init_optimizer(self, optim_params={}):
#        self.optimizers = {'default':getattr(torch.optim, optimMethod)(self.parameters(), **optimArgs)}
        self.optim_params = optim_params
        self.optimizers = {} # optimizers is here a dictionary, in case of multi-step optimization
        self.schedulers = {}

    def update_optimizers(self, options, retain_graph=False):
        if not retain_graph:
            # optimizer update at each 'step' call
            for _, o in self.optimizers.items():
                o.zero_grad() # zero gradients

    # this is a global step function, where the optimize function should be overloaded
    def step(self, loss, options={'epoch':0}, retain_graph=False, *args, **kwargs):
        # update optimizers in case
        # optimize
        self.optimize(loss,retain_graph=retain_graph)
        self.update_optimizers(options, retain_graph=retain_graph)

    def optimize(self, loss, *args, **kwargs):
        pass

    def schedule(self, *args, **kwargs):
        pass


    #############################################
    ###  Loss methods & optimization schemes

    def get_device(self):
        return next(self.parameters()).device

    def cuda(self, device=None):
        if device is None:
            device = torch.cuda.current_device()
        self.device = device
        super(AbstractVAE, self).cuda(device)
        self.is_cuda = True

    def cpu(self):
        self.device = torch.device('cpu')
        super(AbstractVAE, self).cpu()
        self.is_cuda = False


    #############################################
    ###  Load / save methods

    def get_dict(self, *args, **kwargs):
        if self.is_cuda:
            state_dict = OrderedDict(self.state_dict())
            for i, k in state_dict.items():
                state_dict[i] = k.cpu()
        else:
            state_dict = self.state_dict()
        constructor = dict(self.constructor)
        vae_class = self.__class__
        # vae_class.__module__ = '__main__' # trick to pickle dynamically created classes
        save = {'state_dict':state_dict, 'init_args':constructor, 'class':self.__class__, 'manifolds':self.manifolds}
        for k,v in kwargs.items():
            save[k] = v
        return save

    def save(self, filename, *args, **kwargs):
        save = self.get_dict(*args, **kwargs)
        torch.save(save, filename, pickle_module=dill)

    @classmethod
    def load(cls, pickle, with_optimizer=False):
        init_args = pickle['init_args']
        for k,v in init_args['kwargs'].items():
            init_args[k] = v
        del init_args['kwargs']
        #cls.__module__ = "__main__"
        vae = cls(**pickle['init_args'])
        vae.load_state_dict(pickle['state_dict'])
        if with_optimizer:
            vae.optimizers = pickle.get('optimizers', {})
            vae.rs = pickle.get('optimizers', {})

        return vae



    #############################################
    ###  Utility methods

    def format_input_data(self, x, requires_grad=True, pinput=None, onehot=True, sample_norm=False, *args, **kwargs):
        pinput = pinput or self.pinput
        current_device = next(self.parameters()).device
        if x is None:
            return
        if issubclass(type(pinput), list):
            if not issubclass(type(x), list):
                x = [x]
            x_formatted = [None]*len(pinput)
            for i in range(len(x)):
                x_formatted[i] = self.format_input_data(x[i], requires_grad=requires_grad, pinput=pinput[i], onehot=onehot, *args, **kwargs)
        else:
            if isinstance(x, np.ndarray):
                if self.take_sequences and len(x.shape) == 2:
                    x = np.expand_dims(x, 1)
                x_formatted = x.copy()
                if x.dtype!='float32':
                    x_formatted = x_formatted.astype('float32')
                x_formatted = torch.from_numpy(x_formatted)
            else:
                x_formatted = x
            x_formatted = x_formatted.to(current_device, dtype=torch.float32)
        sample_norm = sample_norm or self.sample_norm
        if sample_norm:
            x_formatted = utils.sample_normalize(x_formatted)
        return x_formatted


    def add_manifold(self, name, dataset, method, *args, **kwargs):
       self.manifolds[name] = dr.LatentManifold(method, self, dataset, **kwargs)
       # if n_points is not None:
       #     dataset = dataset.random_subset(n_points)
       # x, y = dataset[:]
       # out = self.encode(x, y=y, *args, **kwargs)
       # params = out[layer]['out_params']
       # if isinstance(params, list):
       #      params = params[0]
       # self.manifolds[name]['projection'] = method(n_components=ndims)
       # self.manifolds[name]['projection'].fit(params)
       # self.manifolds[name]['points'] = params
       # self.manifolds[name]['anchors'] = []
#
