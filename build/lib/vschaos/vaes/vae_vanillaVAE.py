#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 18:38:11 2017

@author: chemla
"""

import pdb
import torch.nn as nn
import torch.optim

from ..data.data_transforms import Transform, OneHot
from ..modules.modules_hidden import  HiddenModule
from ..utils.misc import GPULogger, denest_dict, apply, apply_method, apply_distribution, print_stats, flatten_seq_method, checklist
from ..distributions import Flow, Multinomial, Categorical
from . import AbstractVAE 

logger = GPULogger(verbose=False)

class VanillaVAE(AbstractVAE):
    HiddenModuleClass = [HiddenModule, HiddenModule]
    # initialisation of the VAE
    def __init__(self, input_params, latent_params, hidden_params = {"dim":800, "nlayers":2}, hidden_modules=None, *args, **kwargs):
        self.set_hidden_modules(hidden_modules)
        super(VanillaVAE, self).__init__(input_params, latent_params, hidden_params, *args, **kwargs)

    @property
    def input_params(self):
        return self.pinput

    @property
    def hidden_params(self):
        return self.phidden

    @property
    def latent_params(self):
        return self.platent

    def make_encoders(self, input_params, latent_params, hidden_params, *args, **kwargs):
        encoders = nn.ModuleList()
        for layer in range(len(latent_params)):
            if layer==0:
                encoders.append(self.make_encoder(input_params, latent_params[0], hidden_params[0], name="vae_encoder_%d"%layer, *args, **kwargs))
            else:
                encoders.append(self.make_encoder(latent_params[layer-1], latent_params[layer], hidden_params[layer], name="vae_encoder_%d"%layer, *args, **kwargs))
        return encoders

    @classmethod
    def make_encoder(cls, input_params, latent_params, hidden_params, *args, **kwargs):               
        kwargs['name'] = kwargs.get('name', 'vae_encoder')
#        ModuleClass = hidden_params.get('class', DEFAULT_MODULE)
#        module = latent_params.get('shared_encoder') or ModuleClass(input_params, latent_params, hidden_params, *args, **kwargs)
        module_class = kwargs.get('module_class', cls.HiddenModuleClass[0])
        module = module_class(input_params, hidden_params, latent_params, *args, **kwargs)
        return module
    
    def make_decoders(self, input_params, latent_params, hidden_params, extra_inputs=[], *args, **kwargs):
        decoders = nn.ModuleList()
        for layer in range(len(latent_params)):
            if layer==0:
                new_decoder = self.make_decoder(input_params, latent_params[0], hidden_params[0],
                                                name="vae_decoder_%d"%layer, encoder = self.encoders[layer].hidden_modules, *args, **kwargs)
            else:
                new_decoder = self.make_decoder(latent_params[layer-1], latent_params[layer], hidden_params[layer],
                                                name="vae_decoder_%d"%layer, encoder=self.encoders[layer].hidden_modules,
                                                *args, **kwargs)
            decoders.append(new_decoder)
        return decoders
    
    @classmethod
    def make_decoder(cls, input_params, latent_params, hidden_params, *args, **kwargs):
        kwargs['name'] = kwargs.get('name', 'vae_decoder')
#        ModuleClass = hidden_params.get('class', DEFAULT_MODULE)
#        module = hidden_params.get('shared_decoder') or ModuleClass(latent_params, input_params, hidden_params, *args, **kwargs)
        module_class = kwargs.get('module_class', cls.HiddenModuleClass[1])
        module = module_class(latent_params, hidden_params, input_params, *args, **kwargs)
        return module

    def set_hidden_modules(self, hidden_modules):
        if hidden_modules is None:
            return
        if issubclass(type(hidden_modules), type):
            self.HiddenModuleClass = [hidden_modules, hidden_modules]
        elif issubclass(type(hidden_modules), list):
            self.HiddenModuleClass = [hidden_modules[0], hidden_modules[1]]

    # processing methods
    def sample_latent(self, z_params, latent_params, y):
        if latent_params.get('task') is None:
            try:
                out = apply_method(z_params, 'rsample')
            except NotImplementedError:
                out = apply_method(z_params, 'sample')
                pass
        else:
            if y.get(latent_params['task']) is None:
                try:
                    out = apply_method(z_params, 'rsample')
                except NotImplementedError:
                    out = apply_method(z_params, 'sample')
                    pass
            else:
                out = y.get(latent_params['task']).to(self.device)
                if latent_params['dist'] in (Multinomial, Categorical):
                    out = OneHot(latent_params['dim'])(out)
                    if out.ndim == z_params.probs.ndim - 1:
                        out = out.unsqueeze(-2).expand_as(z_params.probs)
                else:
                    raise NotImplementedError("semi-supervision for task %s not implemented"%(latent_params['dist']))
        return out

    def encode(self, x, y=None, sample=True, from_layer=0, *args, **kwargs):
        ins = x; outs = []
        for layer in range(from_layer, len(self.platent)):
            module_out = self.encoders[layer](ins, y=y, *args, **kwargs)
            out_params = module_out['out_params']
            if module_out.get('out') is None:
                if isinstance(self.platent[layer], list):
                    module_out['out'] = [self.sample_latent(out_params[i], self.latent_params[layer][i], y) for i in range(len(self.latent_params[layer]))]
                else:
                    module_out['out'] = self.sample_latent(out_params, self.latent_params[layer], y)
            outs.append(module_out)
            if issubclass(type(module_out), list):
                ins = torch.cat(module_out['out'], dim=-1)
            else:
                ins = module_out['out']
        return outs



    def decode(self, z, y=None, sample=True, from_layer=-1, *args, **kwargs):
        from_layer = -1 if from_layer is None else from_layer
        if from_layer < 0:
            from_layer += len(self.platent)
        ins = checklist(z)[from_layer]; outs = []
        if issubclass(type(ins), dict):
            ins = ins['out']
            assert ins is not None, "decoding from dictionaries need the 'out' attribute, absent from %d layer"%from_layer
        for i,l in enumerate(reversed(range(from_layer+1))):
            module_out = self.decoders[l](ins, y=y, *args, **kwargs)
            out_params = module_out['out_params']
            if module_out.get('out') is None:
                try:
                    ins = apply_method(out_params, 'rsample')
                except NotImplementedError:
                    ins = apply_method(out_params, 'sample')
                if issubclass(type(ins), tuple):
                    ins, z_preflow = ins
                    module_out['out_preflow'] = z_preflow
                module_out['out'] = ins
            outs.append(module_out)
            if issubclass(type(module_out), list):
                ins = torch.cat(ins, dim=-1)
        outs = list(reversed(outs))
        return outs



    def format_output(self, enc_out, dec_out):
        x_params = dec_out[0]['out_params']
        dec_out = denest_dict(dec_out[1:]) if len(dec_out) > 1 else {}
        enc_out = denest_dict(enc_out)
        logger("output formatted")
        out = {'x_params': x_params,
                'z_params_dec': dec_out.get('out_params'), 'z_dec': dec_out.get('out'),
                'z_params_enc': enc_out['out_params'], 'z_enc': enc_out['out'],
                "z_preflow_enc": enc_out.get('out_preflow'), "z_preflow_dec": dec_out.get('out_preflow')}
        for k in dec_out.keys():
            if not k in ['out_params', 'out', 'out_preflow']:
                out[k+"_dec"] = dec_out[k]
        for k in enc_out.keys():
            if not k in ['out_params', 'out', 'out_preflow']:
                out[k+"_enc"] = enc_out[k]

        dull_keys = []
        for k, v in out.items():
            if v is None:
                dull_keys.append(k)
        for k in dull_keys:
            del out[k]

        return out

    def forward(self, x, y=None, options={}, from_layer=-1, multi_decode=False,  *args, **kwargs):
        x = self.format_input_data(x, requires_grad=False)
        # logger("data formatted")
        enc_out = self.encode(x, y=y, *args)
        logger("data encoded")
        dec_out = self.decode(enc_out, y=y, *args, from_layer=from_layer)
        logger("data decoded")
        return self.format_output(enc_out, dec_out)

    def get_flows_modules(self):
        flows = [[]]*len(self.platent)
        for i, layer in enumerate(self.platent):
            layer = checklist(layer)
            for pl in layer: 
                if issubclass(type(pl['dist']), Flow):
                    flows[i].append(pl['dist'].flow) 
        return flows

    # define optimizer
    def init_optimizer(self, optim_params={}, init_scheduler=True):
        self.optim_params = optim_params
        alg = optim_params.get('optimizer', 'Adam')
        optim_args = optim_params.get('optim_args', {'lr':1e-3})
        optimization_mode = optim_params.get('mode', 'full')
        if issubclass(type(optim_args['lr']), list):
            if len(optim_args['lr']) != len(self.platent):
                optim_args['lr'] = optim_args['lr'][0]
            else:
                optim_args = [{**optim_args, 'lr':optim_args['lr'][i]} for i in range(len(self.platent))]

        optim_args = checklist(optim_args, n=len(self.platent))
        param_groups = []
        for l in range(len(self.platent)):
            print('layer %d : %s'%(l, optim_args[l]))
            params = []
            if optimization_mode in ['full', 'encoder']:
                params.extend(list(self.encoders[l].parameters()))
            if optimization_mode in ['full', 'decoder']:
                params.extend(list(self.decoders[l].parameters()))
            #     param_groups.append({'params':self.decoders[l].parameters(), **optim_args[l]})
            param_groups.append({'params': params, **optim_args[l]})
        optimizer = getattr(torch.optim, alg)(param_groups)

        self.optimizers = {'default':optimizer}
        if init_scheduler:
            self.init_scheduler(optim_params)

    def init_scheduler(self, optim_params):
        self.schedulers = {}
        scheduler = optim_params.get('scheduler', 'ReduceLROnPlateau')
        if scheduler is not None:
            scheduler_args = optim_params.get('scheduler_args', {'patience':100, "factor":0.2, 'eps':1e-10})
            self.schedulers['default'] = getattr(torch.optim.lr_scheduler, scheduler)(self.optimizers['default'])
        
        
    def optimize(self, loss, options={}, retain_graph=False, *args, **kwargs):
        # optimize
        self.optimizers['default'].step()

    def schedule(self, loss, options={}):
        if self.schedulers.get('default') is not None:
            self.schedulers['default'].step(loss)

    def get_scripted(vae, attach_transform=None):
        class VanillaVAEScriptable(nn.Module):
            def __init__(self, vae, transform):
                nn.Module.__init__(self)
                self.encoders = vae.encoders
                self.decoders = vae.decoders
                self.transform = transform

            def encode(self, x, y, noise_amount):
                with torch.no_grad():
                    ins = self.transform(x);
                    for encoder in self.encoders:
                        ins = encoder(ins, y)['out_params']
                        ins = ins.mean + ins.stddev * noise_amount * torch.randn_like(ins.mean)
                return ins

            def decode(self, x, y, noise_amount):
                with torch.no_grad():
                    ins = x;
                    for decoder in reversed(self.decoders):
                        ins = decoder(ins, y)['out_params']
                        ins = ins.mean + ins.stddev * noise_amount * torch.randn_like(ins.mean)
                return self.transform.invert(ins)

            def forward(self, x, y, latent_noise, data_noise):
                #return self.decode(self.encode(x, y, latent_noise), y, data_noise)
                return x

        vae.eval()
        with torch.no_grad():
            batch_size = (1,)
            # input_size = (1,)
            # if vae.pinput.get('channels'):
            #     input_size += (vae.pinput['channels'],)
            input_dict = {'encode': (torch.zeros(*batch_size, vae.pinput['dim']), torch.zeros(1), torch.tensor(0.5)),
                          'decode': (torch.zeros(*batch_size, vae.platent[-1]['dim']), torch.zeros(0), torch.tensor(0.5)),
                          'forward': (torch.zeros(*batch_size, vae.pinput['dim']), torch.zeros(0),
                                      torch.tensor(0.5), torch.tensor(0.5))}
            if attach_transform is not None:
                transform = attach_transform
            else:
                transform = Transform()
            scriptable_vae = VanillaVAEScriptable(vae, transform)

            # test first in python
            out = scriptable_vae.encode(*input_dict['encode'])
            out = scriptable_vae.decode(*input_dict['decode'])
            out = scriptable_vae.forward(*input_dict['forward'])

            return torch.jit.trace_module(scriptable_vae, input_dict)

# define losses


