#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 18:12:05 2018

@author: chemla
"""
import torch
import torch.nn as nn

from .modules_bottleneck import MLP
from .modules_distribution import get_module_from_density
from ..distributions.distribution_priors import IsotropicGaussian
from ..utils import checklist
from ..utils.misc import recgetattr



# HiddenModule is an abstraction for inter-layer modules

# it has the following components:
#    hidden_modules : creates the hidden modules specified by phidden (can be a list)
#    out_modules : creates the modules corresponding to the output distributions specified by platent


class HiddenModule(nn.Module):
    default_module = MLP
    take_sequences = False
    dump_patches = True

    def __init__(self, pins, phidden=None, pouts=None, linked=None, *args, **kwargs):
        super(HiddenModule, self).__init__()
        self.pins = pins; self.phidden = phidden; self.pouts = pouts
        self.linked = linked or checklist(self.phidden)[0].get('linked', False)

        # specification check
        if issubclass(type(self.pins), list):
            if issubclass(type(self.pouts), list):
                if not self.linked:
                    assert len(self.pouts) == len(self.pins), f"number of inputs ({len(self.pins)}) /" \
                                                                f"outputs ({len(self.pouts)}) do not match"
                    self.phidden = checklist(self.phidden, n=len(self.pouts))
                    assert len(self.phidden) == len(self.pins), f"number of hidden ({len(self.phidden)}) /" \
                                                              f"outputs ({len(self.pouts)}) do not match"
                else:
                   assert not issubclass(type(self.phidden), list), "please give one specification if different"\
                                                                        "number of outputs / inputs are used"
            else:
                if not self.linked:
                    self.phidden = checklist(self.phidden, n=len(self.pins))
                    assert len(self.phidden) == len(self.pins), "if linked is None, number of hidden specifications must be " \
                                                           "one or match the number of inputs"
        else:
            if issubclass(type(self.pouts), list):
                if not self.linked:
                    self.phidden = checklist(phidden, len(self.pouts))
                    assert len(self.phidden) == len(self.pouts), f"number of hidden ({len(self.phidden)}) /" \
                                                                f"outputs ({len(self.pouts)}) do not match"

        # if self.phidden:
        self._hidden_modules = self.make_hidden_layers(self.pins, self.phidden, pouts=self.pouts, linked=self.linked, *args, **kwargs)
        # else:
        #     self._hidden_modules = Identity()

        # get output layers
        # self.out_modules = None
        # if pouts:
        #     pouts_input = phidden if phidden else pins
        if pouts: 
            self.out_modules = self.make_output_layers(self.phidden, pouts, *args, linked=self.linked, **kwargs)

    def make_hidden_layers(self, pins, phidden={"dim":800, "nlayers":2, 'label':None, 'conditioning':'concat'}, linked=False, pouts=None, encoder=None, *args, **kwargs):
        if issubclass(type(pins), list):
            if issubclass(type(pouts), list):
                if linked:
                    module_class = phidden.get('class', self.default_module)
                    return module_class(pins, phidden, pouts=pouts, encoder=encoder, *args, **kwargs)
                else:
                    encoder = encoder or [None]*len(pouts)
                    return nn.ModuleList(
                        [self.make_hidden_layers(pins[i], phidden[i], *args, pouts=pouts[i], encoder=encoder[i], **kwargs) for i in
                         range(len(phidden))])
            else:
                if linked:
                    module_class = phidden.get('class', self.default_module)
                    return module_class(pins, phidden, pouts=pouts, *args,encoder=encoder,  **kwargs)
                else:
                    encoder = encoder or [None] * len(pouts)
                    return nn.ModuleList(
                        [self.make_hidden_layers(pins[i], phidden[i], *args, pouts=pouts, encoder=encoder[i], **kwargs) for i in
                         range(len(phidden))])
        else:
            if issubclass(type(pouts), list):
                if linked:
                    module_class = phidden.get('class', self.default_module)
                    return module_class(pins, phidden, pouts=pouts, *args, **kwargs)
                else:
                    encoder = checklist(encoder, n=len(phidden)) or [None] * len(pouts)
                    return nn.ModuleList(
                        [self.make_hidden_layers(pins, phidden[i], *args, pouts=pouts[i], encoder=encoder[i], **kwargs) for i in
                         range(len(phidden))])
            else:
                module_class = phidden.get('class', self.default_module)
                return module_class(pins, phidden, pouts=pouts, encoder=encoder, **kwargs)

        # if issubclass(type(phidden), list):
        #     if not linked:
        #         if issubclass(type(pins), list):
        #             pouts_tmp = checklist(pouts, n=len(pins)) if pouts is None else pouts
        #             return nn.ModuleList([self.make_hidden_layers(pins[i], dict(ph), *args, pouts=pouts_tmp[i], **kwargs) for i, ph in enumerate(phidden)])
        #         else:
        #             pouts_tmp = checklist(pouts, n=len(pins)) if pouts is None else pouts
        #             return nn.ModuleList([self.make_hidden_layers(pins, dict(ph), pouts=pouts_tmp[i], *args, **kwargs) for ph in phidden])
        #     else:
        #         return nn.ModuleList([self.make_hidden_layers(pins, dict(ph), *args, pouts=pouts, **kwargs) for ph in phidden])
        # module_class = phidden.get('class', self.default_module)
        # hidden_modules = module_class(pins, phidden, pouts=pouts, *args, **kwargs)

    @property
    def hidden_modules(self):
        return self._hidden_modules

    @property
    def hidden_out_params(self, hidden_modules = None):
        hidden_modules = hidden_modules or self._hidden_modules
        if issubclass(type(hidden_modules), nn.ModuleList):
            params = []
            for i, m in enumerate(hidden_modules):
                if hasattr(hidden_modules[i], 'phidden'):
                    params.append(hidden_modules[i].phidden)
                else:
                    params.append(checklist(self.phidden, n=len(hidden_modules))[i])
            return params
        else:
            if hasattr(hidden_modules, 'phidden'):
                return checklist(hidden_modules.phidden)[-1]
            else:
                return checklist(checklist(self.phidden)[0])[-1]

    def make_output_layers(self, pins, pouts, *args, linked=True, **kwargs):
        '''returns output layers with resepct to the output distribution
        :param in_dim: dimension of input
        :type in_dim: int
        :param pouts: properties of outputs
        :type pouts: dict or [dict]
        :returns: ModuleList'''
        #current_hidden_params = checklist(self.hidden_out_params, n=len(pouts))

        if issubclass(type(pouts), list):
            modules = [None]*len(pouts)
            for i, pout in enumerate(pouts):
                if (issubclass(type(pins), list) and not linked) or (not linked):
                    modules[i] = self.get_module_from_density(pout["dist"])\
                        (pins[i], pout, hidden_module=self.hidden_modules[i], **kwargs)
                else:
                    modules[i] = self.get_module_from_density(pout["dist"]) \
                        (pins, pout, hidden_module=self.hidden_modules, **kwargs)
            return nn.ModuleList(modules)
        else:
            return nn.ModuleList([self.get_module_from_density(pouts["dist"]) \
                    (pins, pouts, hidden_module=self.hidden_modules, **kwargs)])



    def get_module_from_density(self, dist):
        return get_module_from_density(dist)

    def forward_hidden(self, x, y=None, *args, **kwargs):
        
        if issubclass(type(self.hidden_modules), nn.ModuleList):
            x = checklist(x, n=len(self.hidden_modules))
            hidden_out = []
            for i, h in enumerate(self.hidden_modules):
                if self.phidden[i].get('label_params'):
                    y_tmp = {k: y[k] for k in self.phidden[i]['label_params']}
                else:
                    y_tmp = None
                hidden_out.append(h(x[i], y=y_tmp, sample=True, *args, **kwargs))
        else:
            if self.phidden.get('label_params'):
                y_tmp = {k: y[k] for k in self.phidden['label_params']}
            else:
                y_tmp = None
            hidden_out = self.hidden_modules(x, y=y_tmp, *args, **kwargs)
        
        if self.linked and issubclass(type(self.hidden_modules), list): 
            hidden_out = torch.cat(tuple(hidden_out), -1)
        else:
            hidden_out = hidden_out
        return hidden_out
    
    def forward_params(self, hidden, y=None, *args, **kwargs):
        # get distirbutions from distribution module
        z_dists = []
        for i, out_module in enumerate(self.out_modules):
            if issubclass(type(self.hidden_modules), nn.ModuleList):
                indices = None
                if out_module.requires_deconv_indices:
                    indices = self.hidden_modules[i].get_pooling_indices()[0]
                if self.linked: 
                    z_dists.append(out_module(hidden, indices=indices))
                else:
                    if len(self.out_modules) != 1:
                        assert len(hidden) == len(self.out_modules)
                        z_dists.append(out_module(hidden[i], indices=indices))
                    else:
                        h = torch.cat(hidden, dim=-1)
                        z_dists.append(out_module(h, indices=indices))
            else: 
                requires_deconv_indices = recgetattr(out_module, 'requires_deconv_indices')
                indices = None
                if requires_deconv_indices:
                    indices = self.hidden_modules.get_pooling_indices()

                if issubclass(type(hidden), list):
                    z_dists.append(out_module(hidden[i], indices=checklist(indices)[i]))
                else:
                    z_dists.append(out_module(hidden, indices=indices))

        if not issubclass(type(self.pouts), list):
            z_dists = z_dists[0]
        return z_dists
        

    def forward(self, x, y=None, sample=True, return_hidden=False, *args, **kwargs):
        # get hidden representations
        out = {}
        hidden_out = self.forward_hidden(x, y=y, *args, **kwargs)
        if return_hidden:
            out['hidden'] = hidden_out
        if self.out_modules is not None:
            # get output distributions
            out['out_params'] = self.forward_params(hidden_out, y=y, *args, **kwargs)

        return out


class AveragingHiddenModule(HiddenModule):
    def forward_hidden(self, x, y=None, *args, **kwargs):
        hidden = super(AveragingHiddenModule, self).forward_hidden(x, y=y, *args, **kwargs)
        return hidden.mean(1)


"""(Deprecated)
class DLGMModule(HiddenModule):
    def get_module_from_density(self, dist):
        return DLGMLayer
    
    def sample(self, h, eps, *args, **kwargs):
        z = []
        for i, out_module in enumerate(self.out_modules):
            if issubclass(type(self.hidden_modules), nn.ModuleList):
                if self.linked: 
                    z.append(out_module(h, eps))
                else:
                    z.append(out_module(h[i], eps[i]))
            else: 
                z.append(out_module(h, eps))
        # sum
        z = [current_z[0] + current_z[1] for current_z in z]
        if not issubclass(type(self.pouts), list):
            z = z[0]
        return z
    
    def forward_params(self, h, *args, **kwargs):
        batch_size = h[0].shape[0] if issubclass(type(h), list) else h.shape[0]
        if issubclass(type(self.pouts), list):
            target_shapes = [(batch_size, self.pouts[i]['dim']) for i in range(len(self.pouts))]
            params = [IsotropicGaussian(*target_shapes[i], device=h.device) for i in range(len(out['out_params']))]
        else:
            params = IsotropicGaussian(batch_size, self.pouts['dim'], device=h.device)
        return params

    def forward(self, x, y=None, sample=True, *args, **kwargs):
        # get hidden representations
        out = {'hidden': self.forward_hidden(x, y=y, *args, **kwargs)}
        if self.out_modules is not None:
            # get output distributions
            out['out_params'] = self.forward_params(out['hidden'], y=y, *args, **kwargs)
            # get samples
            if sample:
                out['out'] = self.sample(out['hidden'], y=y, *args, **kwargs)
        return out
        
"""
