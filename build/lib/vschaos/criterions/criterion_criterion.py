#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 20:26:17 2018

@author: chemla
"""
import torch
from time import process_time
import torch.nn as nn
import copy, pdb
import numpy as np
from . import utils
from ..utils import NaNError

def reduce(data, reduction):
    if type(data) in (int, float):
        return 0
    if reduction == "mean" or len(data.shape)==1:
        # global average
        return torch.mean(data)
    elif reduction == "seq":
        sum_dims = tuple(range(len(data.shape))[2:])
        return torch.sum(data, dim=sum_dims).mean()
    elif reduction == "batch_mean":
        mean_sums = tuple(list(range(len(data.shape)))[1:])
        return torch.mean(data, dim=mean_sums)
    elif reduction == "batch_sum":
        mean_sums = tuple(list(range(len(data.shape)))[1:])
        return torch.sum(data, dim=mean_sums)
    else:
        # just average on batches (mathematically better, no?)
#        return torch.sum(data)
        sum_dims = tuple(range(len(data.shape))[1:])
        return torch.sum(data, dim=sum_dims).mean()

class Criterion(nn.modules.loss._Loss):
    dump_patches = True
    def __init__(self, size_average=None, reduce=None, reduction=None, weight=1.0, pow=1.0, name=None, **kwargs):
        super(Criterion, self).__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        self._weight = weight
        self._pow =  pow
        self._name = name
        self.reset()
        self.unsupervised()

    def loss(self, *args, **kwargs):
        return 0.
    
    def step(self, *args, **kwargs):
        return

    def supervised(self):
        self._supervised = True
        return

    def unsupervised(self):
        self._supervised = False
        return

    def __repr__(self):
        return "Criterion(reduction=%s)"%self.reduction

    @property
    def loss_history(self):
        return self._loss_history

    def reset(self):
        self._loss_history = {}

    def get_named_losses(self, losses):
        name = 'dummy' if self._name is None else '%s_dummy'%self._name
        return {name : losses}

    def write(self, name, losses, time=None):
        if time is None:
            time = process_time()
        losses = self.get_named_losses(losses)
        if not name in self._loss_history.keys():
            self.loss_history[name] = {}
        for loss_name, value in losses.items():
            value = np.array([utils.decudify(value, scalar=True)])
            if not loss_name in list(self._loss_history[name].keys()):
                self._loss_history[name][loss_name] = {'values':value, 'time':[time]} 
            else:
                self._loss_history[name][loss_name]['values'] = np.concatenate([self._loss_history[name][loss_name]['values'], value])
                self._loss_history[name][loss_name]['time'].append(time)

    def __call__(self, *args, **kwargs):
        loss, losses = self.loss(*args, **kwargs)
        loss *= self._weight
        return loss, losses

        
    def __add__(self, c):
        if issubclass(type(self), CriterionContainer):
            nc = copy.deepcopy(self)
            nc._criterions.append(c)
        if issubclass(type(self), Criterion):
            # c = copy.deepcopy(c)
            nc = CriterionContainer([self, c])
        return nc
    
    def __radd__(self, c):
        return self.__add__(c)
        
    def __sub__(self, c):
        if issubclass(type(self), CriterionContainer):
            nc = copy.deepcopy(self)
            c = copy.deepcopy(c)
            c._weight = -1.0
            nc._criterions.append(c)
        if issubclass(type(self), Criterion):
            nc = copy.deepcopy(self)
            c = copy.deepcopy(c)
            c._weight = -1.0
            nc = CriterionContainer([nc, c])
        return nc
    
    def __rsub__(self, c):
        return self.__sub__(c)
        
    def __mul__(self, f):
        assert issubclass(type(f), (int, float)) or issubclass(type(f), np.ndarray)
        new = copy.deepcopy(self)
        new._weight *= f
        return new
        
    def __rmul__(self, c):
        return self.__mul__(c)
 
    def __truediv__(self, f):
        assert issubclass(type(f), (float, int))
        new = copy.deepcopy(self)
        new._weight /= f
        return new

    def __rtruediv__(self, f):
        assert issubclass(type(f), (float, int))
        new = copy.deepcopy(self)
        new._weight *= f
        new._pow *= -1
        return new

    def __pow__(self, f):
        assert issubclass(type(f), (float, int))
        new = copy.deepcopy(self)
        new._pow = f
        return new

    def reduce(self, data, reduction=None):
        reduction = reduction or self.reduction
        if type(data) in (int, float):
            return 0
        if reduction == "mean" or len(data.shape)==1:
            # global average
            return torch.mean(data)
        elif reduction == "seq":
            sum_dims = tuple(range(len(data.shape))[2:])
            return torch.sum(data, dim=sum_dims).mean()
        else:
            # just average on batches (mathematically better, no?)
    #        return torch.sum(data)
            sum_dims = tuple(range(len(data.shape))[1:])
            return torch.sum(data, dim=sum_dims).mean()
            
        # def __rdiv__(self, c):
        #     raise NotImplementedError
        #     return self.__div__(c)
            

def recursive_method(obj, method, *args, **kwargs):
    if issubclass(type(obj), (tuple, list)):
        [recursive_method(o, method, *args, **kwargs) for o in obj]
    else:
        func = getattr(obj, method)
        func(*args, **kwargs)


class CriterionContainer(Criterion):
    def __init__(self,  criterions=[], options={}, weight=1.0, **kwargs):
        self._criterions = criterions
        super().__init__(**kwargs)

    def __add__(self, other):
        if not isinstance(other, Criterion):
            raise ValueError('Criterions can only be added to other criterions')
        if isinstance(other, CriterionContainer):
            return super(CriterionContainer, self).__add__(other)
        else:
            return CriterionContainer(criterions=self._criterions + [other])

    def __len__(self):
        return len(self._criterions)

    def __getitem__(self, item):
        return self._criterions[item]
    
    def loss(self, *args, **kwargs):
        loss = 0.; losses = list()
        for c in self._criterions:
            l, ls = c.loss(*args, **kwargs)
            if torch.isnan(l):
                raise NaNError()
            loss = loss + c._weight * l**c._pow
            losses.append(ls)
        losses = tuple(losses)
        return loss, losses

    def __repr__(self):
        return "CriterionContainer(%s)"%[s.__repr__() for s in self._criterions]

    def reset(self):
        recursive_method(self._criterions, "reset")

    @property
    def loss_history(self):
        histories = [c.loss_history for c in self._criterions]
        partitions = set(sum([list(h.keys()) for h in histories], []))
        full_history = {k:{} for k in partitions}
        for k in partitions:
            for h in histories:
                full_history[k] = {**full_history[k], **h[k]}
        return full_history

    def write(self, name, losses, **kwargs):
        for i,c in enumerate(self._criterions):
            c.write(name, losses[i], **kwargs)

    def step(self, *args, **kwargs):
        recursive_method(self._criterions, "step")
    
    def get_named_losses(self, losses):
        named_losses=dict()
        for i, l in enumerate(self._criterions):
            current_loss = l.get_named_losses(losses[i])
            named_losses = {**named_losses, **current_loss}
        return named_losses

    def cuda(self, *args, **kwargs):
        for c in self._criterions:
            c.cuda(*args, **kwargs)

            
            

