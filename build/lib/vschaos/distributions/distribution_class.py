#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 23:27:11 2018

@author: chemla
"""
from . import Distribution, Normal
from numpy import ones, ndarray
from torch import from_numpy, Tensor, index_select, LongTensor, cat, zeros, ones
from .. import utils
import random
import pdb



# left to do : override all Distribution methods with an additional class argument

class ClassPrior(Distribution):
    
    def __repr__(self):
        string = ""
        for i in range(self.n_classes):
            string += "%s\n"%(self.get_distrib(i))
        
        
    def __init__(self, params, dist=Normal, validate_args=None):
        self._dist = dist
        # import parameters
        self._params = []
        for i in range(len(params)):
            p = params[i]
            if issubclass(type(p), ndarray):
                p = from_numpy(p)
            p.requires_grad_(False)
            self._params.append(p)
            
        self._params = tuple(self._params) # Warning! Here params are Gaussian Parameters for each class
        self.n_classes = len(self._params)
        super(ClassPrior, self).__init__(event_shape = params[0].size(0), validate_args=validate_args)        
        
    def get_distrib(self, y):
        return self.dist(self.get_params(y))
        
    def get_params(self, y=[], cuda=False, *args, **kwargs):
        params = []
        y = utils.onehot.fromOneHot(y)
        if cuda:
            y = y.cuda()
        for i in range(len(self.params)):
            if cuda:
                param = self.params[i].cuda()
            else:
                param = self.params[i]
            p = index_select(param, 0, y) 
            params.append(p)
        return tuple(params)
        
    def __call__(self, y=[], cuda=False, *args, **kwargs):
        with_undeterminate = kwargs.get('with_undeterminate', False)
        if with_undeterminate:
            undeterminate_id = kwargs.get('undeterminate_id', -1)
            y = self.remove_undeterminate(y, undeterminate_id)
        if y.ndim > 1:
            y = utils.onehot.fromOneHot(y, )
        z = zeros((y.size(0), self.dim), requires_grad=True, device=y.device)
        for i in range(y.size(0)):
            param = []
            for p in range(len(self.params)):
                p = self.params[p][y[i]]
                param.append(p)
            param = tuple(param)
            z[i, :] = self.dist(*param)
        return z
