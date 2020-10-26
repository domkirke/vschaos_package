#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:46:21 2018

@author: chemla
"""
import pdb
from numbers import Number
import torch, numpy as np
from torch.autograd import Variable

def oneHot(labels, dim):
    if isinstance(labels, Number):
        t = torch.zeros((1, dim))
        t[0, int(labels)] = 1
    else:
        original_shape = labels.shape
        labels_tmp = labels.reshape(-1).long()
        onehot_tmp = torch.zeros(labels_tmp.shape[0], dim)
        for i, l in enumerate(labels_tmp):
            onehot_tmp[i, l] = 1
        t = onehot_tmp.view(*original_shape, -1)
    return t

def fromOneHot(vector, is_sequence=False):
    axis = 2 if is_sequence else 1
    if issubclass(type(vector), np.ndarray):
        ids = np.argmax(vector, axis=axis)
        return ids
    elif issubclass(type(vector), torch.Tensor):
        return torch.argmax(vector, dim=axis)
    else:
        raise TypeError('vector must be whether a np.ndarray or a tensor.')
