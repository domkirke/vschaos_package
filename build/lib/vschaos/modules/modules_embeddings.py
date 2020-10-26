#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:49:58 2017

@author: chemla

"""

import torch, torch.nn as nn
from . import MLP
from ..utils import checklist, flatten_seq_method
from ..data.data_transforms import OneHot


class OneHotEmbedding(nn.Module):
    def __repr__(self):
        return "OneHotEmbedding(n_classes=%s)"%(self.n_classes)

    def __init__(self, input_params, out_params, **kwargs):
        super(OneHotEmbedding, self).__init__()
        self.onehot = OneHot(classes=out_params['dim'])
        self.n_classes = out_params['dim']

    @flatten_seq_method
    def forward(self, x, **kwargs):
        return self.onehot(x)

    @property
    def out_dim(self):
        return self.n_classes

class SoftmaxEmbedding(nn.Module):
    #TODO make stochastic embeddings?
    def __init__(self, input_params, out_params, hidden_params={'dim':200, 'nlayers':2, 'normalization':'batch', 'log':False}):
        super(SoftmaxEmbedding, self).__init__()
        hidden_params['dim'] = checklist(hidden_params['dim'], n=hidden_params['nlayers']) + [out_params['dim']]
        self.embedding = MLP(input_params, hidden_params)
        self.log = hidden_params.get('log', False)
        if self.log:
            self.softmax_layer = nn.LogSoftmax(dim=-1)
        else:
            self.softmax_layer = nn.Softmax(dim=-1)

    @flatten_seq_method
    def forward(self, x):
        return self.softmax_layer(self.embedding(x))

class Time2Vec(nn.Module):
    def __init__(self, input_params, out_params):
        super(Time2Vec, self).__init__()
        self.embedding = nn.Linear(input_params['dim'], out_params['dim'])
        self.pinput = input_params
        self.poutput = out_params

    @property
    def out_dim(self):
        return self.poutput['dim']


    @flatten_seq_method
    def forward(self, x):
        embedding_out = self.embedding(x)
        out = torch.cat([embedding_out[:, 0].unsqueeze(-1), torch.sin(embedding_out[:, 1:])], dim=1)
        return out

