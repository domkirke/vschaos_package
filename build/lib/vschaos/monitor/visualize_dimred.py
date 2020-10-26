#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 19:19:14 2017

@author: chemla
"""
#import torch
#from torch.autograd import Variable

import torch
import numpy as np
import sklearn.manifold as manifold
import sklearn.decomposition as decomposition
from ..utils.dataloader import DataLoader
from ..utils import decudify, merge_dicts, flatten_seq_method

try:
    from matplotlib import pyplot as plt
except:
    import matplotlib 
    matplotlib.use('agg')
    from matplotlib import pyplot as plt
    
    
from numpy.random import permutation


#######################################################################
##########      Transformations
###

class Embedding(object):

    def __init__(self, *args, **kwargs):
        pass 
    
    def transform(*args, **kwargs):
        return np.array([])
        
    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

class Dummy(Embedding):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, z):
        return z

    def __invert__(self, z):
        return z



class PCA(decomposition.PCA, Embedding):
    invertible = True

    def __repr__(self):
        return "PCA"

    def __init__(self, *args, **kwargs):
        kwargs['n_components'] = int(kwargs.get('n_components', 3))
        super(PCA, self).__init__(*args, **kwargs)

    @property
    def dim(self):
        return self.n_components_

    def fit(self, out, sample=False):
        params = out['out_params']
        if sample:
            # sample from distributions
            orig_z = params.sample()
        else:
            # sample mean
            orig_z = params.mean
        orig_z = orig_z.cpu().detach().numpy()
        if len(orig_z.shape) > 2:
            orig_z = orig_z.reshape(np.cumprod(orig_z.shape[:-1])[-1], orig_z.shape[-1])
        super(PCA, self).fit(orig_z)

    @flatten_seq_method
    def transform(self, X):
        shape = X.shape
        if len(X.shape) > 2:
            X = X.reshape(np.cumprod(X.shape[:-1])[-1], X.shape[-1])
        return super(PCA, self).transform(X).reshape((*shape[:-1], self.n_components_))

class ICA(decomposition.FastICA, Embedding):
    invertible = True

    def __repr__(self):
        return "ICA"

    @property
    def dim(self):
        return self.n_components

    def __init__(self, *args, **kwargs):
        kwargs['n_components'] = int(kwargs.get('n_components', 3))
        super(ICA, self).__init__(*args, **kwargs)

    def fit(self, out, sample=False):
        params = out['out_params']
        if sample:
            # sample from distributions
            orig_z = params.sample()
        else:
            # sample mean
            orig_z = params.mean
        orig_z = orig_z.cpu().detach().numpy()
        if len(orig_z.shape) > 2:
            orig_z = orig_z.reshape(np.cumprod(orig_z.shape[:-1])[-1], orig_z.shape[-1])
        super(ICA, self).fit(orig_z)

    def transform(self, X):
        shape = X.shape
        if len(X.shape) > 2:
            X = X.reshape(np.cumprod(X.shape[:-1])[-1], X.shape[-1])
        return super(ICA, self).transform(X).reshape((*shape[:-1], self.n_components))

class LocallyLinearEmbedding(manifold.LocallyLinearEmbedding, Embedding):
    invertible = False
    def __init__(self, *args, **kwargs):
        super(LocallyLinearEmbedding, self).__init__(*args, **kwargs)

class MDS(manifold.MDS, Embedding):
    invertible = False
    def __init__(self, *args, **kwargs):
        super(MDS, self).__init__(*args, **kwargs)

class TSNE(manifold.TSNE, Embedding):
    invertible = False
    def __init__(self, *args, **kwargs):
        super(TSNE, self).__init__(*args, **kwargs)

#TODO : implement ICA,  Independant ICA and Topographic ICA
#class ICA()


class DimensionSlice(Embedding):
    invertible = True
    def __init__(self, *args, fill_value = 0, **kwargs):
        super(DimensionSlice, self).__init__(*args, **kwargs)
        self.dimensions = np.array(args)
        self.fill_value = fill_value
        self.init_dim = None
        
    def fit(self, z):
        if z.ndim == 1:
            self.init_dim = z.shape[0]
        else:
            self.init_dim = z.shape[-1]
    
    def transform(self, z):
        return z[self.dimensions]
    
    def inverse_transform(self, z, target_dim=None):
        if target_dim is None:
            target_dim = self.init_dim
        if self.init_dim is None:
            raise Exception('[Warning] Please inform a target dimension to invert data.')
        if z.ndim == 1:
            invert_z = np.zeros(target_dim)
            invert_z[self.dimensions] = z
        else:
            invert_z = np.zeros((z.shape[0], target_dim))
            invert_z[:, self.dimensions] = np.squeeze(z)
        return invert_z


class DimensionSliceSort(Embedding):
    invertible = True
    def __init__(self, n_components=3, descending=False, **kwargs):
        super(DimensionSliceSort, self).__init__()
        self.dim = int(n_components)
        self.init_dim = None
        self.dimensions = None
        self.descending = bool(int(descending))

    def fit(self, out):
        params = out['out_params']
        if params.stddev.shape == 1:
            self.init_dim = params.stddev.shape[0]
            self.dimensions = torch.argsort(params.stddev, descending=self.descending)[:self.dim].int().numpy()
        else:
            mean_of_vars = params.stddev.mean(dim=0)
            self.init_dim = params.mean.shape[1]
            self.dimensions = torch.argsort(mean_of_vars, descending=self.descending)[:self.dim].int().numpy()

    def transform(self, z):
        if z.ndim == 1:
             return z[self.dimensions]
        else:
             return z[:, self.dimensions]

    def inverse_transform(self, z, target_dim=None):
        if target_dim is None:
            target_dim = self.init_dim
        if self.init_dim is None:
            raise Exception('[Warning] Please inform a target dimension to invert data.')
        if z.ndim == 1:
            invert_z = np.zeros(target_dim)
            invert_z[self.dimensions] = z
        else:
            invert_z = np.zeros((z.shape[0], target_dim))
            invert_z[:, self.dimensions] = np.squeeze(z)
        return invert_z

        
    
    
#######################################################################
##########      Manifold (points + transformation)
###
        
class Manifold(object):
    def __init__(self, transformation, z):
        super(Manifold, self).__init__()
        self.transformation = transformation
        self.orig_z = z
        self.z = transformation.transform(z)
        self.anchors = {}
        self.ids = None


    def __call__(self, z, **kwargs):
        if torch.is_tensor(z):
            z = z.detach().cpu().numpy()
        if self.transformation is not None:
            return self.transformation.transform(z)
        else:
            return z

    def invert(self, z):
        if torch.is_tensor(z):
            z = z.detach().cpu().numpy()
        if self.transformation:
            if self.transformation.invertible:
                return self.transformation.inverse_transform(z)
            else:
                raise Exception('transformation %s of manifold %s is not invertible'%(self.transformation, self))
        else:
            return z


class LatentManifold(Manifold):
    def __init__(self, transformation, model, dataset, preprocessing=None, loader=DataLoader, n_points=None, layer=-1, *args, **kwargs):
        self.layer = layer
        self.anchors = {}
        self.transformation = None
        self.platent = model.platent[layer]

        if transformation is not None:
            self.transformation = transformation(**kwargs)
            if not n_points is None:
                self.ids = np.random.permutation(len(dataset))[:int(n_points)]
            else:
                self.ids = range(len(dataset))

            dataset = dataset.retrieve(self.ids)
            loader = loader or DataLoader
            loader = loader(dataset, kwargs.get('batch_size', None), tasks=kwargs.get('tasks'), preprocessing=preprocessing)
            model.eval()
            with torch.no_grad():
                outs = []
                for x, y in loader:
                    x = model.format_input_data(x)
                    outs.append(decudify(model.encode(x, y=y)))
            outs = merge_dicts(outs)
            self.transformation.fit(outs[layer])

    def add_anchor(self, name, *args):
        self.anchors[name] = np.array(args)

    @property
    def dim(self):
        if self.transformation:
            return self.transformation.dim
        else:
            return self.platent['dim']


        
