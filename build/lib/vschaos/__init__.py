# -*-coding:utf-8 -*-

"""
    The ``datasets`` module
    ========================

    This package contains all datasets classes

    :Example:

    >>> from data.sets import DatasetAudio
    >>> DatasetAudio()

    Subpackages available
    ---------------------

        * Generic
        * Audio
        * Midi
        * References
        * Time Series
        * Pytorch
        * Tensorflow

    Comments and issues
    ------------------------

        None for the moment

    Contributors
    ------------------------

    * Axel Chemla--Romeu-Santos (chemla@ircam.fr)
    * Philippe Esling       (esling@ircam.fr)

"""

# info
__version__ = "0.1.0"
__author__  = "chemla@ircam.fr", "esling@ircam.fr"
__date__    = ""
__all__     = ["criterions", "data", "distributions", "modules", "monitor", "train", "utils", "vaes", "DataParallel"]

import torch, pdb
from torch.nn.parallel.scatter_gather import scatter_kwargs
torch.manual_seed(0)
torch.init_num_threads()

# overriding DataParallel to allow distribution parallelization
# tests for sub-commit
try:
    from matplotlib import pyplot as plt
except:
    import matplotlib
    matplotlib.use('agg')
    from matplotlib import pyplot as plt

class DataParallel(torch.nn.DataParallel):
    def gather(self, *args, **kwargs):
        return utils.gather(*args, **kwargs)

    def __getattr__(self, attribute):
        try:
            return super(DataParallel, self).__getattr__(attribute)
        except AttributeError:
            return getattr(self.module, attribute)

    def scatter(self, inputs, kwargs, device_ids):
        return utils.scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

"""
    def forward(self, *args, **kwargs):
        args = list(args)
        for i in range(len(args)):
            if torch.is_tensor(args[i]):
                if args[i].shape[0] < len(self.device_ids):
                    args[i] = args[i].repeat(len(self.device_ids), *args[i].shape[1:])
        return super(DataParallel, self).forward(*tuple(args),**kwargs)
"""

def load(path, **kwargs):
    loaded_data = torch.load(path, **kwargs)
    return loaded_data


from . import distributions
from . import utils
from . import criterions
from . import data
#from . import misc
from . import modules
from . import monitor
from . import vaes
from . import train
from itertools import chain

from types import SimpleNamespace

# hashes for argument parsing

hashes = SimpleNamespace()
hashes.layer_hash = {'linear':modules.MLPLayer, 'residual':modules.MLPResidualLayer, 'gated':modules.MLPGatedLayer, 'gated_residual':modules.MLPGatedResidualLayer}
hashes.nnlin_hash = {'none':None, 'softplus':"Softplus", "tanh":"Tanh", "softsign":"Softsign"}
hashes.nnlin_args_hash = {'none':{}, 'softplus':{'beta':1}, "tanh":{}, "softsign":{}}
hashes.preprocessing_hash = {'magnitude':data.Magnitude, 'phase':data.Phase, 'polar':data.Polar, 'inst-f':data.InstantaneousFrequency, 'mag+inst-f':data.Polar}

hashes.rec_hash = {'ll':criterions.LogDensity, 'mse':criterions.MSE}
hashes.flow_hash = {'radial': modules.flow.RadialFlow, 'planar':modules.flow.PlanarFlow}
hashes.reg_hash = {'kld':criterions.KLD, 'mmd':criterions.MMD, 'renyi':criterions.RD}
hashes.conv_hash = {'conv': [modules.ConvolutionalLatent, modules.DeconvolutionalLatent],
                    'gated': [modules.GatedConvolutionalLatent, modules.GatedDeconvolutionalLatent],
                    'multi_conv': [modules.MultiHeadConvolutionalLatent, modules.MultiHeadDeconvolutionalLatent],
                    'multi_gated': [modules.MultiHeadGatedConvolutionalLatent, modules.MultiHeadGatedDeconvolutionalLatent]}
hashes.prior_hash = {'isotropic':distributions.priors.IsotropicGaussian, 'wiener':distributions.priors.WienerProcess, 'none':None}
hashes.async_hash = {'random':'RandomPick', 'random_slice':'RandomRangePick', 'all':'Selector', 'sequence':'SequencePick', "sequence_batch": "SequenceBatchPick"}



