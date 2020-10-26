# -*-coding:utf-8 -*-
 
"""
    The ``flow`` module
    ========================
 
    This package contains all normalizing and generative flow classes
 
    :Example:
 
    >>> 
 
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
        
    * Philippe Esling       (esling@ircam.fr)
 
"""

from .activation import PReLUFlow
from .affine import AffineFlow, AffineLUFlow
from .convolution import Invertible1x1ConvFlow
from .coupling import AffineCouplingFlow, MaskedCouplingFlow, ConvolutionalCouplingFlow
from .flow import Flow, NormalizingFlow, NormalizingFlowContext, GenerativeFlow, FlowList
from .householder import HouseholderFlow
from .iaf import IAFlow, ContextIAFlow, DDSF_IAFlow
from .maf import MAFlow, ContextMAFlow, DDSF_MAFlow
from .naf import *
from .normalization import BatchNormFlow, ActNormFlow
from .order import ReverseFlow, ShuffleFlow, SplitFlow, SqueezeFlow
from .planar import PlanarFlow
from .radial import RadialFlow
from .sylvester import SylvesterFlow, TriangularSylvesterFlow
 
# info
__version__ = "1.0"
__author__  = "esling@ircam.fr, chemla@ircam.fr"
__date__    = ""
__all__     = [
    'PReLUFlow', 
    'AffineFlow', 'AffineLUFlow',
    'Invertible1x1ConvFlow',
    'AffineCouplingFlow', 'MaskedCouplingFlow', 'ConvolutionalCouplingFlow',
    'Flow', 'NormalizingFlow', 'NormalizingFlowContext', 'GenerativeFlow', 'FlowList',
    'HouseholderFlow',
    'IAFlow', 'ContextIAFlow', 'DDSF_IAFlow',
    'MAFlow', 'ContextMAFlow', 'DDSF_MAFlow',
    'DeepDenseSigmoidFlow', 'DeepSigmoidFlow',
    'BatchNormFlow', 'ActNormFlow',
    'ReverseFlow', 'ShuffleFlow', 'SplitFlow', 'SqueezeFlow',
    'PlanarFlow', 'RadialFlow',
    'SylvesterFlow', 'TriangularSylvesterFlow'
]




# import sub modules
#from . import ar
#from . import basic
#from . import flow
#from . import generative
#from . import layers
#from . import order
#from . import temporal

