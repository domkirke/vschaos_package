
# -*-coding:utf-8 -*-
 
"""
    The ``criterions`` module
    ========================
 
    This package contains different criterions and criterion components for VAE training
 

    Comments and issues
    ------------------------
        
        None for the moment
 
    Contributors
    ------------------------
        
    * Axel Chemla--Romeu-Santos (chemla@ircam.fr)
 
"""

# import sub modules
import pdb
from .. import utils
from .criterion_criterion import *
from .criterion_logdensities import *
from .criterion_divergence import *
from .criterion_functional import *
from .criterion_spectral import SpectralLoss
from .criterion_elbo import *
from .criterion_misc import *

