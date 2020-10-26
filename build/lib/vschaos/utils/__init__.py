# -*-coding:utf-8 -*-
 
"""
    The ``utils`` module
    ========================
 
    This package contains all utility and side functions and classes
 
    Examples
    --------
 
    Subpackages available
    ---------------------
 
    Comments and issues
    ------------------------
    None for the moment
 
    Contributors
    ------------------------
    * Philippe Esling       (esling@ircam.fr)
 
"""
 
# info
__version__ = "1.0"
__author__  = "chemla@ircam.fr"
__date__    = ""
__all__     = []
 
import sys, os

# import sub modules
from . import trajectories
from .onehot import oneHot, fromOneHot
from .schedule import *
from .cage_deform import SerieDeformation
from .misc import *
from .utils_modules import *
from .gather_distrib import *
from .scatter import *
from .dataloader import DataLoader, MultiDataLoader
