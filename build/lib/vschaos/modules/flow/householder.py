# -*- coding: utf-8 -*-
import torch
from torch import nn
# Internal imports
from .flow import Flow

class HouseholderFlow(Flow):
    """
    Householder normalizing flow, as defined in
    Improving Variational Auto-Encoders using Householder Flow - Tomczak et al. (2016)
    https://arxiv.org/pdf/1611.09630
    """
    
    def __init__(self, dim, n_hidden=64, n_layers=3, activation=nn.ReLU, amortized='none'):
        """ 
        Initialize normalizing flow 
        """
        super(HouseholderFlow, self).__init__()
        self.v = []
        if (amortized == 'none'):
            self.v = nn.Parameter(torch.Tensor(1, dim))
        self.amortized = amortized
        self.init_parameters()
        self.bijective = True
        self.dim = dim
        
    def _call(self, z):
        """
        :param z: batch_size (B) x latent_size (L)
        :return: z_new = z - 2* v v_T / norm(v,2) * z
        """
        # v * v_T
        vvT = torch.bmm(self.v.unsqueeze(2), self.v.unsqueeze(1))
        # v * v_T * z
        vvTz = torch.bmm(vvT, z.unsqueeze(2) ).squeeze(2) 
        # calculate norm ||v||^2
        norm_sq = torch.sum(self.v * self.v, 1).unsqueeze(1)
        norm_sq = norm_sq.expand(norm_sq.size(0), self.v.size(1))
        # calculate new z
        z_new = z - 2 * vvTz / norm_sq
        return z_new

    def _inverse(self, z):
        raise Exception('Not implemented')

    def log_abs_det_jacobian(self, z):
        return 0
    
    def set_parameters(self, params, batch_dim):
        """ Set parameters values (sub-modules) """
        if (self.amortized != 'none'):
            self.v = params[:, :]
        if (self.amortized != 'input'):
            self.v = self.v.repeat(batch_dim, 1)
    
    def n_parameters(self):
        """ Return number of parameters in flow """
        return self.dim;flo
