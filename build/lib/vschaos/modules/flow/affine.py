# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import linalg as splin
import numpy as np

# Internal imports
from .flow import Flow
from .layers import amortized_init

class AffineFlow(Flow):
    """
    Normalizing flow version of an affine transform.
    Dimensionality must remain the same, problem is that we call the expensive
    torch.slogdet function (which computes determinant of any matrix)

    Contains one learnable parameter
        - weights (weight matrix of the transform)
    """

    def __init__(self, dim, amortized='none'):
        super(AffineFlow, self).__init__()
        self.dim = dim
        self.weights = amortized_init(amortized, (dim, dim))
        self.amortized = amortized
        self.init_parameters()

    def _call(self, z):
        return z @ self.weights

    def _inverse(self, z):
        return z @ torch.inverse(self.weights)

    def log_abs_det_jacobian(self, z):
        return torch.slogdet(self.weights)[-1].unsqueeze(0).repeat(z.size(0), 1)

    def set_parameters(self, params, batch_dim):
        """ Set parameters values (sub-modules) """
        if (self.amortized != 'none'):
            self.weights = params.view(self.dim, self.dim)

    def n_parameters(self):
        """ Return number of parameters in flow """
        return (self.dim ** 2);

## Affine Flow with LU decomposition
class AffineLUFlow(Flow):
    """
    Normalizing flow version of an affine transform with LU decomposition
    Here the LU decomposition allows to avoid computing the expensive determinant
    and to rely on the diagonal component of the decomposition
    W = P * L * (U + diag(s))

    Contains three learnable parameter
        - P
        - L
        - s
    """

    def __init__(self, dim, bias=None, amortized='none'):
        super(AffineLUFlow, self).__init__()
        if bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.zeros((1, dim)))
        self.dim = dim
        weights = torch.Tensor(dim, dim)
        nn.init.orthogonal_(weights)
        # Compute the parametrization
        P, L, U = splin.lu(weights.numpy())
        self.P = torch.Tensor(P)
        self.L = nn.Parameter(torch.Tensor(L))
        self.U = nn.Parameter(torch.Tensor(U))
        # Need to create masks for enforcing triangular matrices
        self.mask_low = torch.tril(torch.ones(weights.size()), -1)
        self.mask_up = torch.triu(torch.ones(weights.size()), -1)
        self.I = torch.eye(weights.size(0))
        # Now compute s
        self.s = nn.Parameter(torch.Tensor(np.diag(U)))
        # Register buffers for CUDA call
        self.register_buffer('P_c', self.P)
        self.register_buffer('mask_low_c', self.mask_low)
        self.register_buffer('mask_up_c', self.mask_up)
        self.register_buffer('I_c', self.I)

    def _call(self, z):
        L = self.L * self.mask_low + self.I
        U = self.U * self.mask_up + torch.diag(self.s)
        weights = self.P @ L @ U
        return F.linear(z, weights, self.bias)

    def _inverse(self, z):
        L = self.L * self.mask_low + self.I
        U = self.U * self.mask_up + torch.diag(self.s)
        weights = self.P @ L @ U
        return (z - self.bias) @ torch.inverse(weights)

    def log_abs_det_jacobian(self, z):
        return self.s.abs().log().sum().unsqueeze(0).repeat(z.size(0), 1)

    def set_parameters(self, params, batch_dim):
        """ Set parameters values (sub-modules) """
        if self.amortized != 'none':
            self.L = params[:, :(self.dim**2)]
            self.U = params[:, (self.dim**2):(self.dim**2)*2]
            self.s = params[:, (self.dim**2)*2:(self.dim**2)*3]
            if self.bias is not None:
                self.bias.data = params[:, (self.dim**2)*3:(self.dim**2)*3 + self.dim]

    def n_parameters(self):
        """ Return number of parameters in flow """
        return (self.dim ** 2) * 3 + (self.dim if self.bias is not None else 0)

