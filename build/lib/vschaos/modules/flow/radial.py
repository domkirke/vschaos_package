# -*- coding: utf-8 -*-
import torch
from torch.nn import functional as F
# Internal imports
from .flow import Flow
from .layers import amortized_init, sum_dims
import pdb

class RadialFlow(Flow):
    """
    Radial normalizing flow, as defined in
    Variational Inference with Normalizing Flows - Rezende et al. (2015)
    http://proceedings.mlr.press/v37/rezende15.pdf
    """

    def __init__(self, dim, amortized='none'):
        """Initialize normalizing flow."""
        super(RadialFlow, self).__init__()
        self.z0 = amortized_init(amortized, (1, dim))
        self.alpha = amortized_init(amortized, (1, 1))
        self.beta = amortized_init(amortized, (1, 1))
        self.amortized = amortized
        self.dim = dim
        self.init_parameters()

    def _call(self, z):
        r = torch.norm(z - self.z0, 2, dim=1).unsqueeze(1)
        h = 1 / (self.alpha + r + 1e-5)
        return z + (self.beta * h * (z - self.z0))

    def log_abs_det_jacobian(self, z):
        r = torch.norm(z - self.z0, 2, dim=1).unsqueeze(1)
        h = 1 / (self.alpha + r + 1e-5)
        hp = - 1 / ((self.alpha + r) ** 2 + 1e-5)
        bh = self.beta * h
        det_grad = ((1 + bh) ** (self.dim - 1)) * (1 + bh + (self.beta * hp * r))
        return sum_dims(torch.log(det_grad.abs() + 1e-9)).unsqueeze(0)

    def set_parameters(self, p_list, rep_dim=64):
        if (self.amortized != 'none'):
            self.z0 = p_list[:, :self.dim]
            self.alpha = F.softplus(p_list[:, self.dim]).unsqueeze(1)
            self.beta = p_list[:, self.dim+1].unsqueeze(1)
        # Handle self or no amortization
        if (self.amortized in ['none', 'self']):
            self.z0 = self.z0.repeat(rep_dim, 1)
            self.alpha = self.alpha.repeat(rep_dim, 1)
            self.beta = self.beta.repeat(rep_dim, 1)

    def n_parameters(self):
        """Return number of parameters in flow."""
        return self.dim + 2

