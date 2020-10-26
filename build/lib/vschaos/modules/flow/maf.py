"""

Masked Autoregressive Flow (MAF) implementation

Masked Autoregressive Flow for Density Estimation - Papamakarios et al. (2017)
(https://arxiv.org/pdf/1705.07057).

"""
# -*- coding: utf-8 -*-
import torch
from torch import nn
# Internal imports
from .flow import Flow
from .utils import get_nelement
from .layers import MaskedLinear
from .naf import *


class MAFlow(Flow):
    """
    Masked autoregressive flow layer as defined in
    Masked Autoregressive Flow for Density Estimation - Papamakarios et al. (2017)
    (https://arxiv.org/pdf/1705.07057).
    """

    def __init__(self, dim, n_hidden=32, n_context=0, n_layers=2, activation=nn.ReLU, amortized='none'):
        super(MAFlow, self).__init__()
        self.n_hidden = n_hidden
        self.n_context = n_context
        # Create auto-regressive transform
        self.transform = self.transform_net(dim, n_hidden, activation)
        self.init_parameters()
        self.bijective = True

    def create_mask(self, in_f, out_f, in_flow, m_type = None):
        in_degrees = torch.arange(in_f) % (in_flow - 1)
        out_degrees = torch.arange(out_f) % (in_flow - 1)
        if m_type == 'input':
            in_degrees = torch.arange(in_f) % in_flow
        if m_type == 'output':
            out_degrees = torch.arange(out_f) % in_flow - 1
        return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()

    def transform_net(self, dim, n_hidden, activation):
        input_mask = self.create_mask(dim, n_hidden, dim, 'input')
        hidden_mask = self.create_mask(n_hidden, n_hidden, dim)
        output_mask = self.create_mask(n_hidden, dim * 2, dim, 'output')
        net = nn.Sequential(
                MaskedLinear(dim, n_hidden, input_mask),
                activation(),
                MaskedLinear(n_hidden, n_hidden, hidden_mask),
                activation(),
                MaskedLinear(n_hidden, dim * 2, output_mask))
        return net

    def _call(self, z):
        """ Forward a batch to apply flow """
        mu, log_var = self.transform(z).chunk(2, 1)
        zp = (z - mu) * torch.exp(-log_var)
        return zp

    def _inverse(self, z):
        """ Apply inverse flow """
        zp = torch.randn_like(z)
        for col in range(z.shape[1]):
            mu, log_var = self.transform(zp).chunk(2, 1)
            zp[:, col] = z[:, col] * torch.exp(log_var[:, col]) + mu[:, col]
        return zp

    def log_abs_det_jacobian(self, z):
        """ Compute log det Jacobian of flow """
        mu, log_var = self.transform(z).chunk(2, 1)
        return -torch.sum(log_var, -1, keepdim=True)

    def n_parameters(self):
        return get_nelement(self.transform)


class ContextMAFlow(MAFlow):
    """
    Contextualized version of the Masked Autoregressive Flow.
    """

    def __init__(self, dim, n_hidden=32, context=0, n_layers=2, activation=nn.ReLU, amortized='none'):
        super(ContextMAFlow, self).__init__(dim, n_hidden=32, context=0, n_layers=2, activation=nn.ReLU, amortized='none')
        self.context = MaskedLinear(dim, n_hidden, self.create_mask(dim, n_hidden, dim, 'input'))
        self.transform = (activation(),
                          MaskedLinear(dim, n_hidden, self.create_mask(n_hidden, n_hidden, dim)),
                          activation(),
                          MaskedLinear(dim, n_hidden, self.create_mask(n_hidden, dim * 2, dim, 'output')))

    def _call(self, z, h=None):
        """ Forward a batch to apply flow """
        # Mix input and context
        h = self.context(z, h)
        mu, log_var = self.transform(h).chunk(2, 1)
        zp = (z - mu) * torch.exp(-log_var)
        return zp

    def _inverse(self, z, h):
        """ Apply inverse flow """
        zp = torch.zeros_like(z)
        for col in range(z.shape[1]):
            h_t = self.context(zp, h)
            mu, log_var = self.transform(h_t).chunk(2, 1)
            zp[:, col] = z[:, col] * torch.exp(log_var[:, col]) + mu[:, col]
        return zp

    def log_abs_det_jacobian(self, z, h):
        """ Compute log det Jacobian of flow """
        h = self.context(z, h)
        mu, log_var = self.transform(h).chunk(2, 1)
        return -torch.sum(log_var, -1, keepdim=True)


class DDSF_MAFlow(MAFlow):

    """
    Deep Sigmoid Flow version of MAF, from
    Neural Autoregressive Flow - Huang et al. (2018)
    (https://arxiv.org/pdf/1705.07057).
    """

    def __init__(
        self, dim, ddsf=None, n_context=0, n_layers=2, activation=nn.ELU,
        amortized='none', forget_bias=1.
    ):
        if ddsf is None:
            ddsf = DeepDenseSigmoidFlow(dim, amortized=amortized)
        super().__init__(
            dim, ddsf.n_parameters(), n_context, n_layers, activation,
            amortized
        )
        self.ddsf = ddsf
        self._cache = None  # Cache for log_abs_det_jacobian

    def _call(self, z):
        params = self.transform(z)
        self.ddsf.set_parameters(params)
        output = self.ddsf(z)
        self._cache = {z: self.ddsf.log_abs_det_jacobian(z)}
        return output 

    def log_abs_det_jacobian(self, z):
        if self._cache is None or z not in self._cache:
            _ = self(z)
        return self._cache[z]

