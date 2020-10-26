# -*- coding: utf-8 -*-
"""

Neural autoregressive flows (NAF).

Major contributions of the NAF paper is to define a family of sigmoidal flows 
that can be used as transformers in the other autoregressive flows (typically
in the scale-and-shift transform computed in IAF and MAF).

Hence, we can either use directly a Deep Sigmoid Flow (DSF) or the dense version
called Deep Dense Sigmoid Flow (DDSF)

Neural Autoregressive Flow - Huang et al. (2018)
(https://arxiv.org/pdf/1804.00779.pdf).

"""
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
# Internal imports
from flow.flow import Flow
from flow.layers import logsigmoid, amortized_init, sum_dims

class DeepSigmoidFlow(Flow):
    """
    Deep sigmoid flow layer as defined in     
    Neural Autoregressive Flow - Huang et al. (2018)
    (https://arxiv.org/pdf/1705.07057).
    """
    
    def __init__(self, dim, n_hidden=32, amortized='none'):
        super(DeepSigmoidFlow, self).__init__(amortized)
        self.n_hidden = n_hidden
        self.a = amortized_init(amortized, (1, dim))
        self.b = amortized_init(amortized, (1, dim))
        self.w = amortized_init(amortized, (1, dim))
        self.init_parameters()
        self.dim = dim
        self.eps = 1e-6
        
    def _call(self, z):
        """ Forward a batch to apply flow """ 
        self.a = F.softplus(self.a)
        self.w = F.softmax(self.w, dim=1)
        # Compute
        pre_sigm = self.a * z + self.b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = self.w * sigm
        if (len(z.shape) > 2):
            x_pre = torch.sum(self.w * sigm, dim=1)
        x_pre_clipped = x_pre * (1 - self.eps) + self.eps * 0.5
        zp = torch.log(x_pre_clipped) - torch.log(1 - x_pre_clipped)
        return zp

    def _inverse(self, z):
        """ Apply inverse flow """
        return z

    def log_abs_det_jacobian(self, z):
        """ Compute log det Jacobian of flow """
        self.a = F.softplus(self.a)
        self.w = F.softmax(self.w, dim=1)
        pre_sigm = self.a * z + self.b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = self.w * sigm
        if (len(z.shape) > 2):
            x_pre = torch.sum(self.w * sigm, dim=1)
        x_pre_clipped = x_pre * (1 - self.eps) + self.eps * 0.5
        logj = F.log_softmax(self.w, dim=1) + logsigmoid(pre_sigm) + logsigmoid(-pre_sigm) + torch.log(self.a)
        logj = torch.log(torch.sum(torch.exp(logj)))#,2).sum(2)
        logdet = logj + np.log(1 - self.eps) - (torch.log(x_pre_clipped) + torch.log(-x_pre_clipped + 1))
        return sum_dims(logdet)  

    def set_parameters(self, p_list, batch_dim=64):
        if (self.amortized in ['input', 'self']):
            self.a = p_list[:, :self.dim]
            self.b = p_list[:, self.dim:self.dim*2]
            self.w = p_list[:, self.dim*2:]
    
    def n_parameters(self):
        return 3 * self.dim

class DeepDenseSigmoidFlow(Flow):
    """
    Deep dense sigmoid flow layer as defined in     
    Neural Autoregressive Flow - Huang et al. (2018)
    (https://arxiv.org/pdf/1705.07057).
    """
    
    def __init__(self, in_dim, hidden_dim, out_dim, amortized='none'):
        super(DeepDenseSigmoidFlow, self).__init__(amortized)
        self.eps = 1e-6
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.u_ = nn.Parameter(torch.Tensor(hidden_dim, in_dim))
        self.w_ = nn.Parameter(torch.Tensor(out_dim, hidden_dim))
        self.a = amortized_init(amortized, (1, 1, hidden_dim))
        self.b = amortized_init(amortized, (1, 1, hidden_dim))
        self.w = amortized_init(amortized, (1, 1, hidden_dim))
        self.u = amortized_init(amortized, (1, 1, in_dim))
        self.inv = np.log(np.exp(1 - self.eps) - 1) 
        self.init_parameters()
        
    def forward(self, z):
        pre_u = self.u_ + self.u
        pre_w = self.w_ + self.w
        a = F.softplus(self.a + self.inv)
        w = F.softmax(pre_w, dim=3)
        u = F.softmax(pre_u, dim=3)
        # Perform computation
        pre_sigm = torch.sum(u * a * z, 3) + self.b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w * sigm, dim=3)
        x_pre_clipped = x_pre * (1 - self.eps) + self.eps * 0.5
        zp = torch.log(x_pre_clipped) - torch.log(1-x_pre_clipped)
        return zp
    
    def log_abs_det_jacobian(self, z):
        """ Compute log det Jacobian of flow """
        pre_u = self.u_ + self.u
        pre_w = self.w_ + self.w
        a = F.softplus(self.a + self.inv)
        w = F.softmax(pre_w, dim=3)
        u = F.softmax(pre_u, dim=3)
        # Perform computation
        pre_sigm = torch.sum(u * a * z, 3) + self.b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w * sigm, dim=3)
        x_pre_clipped = x_pre * (1 - self.eps) + self.eps * 0.5
        logj = F.log_softmax(pre_w, dim=3) + logsigmoid(pre_sigm) + logsigmoid(-pre_sigm) + torch.log(a)
        # n, d, d2, dh
        logj = logj + F.log_softmax(pre_u, dim=3)
        # n, d, d2, dh, d1
        logj = torch.log(torch.sum(torch.exp(logj),3))
        # n, d, d2, d1
        logdet_ = logj + np.log(1 - self.eps) - (torch.log(x_pre_clipped) + torch.log(-x_pre_clipped + 1))
        return logdet_

    def set_parameters(self, p_list, batch_dim=64):
        if (self.amortized in ['input', 'self']):
            self.a = p_list[:, :self.hidden_dim].unsqueeze(1)
            self.b = p_list[:, self.hidden_dim:self.hidden_dim*2].unsqueeze(1)
            self.w = p_list[:, self.hidden_dim*2:self.hidden_dim*3].unsqueeze(1)
            self.u = p_list[:, self.hidden_dim*3:].unsqueeze(1)
    
    def n_parameters(self):
        return (3 * self.dim) + self.in_dim

