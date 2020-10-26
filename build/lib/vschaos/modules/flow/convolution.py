# -*- coding: utf-8 -*-
import torch, pdb
from torch import nn
import torch.nn.functional as F
from scipy import linalg as splin
import numpy as np
# Internal imports
from .flow import Flow

class Invertible1x1ConvFlow(Flow):
    """
    Implementation of the invertible 1x1 convolution layer defined in 
    Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    
    This code is a modified version of the repository
    https://github.com/rosinality/glow-pytorch
    
    """
    
    def __init__(self, num_channels, LU_decomposed=True, amortized='none'):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not LU_decomposed:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = splin.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)
            self.p = torch.Tensor(np_p.astype(np.float32))
            self.sign_s = torch.Tensor(np_sign_s.astype(np.float32))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, z, reverse):
        w_shape = self.w_shape
        pixels = z.shape[2] * z.shape[3]
        if not self.LU:
            dlogdet = torch.slogdet(self.weight)[1] * pixels
            if not reverse:
                weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
            else:
                weight = torch.inverse(self.weight.double()).float()\
                              .view(w_shape[0], w_shape[1], 1, 1)
            return weight, dlogdet.repeat(z.shape[0])
        else:
            self.p = self.p.to(z.device)
            self.sign_s = self.sign_s.to(z.device)
            self.l_mask = self.l_mask.to(z.device)
            self.eye = self.eye.to(z.device)
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            dlogdet = (torch.sum(self.log_s) * pixels).repeat(z.shape[0])
            if not reverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.double()).float()
                u = torch.inverse(u.double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
            return w.view(w_shape[0], w_shape[1], 1, 1), dlogdet

    def _call(self, z):
        weight, dlogdet = self.get_weight(z, False)
        z = F.conv2d(z, weight)
        return z
    
    def _inverse(self, z):
        weight, dlogdet = self.get_weight(z, True)
        z = F.conv2d(z, weight)
        return z
    
    def log_abs_det_jacobian(self, z):
        weight, dlogdet = self.get_weight(z, False)
        return dlogdet
