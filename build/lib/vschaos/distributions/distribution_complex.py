import math
from numbers import Number

import torch, numpy as np
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import _standard_normal, broadcast_all
from utils import complex as complex


class CSCNormal(ExponentialFamily):
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by
    :attr:`loc` and :attr:`scale`.

    Example::

        >>> m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # normally distributed with loc=0 and scale=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution
            (often referred to as sigma)
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

    def __init__(self, scale, validate_args=None):
        if not issubclass(scale, complex.ComplexTensor):
            scale = complex.ComplexTensor(scale)
        self.scale = scale
        self.loc = complex.ComplexTensor(self.scale.zero())
        batch_shape = scale.shape
        super(CSCNormal, self).__init__(batch_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        eps = complex.ComplexTensor(torch.zeros(self.scale.shape[:-1], device=self.scale.device))
        eps_scaled = complex.bcmm(self.scale, eps)
        return complex.ComplexTensor(0.5 * eps_scaled)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        precision = complex.cinv(self.scale)
        return torch.log(self.scale.det()) - complex.bcmm(complex.bcmm(value.conj(), precision), value) - self.scale.rshape[1] * np.log(np.pi)

