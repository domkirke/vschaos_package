import math, pdb
import warnings
from functools import total_ordering

import torch
from torch._six import inf

from torch.distributions.beta import Beta
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.distribution import Distribution
from torch.distributions.exponential import Exponential
from torch.distributions.gamma import Gamma
from torch.distributions.laplace import Laplace
from torch.distributions.multivariate_normal import MultivariateNormal#, _batch_diag, _batch_mahalanobis,
                                  #_batch_trtrs_lower)
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.uniform import Uniform
from torch.distributions.utils import _sum_rightmost
from torch.distributions import kl
from numpy import nan

_JSD_REGISTRY = {}
_JSD_MEMOIZE = {}

def register_jsd(type_p, type_q):
    def decorator(fun):
        _JSD_REGISTRY[type_p, type_q] = fun
        _JSD_MEMOIZE.clear()  # reset since lookup order may have changed
        return fun
    return decorator

@total_ordering
class _Match(object):
    __slots__ = ['types']

    def __init__(self, *types):
        self.types = types

    def __eq__(self, other):
        return self.types == other.types

    def __le__(self, other):
        for x, y in zip(self.types, other.types):
            if not issubclass(x, y):
                return False
            if x is not y:
                break
        return True



def _dispatch_js(type_p, type_q):
    """
    Find the most specific approximate match, assuming single inheritance.
    """
    matches = [(super_p, super_q) for super_p, super_q in _JSD_REGISTRY
               if issubclass(type_p, super_p) and issubclass(type_q, super_q)]
    if not matches:
        return NotImplemented
    # Check that the left- and right- lexicographic orders agree.
    left_p, left_q = min(_Match(*m) for m in matches).types
    right_q, right_p = min(_Match(*reversed(m)) for m in matches).types
    left_fun = _JSD_REGISTRY[left_p, left_q]
    right_fun = _JSD_REGISTRY[right_p, right_q]
    if left_fun is not right_fun:
        warnings.warn('Ambiguous renyi_divergence({}, {}). Please register_renyi({}, {})'.format(
            type_p.__name__, type_q.__name__, left_p.__name__, right_q.__name__),
            RuntimeWarning)
    return left_fun


def _infinite_like(tensor):
    """
    Helper function for obtaining infinite renyi Divergence throughout
    """
    return tensor.new_tensor(inf).expand_as(tensor)


def _x_log_x(tensor):
    """
    Utility function for calculating x log x
    """
    return tensor * tensor.log()


def _batch_trace_XXT(bmat):
    """
    Utility function for calculating the trace of XX^{T} with X having arbitrary trailing batch dimensions
    """
    n = bmat.size(-1)
    m = bmat.size(-2)
    flat_trace = bmat.reshape(-1, m * n).pow(2).sum(-1)
    return flat_trace.reshape(bmat.shape[:-2])


def js_divergence(p, q, alpha):
    r"""
    Compute Jensen-Shannon divergence :math:`JSD(p \| q)` between two distributions.

    Args:
        p (Distribution): A :class:`~torch.distributions.Distribution` object.
        q (Distribution): A :class:`~torch.distributions.Distribution` object.

    Returns:
        Tensor: A batch of renyi divergences of shape `batch_shape`.

    Raises:
        NotImplementedError: If the distribution types have not been registered via
            :meth:`register_js`.
    """
    try:
        fun = _JSD_MEMOIZE[type(p), type(q)]
    except KeyError:
        fun = _dispatch_js(type(p), type(q))
        _JSD_MEMOIZE[type(p), type(q)] = fun
    if fun is NotImplemented:
        raise NotImplementedError
    return fun(p, q, alpha)

################################################################################
# Jensen_Shannon Divergence Implementations
################################################################################


@register_jsd(Normal, Normal)
def _js_normal_normal(p,q, alpha=0.5):
    if len(p.batch_shape) > 2:
        cum_size = torch.cumprod(torch.tensor(p.batch_shape[:-1]), 0)[-1]
        p = p.view(cum_size, p.batch_shape[-1])
    if len(q.batch_shape) > 2:
        cum_size = torch.cumprod(torch.tensor(q.batch_shape[:-1]), 0)[-1]
        q = q.view(cum_size, q.batch_shape[-1])
    original_shape = p.batch_shape
    harmonic_std = torch.reciprocal((1 - alpha)*torch.reciprocal(p.variance) + alpha * torch.reciprocal(q.variance))
    harmonic_mean = torch.bmm(harmonic_std.diag_embed(), torch.bmm((1-alpha) * (p.variance.reciprocal()).diag_embed(), p.mean.unsqueeze(-1)) +  alpha * torch.bmm((q.variance.reciprocal()).diag_embed(), q.mean.unsqueeze(-1))).squeeze(-1)
    harmonic_dist = Normal(harmonic_mean, torch.sqrt(harmonic_std))
    div = (1 - alpha)*kl.kl_divergence(p, harmonic_dist) + alpha * kl.kl_divergence(q, harmonic_dist)
    return div.view(*original_shape)
