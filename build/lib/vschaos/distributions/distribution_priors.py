import pdb, abc
import torch
from torch import zeros, ones, eye
from . import Bernoulli, Normal, MultivariateNormal, Categorical, RandomWalk, Multinomial
from . import Distribution
from .distribution_flow import Flow


def IsotropicGaussian(batch_size=None, device="cpu", requires_grad=False, **kwargs):
    assert batch_size
    return Normal(zeros(*tuple(batch_size), device=device, requires_grad=requires_grad),
            ones(*batch_size, device=device, requires_grad=requires_grad))

def WienerProcess(batch_size=None, device="cpu", requires_grad=False, **kwargs):
    assert batch_size
    return RandomWalk(zeros(*batch_size, device=device, requires_grad=requires_grad),
            ones(*batch_size, device=device, requires_grad=requires_grad))

def IsotropicMultivariateGaussian(batch_size=None, device="cpu", requires_grad=False, **kwargs):
    assert batch_size
    return MultivariateNormal(zeros(*batch_size, device=device, requires_grad=requires_grad),
            covariance_matrix=eye(*batch_size, device=device, requires_grad=requires_grad))


class Prior(object):
    @abc.abstractmethod
    def __init__(self):
        super(Prior, self).__init__()

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class ClassPrior(Prior):

    def __init__(self, task, dist_type, init_args={}, device=None):
        super(ClassPrior, self).__init__()
        self.dist_type = dist_type
        self.init_args = init_args
        self.task = task
        self.device = device

    def get_prior(self, y):
        params = {}
        for k in self.init_args.keys():
            params[k] = []
        for i in range(y.shape[0]):
            for k,v in self.init_args.items():
                params[k].append(torch.from_numpy(v[y[i]]))
        for k, v in params.items():
            params[k] = torch.stack(v)
            if self.device is not None:
                params[k] = params[k].to(self.device)
        if issubclass(self.dist_type, Normal):
            return Normal(params['mean'], params['stddev'])
        return self.dist_type(**params)


    def __call__(self, *args, y=None, **kwargs):
        assert y.get(self.task) is not None
        return self.get_prior(y[self.task])




def get_default_distribution(distrib_type, batch_shape, device="cpu", requires_grad=False):
    if issubclass(type(distrib_type), Flow):
        distrib_type = distrib_type.dist
    if distrib_type == Normal:
        return IsotropicGaussian(batch_shape, device=device, requires_grad=requires_grad)
    if distrib_type == MultivariateNormal:
        return IsotropicMultivariateGaussian(batch_shape, device=device, requires_grad=requires_grad)
    if distrib_type == RandomWalk:
        return WienerProcess(batch_shape, device=device, requires_grad=requires_grad)
    elif distrib_type == Multinomial:
        probs = torch.full(batch_shape, 1/batch_shape[-1]).to(device).requires_grad_(requires_grad)
        return Multinomial(probs=probs)
    else:
        raise TypeError("Unknown default distribution for distribution %s"%distrib_type)
