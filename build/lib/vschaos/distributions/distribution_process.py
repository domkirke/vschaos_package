import torch, numpy as np, pdb
from . import Distribution, SingularMultivariateNormal, MatrixVariateNormalDistribution
from torch.distributions import kl
from sklearn.gaussian_process.kernels import Kernel, RBF
from . import Normal

def get_process_from_normal(normal_dist):
    mean = torch.cat([normal_dist.mean[:, 0].unsqueeze(1), normal_dist.mean[:, 1:] - normal_dist.mean[:,:-1]], dim=1)
    variance = torch.cat([normal_dist.variance[:, 0].unsqueeze(1), normal_dist.variance[:, 1:] + normal_dist.variance[:,:-1]], dim=1)
    return RandomWalk(mean, variance.sqrt())

class Process(Distribution):
    continuous_time=False
    def __init__(self, batch_shape=torch.Size(), event_shape=torch.Size(), sequence_shape=torch.Size(), **kwargs):
        super(Process, self).__init__(batch_shape, event_shape, **kwargs)
        self._sequence_shape = sequence_shape

# Trick described in Bayer & al.
class RandomWalk(Normal, Process):
    def __init__(self, *args, validate_args=None, diagonal=True):
        if len(args)==1:
            process = get_process_from_normal(args[0])
            Normal.__init__(self, process.mean, process.stddev, validate_args=validate_args)
        if len(args)==2:
            loc, scale = args
            assert len(loc.shape) > 2, "locations for Random Walk object must have ndims > 2"
            Normal.__init__(self, loc, scale, validate_args=validate_args)
        self.diagonal = diagonal

    def sample(self, sample_shape=torch.Size()):
        diffs = Normal(self.mean, self.stddev).sample(sample_shape)
        return torch.cumsum(diffs, 1)

    def rsample(self, sample_shape=torch.Size()):
        diffs = Normal(self.mean, self.stddev).rsample(sample_shape)
        return torch.cumsum(diffs, 1)

    def log_prob(self, value):
        assert len(value.shape) > 2, "input value must have ndim > 2, got shape %s"%value.shape
        value_diffs = torch.cat([value[:, 0], value[:, 1:] - value[:, -1]], axis=1)
        return Normal.log_prob(self, value_diffs)

@kl.register_kl(RandomWalk, RandomWalk)
def kl_weiner_weiner(p, q):
    return kl._kl_normal_normal(p, q)

@kl.register_kl(RandomWalk, Normal)
def kl_weiner_normal(p, q):
    return kl._kl_normal_normal(p, get_process_from_normal(q))

@kl.register_kl(Normal, RandomWalk)
def kl_normal_weiner(p, q):
    return kl._kl_normal_normal(get_process_from_normal(p), q)

### Gaussian processes

# Kernel objects for GP regression

def squareform(x):
    s = x.shape
    if len(s) == 2:
        return torch.stack([squareform(x_tmp) for x_tmp in x], dim=0)

    x = x.contiguous()
    d = int(np.ceil(np.sqrt(s[0] * 2)))
    if d * (d - 1) != s[0] * 2:
        raise ValueError('invalid shape %s ; must be of form N(N-1)')
    out = torch.zeros(d, d, requires_grad=x.requires_grad, device=x.device)
    running_id = 0
    for i in range(d):
        for j in range(i+1, d):
            out[i, j] = x[running_id]
            running_id += 1
    out = out + out.t()
    return out

class Kernel(object):
    def __init__(self):
        pass

    def __call__(self, x, y=None, **kwargs):
        return x

class RBF(Kernel):
    def __repr__(self):
        return "RBF(length=%s, order=%s, variance=%s)"%(self.length, self.order, self.variance)
    def __init__(self, length=1, order=5, variance=0.1):
        self.length = float(length)
        self.order = int(order)
        self.variance = float(variance)

    def __call__(self, x, y=None, **kwargs):
        if len(x.shape) == 3:
            if y is not None:
                assert x.shape[0] == y.shape[0], "batch dimension is inconsistent"
                return torch.stack([self(x[i], y=y[i]) for i in range(x.shape[0])])
            else:
                return torch.stack([self(x[i]) for i in range(x.shape[0])])
        if y is None:
            dists = torch.pdist(x / self.length, p=self.order)
            K = torch.exp(-.5 * dists)
            K = squareform(K)
            K.fill_diagonal_(1.0+self.variance)
        else:
            dists = torch.cdist(x / self.length, y / self.length, p=self.order)
            K = torch.exp(-.5 * dists)
        return K


class GaussianProcess(Process):
    continuous_time = True
    def __init__(self, x=None, y=None, batch_shape=None, event_shape=None, sequence_shape=None,
                 kernel=RBF(length=0.1), validate_args=None, device=None):

        assert issubclass(type(kernel), Kernel)
        self.kernel = kernel
        if (x is None) and (y is None):
            self.loc = torch.zeros(batch_shape, event_shape, device=device)
            self.covariance_matrix = torch.eye(event_shape, device=device)[np.newaxis].repeat(batch_shape, 1, 1)
            super(GaussianProcess, self).__init__(batch_shape, event_shape, sequence_shape)
        else:
            assert (x is not None) and (y is not None), "GaussianProcess needs x and y to fit at initialization"
            if x.ndimension() == 1:
                batch_shape = y.shape[0]; event_shape = y.shape[1]; sequence_shape = x.shape[0]
                x = torch.repeat_interleave(x.unsqueeze(0), batch_shape, dim=0)
            else:
                assert x.shape[0] == y.shape[0], "batch dimension x: %s and y: %s inconsistant"%(x.shape, y.shape)
                batch_shape = x.shape[0]; event_shape = y.shape[-1]; sequence_shape = x.shape[1]
            super(GaussianProcess, self).__init__(batch_shape, event_shape, sequence_shape)
            self.fit(x, y)

    def fit(self, x, y):
        self._k_in = self.kernel(x)
        self._L = torch.cholesky(self._k_in)
        self._x = x
        self._y = y
        #alpha = torch.cholesky_solve(x, L)

    def predict(self, x_pred, kernel_args={}):
        k_pred = self.kernel(self._x, x_pred)
        alpha = torch.cholesky_solve(self._y, self._L)

        y_pred = torch.bmm(k_pred.transpose(1, 2), alpha)
        # getting covariances
        t_fpred = self.kernel(x_pred)
        v = torch.bmm(self._L.inverse(), k_pred)
        variance_pred = t_fpred - torch.bmm(v.transpose(1, 2), v)
        return SingularMultivariateNormal(y_pred, variance_pred)

    def rsample(self, x_pred=None, kernel_args={}, sample_shape=torch.Size()):
        if x_pred is None:
            dd = SingularMultivariateNormal(self.loc, self.covariance_matrix)
        else:
            dd = self.predict(x_pred, kernel_args=kernel_args)
        return dd.rsample(sample_shape=sample_shape)


class MultivariateGaussianProcess(GaussianProcess):
    continuous_time = True

    def __init__(self, x=None, y=None, batch_shape=None, event_shape=None, sequence_shape=None, kernel=RBF(length=0.1),
                 pcov_matrix=None, validate_args=None):
        super().__init__(x, y, batch_shape, event_shape, sequence_shape, kernel=kernel)
        self.pcov_matrix = pcov_matrix

    def predict(self, x_pred, kernel_args={}, pcov_matrix=None, return_full=False, sort=True):
        pcov_matrix = pcov_matrix if pcov_matrix is not None else self.get_default_pcov(self.batch_shape, device=x_pred.device)
        if x_pred.ndimension() == 1:
            x_pred = torch.repeat_interleave(x_pred.unsqueeze(0), self.batch_shape, dim=0)

        if return_full:
            x_pred = torch.cat([self._x, x_pred], dim=1)
            if sort:
                indices = torch.argsort(x_pred, dim=1)[0]
            x_pred = x_pred[:, indices.squeeze(-1)]
        k_pred = self.kernel(self._x, x_pred)
        alpha = torch.cholesky_solve(self._y, self._L)

        y_pred = torch.bmm(k_pred.transpose(1, 2), alpha)
        # getting covariances
        t_fpred = self.kernel(x_pred, x_pred)
        v = torch.bmm(self._L.inverse(), k_pred)
        variance_pred = t_fpred - torch.bmm(v.transpose(1, 2), v)
        pred_dist = MatrixVariateNormalDistribution(y_pred, covariance_matrix=(variance_pred, pcov_matrix))
        return pred_dist


    def rsample(self, x_pred=None, kernel_args={}, sample_shape=torch.Size(), pcov_matrix=None):
        pcov_matrix = pcov_matrix or self.get_default_pcov(self.batch_shape)
        if x_pred is None:
            dd = MatrixVariateNormalDistribution(self.loc, (self.covariance_matrix, pcov_matrix))
        else:
            dd = self.predict(x_pred, kernel_args=kernel_args, pcov_matrix=pcov_matrix)
        return dd.rsample(sample_shape=sample_shape)

    def get_default_pcov(self, batch_shape, device=None):
        omega = torch.eye(self.event_shape, device=device)
        for i in range(0, self.event_shape):
            for j in range(i + 1, self.event_shape):
                omega[i, j] = 0.89
        return torch.mm(omega.t(), omega)[np.newaxis].repeat(batch_shape, 1, 1)

