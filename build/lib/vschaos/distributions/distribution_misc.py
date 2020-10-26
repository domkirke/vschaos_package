import torch, numpy as np, pdb
from . import Distribution, Normal, MultivariateNormal, Multinomial
from torch.distributions import constraints, kl

# SingularMultivariateNormal allows to draw samples from a positive semi-definite matrix (not available with
#   vanilla torch module, as sampling requires Cholesky decomposition of covariance matrix)

class SingularMultivariateNormal(Distribution):
    has_rsample = True
    arg_constraints = {'loc':constraints.real_vector,
                       'covariance_matrix':constraints.real_vector}
    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.covariance_matrix

    @property
    def precision_matrix(self):
        if self._precision_matrix is None:
            self._precision_matrix = torch.inv(self.covariance_matrix)
        return self._precision_matrix

    def __init__(self, loc, covariance_matrix=None, lowrank = False, validate_args=None):
        self.loc = loc
        if covariance_matrix is not None:
            self.covariance_matrix = covariance_matrix
        batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]
        self.lowrank = lowrank
        super(SingularMultivariateNormal, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def rsample(self, sample_shape=None, lowrank=False):
        cov_svd = self.covariance_matrix.svd()
        s_sqrt = torch.diag_embed(cov_svd.S.sqrt())
        a = torch.bmm(cov_svd.U, torch.bmm(s_sqrt, cov_svd.V))

        if sample_shape:
            samples = []
            for i in range(sample_shape):
                eps = torch.randn(*self.loc.shape)
                samples.append(self.loc + torch.bmm(a, eps))
            out = torch.stack(samples, dim=0)
        else:
            eps = torch.randn(*self.loc.shape)
            out = self.loc + torch.bmm(a, eps)
        return out

    def entropy(self):
        return torch.log(torch.det(2 * np.pi * np.e * self.covariance_matrix))

def kronecker(A, B):
    assert len(A.shape) == len(B.shape) == 3, "kroncker takes b x _ x _ matrices"
    requires_grad = A.requires_grad or B.requires_grad
    out = torch.zeros(A.shape[0], A.size(1)*B.size(1),  A.size(2)*B.size(2), requires_grad=requires_grad, device=A.device)
    for i in range(A.shape[0]):
        out[i] =  torch.einsum("ab,cd->acbd", A[i], B[i]).contiguous().view(A.size(1)*B.size(1),  A.size(2)*B.size(2))
    return out

@kl.register_kl(SingularMultivariateNormal, MultivariateNormal)
def kl_singular_multi(p, q):
    mu1 = p.mean; mu2 = q.mean;
    cov1 = p.covariance_matrix; cov2 = q.covariance_matrix
    return kl._kl_multivariatenormal_multivariatenormal(p, q)

@kl.register_kl(MultivariateNormal, SingularMultivariateNormal)
def kl_singular_multi(p, q):
    return kl._kl_multivariatenormal_multivariatenormal(p, q)

# Matrix Variate Normal Distribution (cf Multivariate Gaussian Processes)
class MatrixVariateNormalDistribution(Distribution):
    has_rsample = True
    arg_constraints = {'loc':constraints.real_vector}

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return kronecker(self.cov_matrix, self.pcov_matrix)
    #
    # @property
    # def precision_matrix(self):
    #     if self._precision_matrix is None:
    #         self._precision_matrix = torch.inv(self.covariance_matrix)
    #     return self._precision_matrix

    def __init__(self, loc, covariance_matrix=None, lowrank = False, validate_args=None):
        self.loc = loc
        self.cov_matrix, self.pcov_matrix = tuple(covariance_matrix)
        batch_shape, event_shape = self.loc.shape[0], self.loc.shape[1:]
        self.lowrank = lowrank
        super(MatrixVariateNormalDistribution, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size([]), lowrank=False):
        # eps = torch.randn(*self.loc.shape, 1)
        # cov_svd = self.covariance_matrix.svd()
        # s_sqrt = torch.diag_embed(cov_svd.S.sqrt())
        # a = torch.bmm(cov_svd.U, torch.bmm(s_sqrt, cov_svd.V))
        # out = self.loc + torch.bmm(a, eps)[..., 0]
        cov_svd = self.cov_matrix.svd()
        s_sqrt = torch.diag_embed(cov_svd.S.sqrt())
        a = torch.bmm(cov_svd.U, torch.bmm(s_sqrt, cov_svd.V))
        pcov_chol = torch.cholesky(self.pcov_matrix, upper=True)

        if sample_shape == torch.Size([]) or sample_shape is None:
            eps = torch.randn(self.batch_shape, self.cov_matrix.shape[1], self.pcov_matrix.shape[1])
            out = self.loc + torch.bmm(a, torch.bmm(eps, pcov_chol))
        else:
            samples = []
            for i in range(sample_shape):
                eps = torch.randn(self.batch_shape, self.cov_matrix.shape[1], self.pcov_matrix.shape[1])
                samples.append(self.loc + torch.bmm(a, torch.bmm(eps, pcov_chol)))
            out = torch.stack(samples, dim=0)
        return out

    def vectorize(self):
        vec_loc = self.loc.contiguous().view(self.loc.shape[0], self.loc.shape[1]*self.loc.shape[2])
        return MultivariateNormal(vec_loc, self.variance)

    # def entropy(self):
    #     return torch.log(torch.det(2 * np.pi * np.e * self.covariance_matrix))

def get_multi_from_normal(p):
    assert len(p.mean.shape) == 3, "get_multi_from_normal only works with 3-d normal distributions (shape, time, dim)"
    dim = p.mean.shape[2]
    target_shape = p.mean.shape[1]*p.mean.shape[2]
    new_mean = p.mean.contiguous().view(p.mean.shape[0], target_shape)
    cov_tri = torch.diag_embed(p.stddev**2)
    with torch.no_grad():
        new_cov = torch.zeros(p.mean.shape[0], target_shape, target_shape, device=new_mean.device, requires_grad=p.stddev.requires_grad)
    for i in range(cov_tri.shape[1]):
        new_cov[:, i*dim:(i+1)*dim, i*dim:(i+1)*dim] = cov_tri[:, i]
    return MultivariateNormal(new_mean, covariance_matrix=new_cov)

@kl.register_kl(MatrixVariateNormalDistribution, MatrixVariateNormalDistribution)
def kl_mv_mv(p, q):
    return kl.kl_divergence(p.vectorize(), q.vectorize())

@kl.register_kl(MatrixVariateNormalDistribution, Normal)
def kl_mv_normal(p, q):
    #pdb.set_trace()
    return kl.kl_divergence(p.vectorize(), get_multi_from_normal(q))
    # return kl.kl_divergence(p.vectorize(), p.vectorize())

@kl.register_kl(Normal, MatrixVariateNormalDistribution)
def kl_normal_mv(p, q):
    return kl.kl_divergence(get_multi_from_normal(p), q.vectorize())


