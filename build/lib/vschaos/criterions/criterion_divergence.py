from . import Criterion, reduce
import torch.distributions.kl as kl
import torch
from ..distributions.distribution_renyi import renyi_divergence
from ..distributions.distribution_js import js_divergence
from torch import clamp, exp
from numpy import cumprod

eps = 1e-7

def regularize_logdets(logdets):
    if logdets is None:
        return 0
    if logdets[0] is None:
        return 0
    if len(logdets[0].shape) == 1:
        logdets = torch.cat([l.unsqueeze(1) for l in logdets], dim = 1)
    elif len(logdets[0].shape)>=2:
        logdets = torch.cat(logdets, dim=-1)
    return logdets


class KLD(Criterion):
    def __init__(self, *args, **kwargs):
        super(KLD, self).__init__(*args, **kwargs)

    def __repr__(self):
        return "KLD(reduction=%s)"%self.reduction

    def loss(self, params1=None, params2=None, out1=None, out2=None, sample=False, compute_logdets=False, **kwargs):
        sample = (params1 is None) or (params2 is None) or sample
        if not sample:
            try:
                loss = clamp(kl.kl_divergence(params1, params2), min=eps)
            except NotImplementedError:
                loss = self.kl_sampled(params1, params2, out1, out2, **kwargs)
                pass
        else:
            loss = self.kl_sampled(params1, params2, out1, out2, **kwargs)

        loss = reduce(loss, self.reduction);
        if loss.ndim > 0:
            losses = (float(loss.mean().cpu().detach().numpy()),)
        else:
            losses = (float(loss.cpu().detach().numpy()),)

        return loss, losses

    def kl_sampled(self, params1, params2, out1, out2, **kwargs):
        if out1 is None:
            out1 = params1.rsample()
        return params1.log_prob(out1) - params2.log_prob(out1)

    def get_named_losses(self, losses):
        named_losses = {}
        if isinstance(losses, tuple):
            for i, l in enumerate(losses):
                named_losses['kld_%d'%i] = l
        else:
            named_losses['kld'] = losses
        return named_losses
        # return {names[i]:losses[i] for i in range(len(losses))}

class ReverseKLD(KLD):

    def __repr__(self):
        return "ReverseKLD(reduction=%s)"%self.reduction

    def loss(self, params1=None, params2=None, sample=False, **kwargs):
        assert params1, params2
        return super().loss(params2, params1, sample, **kwargs)


    def get_named_losses(self, losses):
        named_losses = {}
        if isinstance(losses, tuple):
            for i, l in enumerate(losses):
                named_losses['reverse_kld_%d'%i] = l
        else:
            named_losses['reverse_kld'] = losses
        return named_losses

def l2_kernel(x, y, reduction=None):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    return reduce(exp(-(x.unsqueeze(1).expand(x_size, y_size, dim) - y.unsqueeze(0).expand(x_size, y_size, dim)).pow(2) / float(dim)), reduction=reduction)  # (x_size, y_size)


# taken from https://github.com/napsternxg/pytorch-practice/
class MMD(Criterion):
    def __repr__(self):
        return "MMD(reduction=%s)"%self.reduction

    def __init__(self, kernel=l2_kernel, max_kernel=60, *args, **kwargs):
        super(MMD, self).__init__()
        self.kernel = kernel
        self.max_kernel = max_kernel

    def loss(self, params1=None, params2=None, sample=False, is_sequence=False, logdets=None, **kwargs):
        assert params1, params2
        if is_sequence:
            loss=sum([self.loss(params1[:,i], params2[:, i], sample=sample, is_sequence=False, **kwargs)[0] for i in range(params1.batch_shape[1])])
            losses = (float(loss),)
            return loss,(float(loss),)
        elif self.max_kernel:
            if params1.batch_shape[0] > self.max_kernel:
                d = params1.batch_shape[0] // self.max_kernel
                params1 = params1[:d*self.max_kernel].view(self.max_kernel, d, *params1.batch_shape[1:])
                params2 = params2[:d*self.max_kernel].view(self.max_kernel, d, *params2.batch_shape[1:])
                loss=sum([self.loss(params1[:,i], params2[:, i], sample=sample, is_sequence=False, **kwargs)[0] for i in range(params1.batch_shape[1])])
                losses = (float(loss),)
                return loss, losses

        sample1 = params1.sample() if not params1.has_rsample else params1.rsample()
        sample2 = params2.sample() if not params2.has_rsample else params2.rsample()
        #pdb.set_trace()
        if len(sample1.shape) > 2 + int(is_sequence):
            sample1 = sample1.contiguous().view(cumprod(sample1.shape[:-1])[-1], sample1.shape[-1])
            sample2 = sample2.contiguous().view(cumprod(sample2.shape[:-1])[-1], sample2.shape[-1])

        x_kernel = self.kernel(sample1, sample1, self.reduction)
        y_kernel = self.kernel(sample2, sample2, self.reduction)
        xy_kernel = self.kernel(sample1, sample2, self.reduction)
        loss = x_kernel + y_kernel - 2*xy_kernel;
        if loss.ndim > 0:
            losses = (float(loss.mean().cpu().detach().numpy()),)
        else:
            losses = (float(loss.cpu().detach().numpy()),)

        """
        if logdets is not None:
            logdet_error = reduce(regularize_logdets(logdets), self.reduction)
            loss = loss - logdet_error
            losses = losses + (float(logdet_error),)
        """
        return loss, losses

    def losses(self, *args, N=1, **kwargs):
        loss = 0; losses = []
        for n in range(N):
            loss_tmp, losses_tmp = self.get_mmd(*args, **kwargs)
            loss = loss + loss_tmp; losses_tmp.append(losses)
        return loss / N, (float(loss / N),)

    def get_named_losses(self, losses):
        named_losses = {}
        if isinstance(losses, tuple):
            for i, l in enumerate(losses):
                named_losses['mmd_%d'%i] = l
        else:
            named_losses['mmd'] = losses
        return named_losses


class RD(Criterion):

    def __repr__(self):
        return "RD(reduction=%s)"%self.reduction

    def __init__(self, alpha=2.0, learnable_alpha=False, *args, **kwargs):
        super(RD, self).__init__()
        assert alpha > 0
        self.alpha = torch.nn.Parameter(torch.tensor(alpha or 2.0), requires_grad=bool(learnable_alpha))

    def loss(self, params1=None, params2=None, **kwargs):
        assert params1, params2
        loss = reduce(renyi_divergence(params1, params2, self.alpha), reduction=self.reduction)
        if loss.ndim > 0:
            losses = (float(loss.cpu().detach().numpy()),)
        else:
            losses = (float(loss.mean().cpu().detach().numpy()),)
        return loss, losses


    def get_named_losses(self, losses):
        named_losses = {}
        if isinstance(losses, tuple):
            for i, l in enumerate(losses):
                named_losses['rd_%d'%i] = l
        else:
            named_losses['rd'] = losses
        return named_losses



class JSD(Criterion):
    def __repr__(self):
        return "JSD(reduction=%s)"%self.reduction

    def __init__(self, alpha=0.5, learnable_alpha=False, *args, **kwargs):
        super(JSD, self).__init__()
        assert alpha > 0 and alpha < 1
        self.alpha = torch.nn.Parameter(torch.tensor(alpha or 2.0), requires_grad=bool(learnable_alpha))

    def loss(self, params1=None, params2=None, **kwargs):
        assert params1, params2
        loss = reduce(js_divergence(params1, params2, self.alpha), reduction=self.reduction)
        losses = (float(loss),)
        if kwargs.get('logdets') is not None:
            logdet_error = reduce(regularize_logdets(kwargs['logdets']), self.reduction)
            loss = loss - logdet_error
            if loss.ndim > 0:
                losses = (float(loss.mean().cpu().detach().numpy()),)
            else:
                losses = (float(loss.cpu().detach().numpy()),)
        return loss, losses


    def get_named_losses(self, losses):
        named_losses = {}
        if isinstance(losses, tuple):
            for i, l in enumerate(losses):
                named_losses['jsd_%d'%i] = l
        else:
            named_losses['jsd'] = losses
        return named_losses



class MultiDivergence(Criterion):

    def __repr__(self):
        return "MultiDivergence(%s)"%''.join([str(d) for d in self._divergences])

    def __getitem__(self, i):
        return self._divergences[i]

    def __len__(self):
        return len(self._divergences)

    def __init__(self, divergences=[], *args):
        super(MultiDivergence, self).__init__()
        module_divergences = []
        for i,d in enumerate(divergences):
            if not issubclass(type(d), Criterion):
                div_args = {}
                if i < len(args):
                    div_args = args[i]
                d = d(**div_args)
            module_divergences.append(d)
        self._divergences = torch.nn.ModuleList(module_divergences)

    def get_named_losses(self, losses):
        loss_dict = {}
        for i in range(len(losses)):
            current_dict = {}
            if isinstance(losses[i], tuple):
                for j in range(len(losses[i])):
                    cd = self._divergences[i].get_named_losses(losses[i][j])
                    cd = {k+'_%d_%d'%(i,j):v for k,v in cd.items()}
                    current_dict = {**current_dict, **cd}
            else:
                nl = self._divergences[i].get_named_losses(losses[i])
                current_dict = {k+'_%d'%i:v for k,v in nl.items()}
            loss_dict = {**loss_dict, **current_dict}
        return loss_dict
            




"""
class Symmetrize(Criterion):
    def __init__(self, div):
        self.divergence = divergence

    def loss(self, params1, params2, **kwargs):
        loss1 = self.divergence(params1, params2)
        loss2 = self.divergence(params2
        return loss, losses

    def get_named_losses(self, losses):
        kld_name = 'jensen-shannon' if self._name is None else 'jensen-shannon_%s'%(self._name)
        named_losses = {kld_name: losses[0]}
        if len(losses) > 1:
            logdet_name = 'logdet' if self._name is None else 'logdet_%s'%(self._name)
            named_losses[logdet_name] = losses[1]
        return named_losses
"""



