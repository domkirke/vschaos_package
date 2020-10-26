import torch, pdb
from torch.nn import functional as F
from numpy import pi
from . import Criterion, reduce
from .. import distributions as dist
from .. import utils


class LogDensity(Criterion):
    def __init__(self, *args, **kwargs):
        super(LogDensity, self).__init__(*args, **kwargs)
#        self.reduction = 'mean'

    def __repr__(self):
        return "LogDensity(reduction=%s)"%self.reduction

    def loss(self, params1=None, params2=None, *args, **kwargs):
        assert params1 is not None and params2 is not None
        if issubclass(type(params1),list):
            losses_full = tuple([self.loss(params1[i], params2[i])  for i in range(len(params1))])
            loss = sum([l[0] for l in losses_full])
            losses = tuple(sum([list(l[1]) for l in losses_full], []))
        else:
            #if len(target.shape)==2:
                #target = target.squeeze(1)
            #pdb.set_trace()
            if issubclass(type(params2), dist.Distribution):
                params2 = params2.rsample()
            loss = reduce(-params1.log_prob(params2), self.reduction or kwargs.get('reduction'))
            if loss.ndim > 0:
                losses = (float(loss.mean().cpu().detach().numpy()),)
            else:
                losses = (float(loss.cpu().detach().numpy()),)
        return loss, losses

    def get_named_losses(self, losses):
        if isinstance(losses, tuple):
            named_losses = {"log_density_%d"%i:losses[i] for i in range(len(losses))}
        else:
            named_losses = {"log_density":losses}
        return named_losses


class Entropy(Criterion):
    def loss(self, params1=None, *args, **kwargs):
        assert params1 is not None and isinstance(params1, dist.Distribution), "params1 keyword must be a Distribution"
        if issubclass(type(params1), list):
            losses_full = tuple([self.loss(params1[i]) for i in range(len(params1))])
            loss = sum([l[0] for l in losses_full])
            losses = tuple(sum([list(l[1]) for l in losses_full], []))
        else:
            loss = reduce(-params1.entropy().unsqueeze(1), self.reduction or self.kwargs('reduction'))
            if loss.ndim > 0:
                losses = (float(loss.mean().cpu().detach().numpy()),)
            else:
                losses = (float(loss.cpu().detach().numpy()),)
        return loss, losses



# adversarial loss
def get_adversarial_loss(d_fake, d_true, options={}):
#    print(torch.log(d_true), d_true)
    return -torch.mean(torch.log(d_true) + torch.log(1-d_fake))


# log-probabilities
def log_bernoulli(x, x_params, size_average=False):
    if torch.__version__ == '0.4.1':
        loss = F.binary_cross_entropy(x_params[0], x, reduction=size_average)
    else:
        loss = F.binary_cross_entropy(x_params[0], x, size_average=size_average)

    if not size_average:
        loss = loss / x.size(0)
    return loss
    #return F.binary_cross_entropy(x_params[0], x, size_average = False)

def log_normal(x, x_params, logvar=False, clamp=True, size_average=False):
    x = x.squeeze()
    if x_params == []:
        x_params = [torch.zeros_like(0, device=x.device), torch.zeros_like(0, device=x.device)]
    if len(x_params)<2:
        x_params.append(torch.full_like(x_params[0], 1e-3, device=x.device))
    mean, std = x_params
    if not logvar:
        std = std.log()
    # average probablities on batches
    #result = torch.mean(torch.sum(0.5*(std + (x-mean).pow(2).div(std.exp())+log(2*pi)), 1))
    loss = std + 0.5*(x-mean).pow(2).div(std.exp()) + torch.tensor(2*pi).sqrt().log()

    if size_average:
        loss = torch.mean(loss)
    else:
        loss = torch.mean(torch.sum(loss, 1))
    #result = F.mse_loss(x, x_params[0])
    return loss

def log_categorical(y, y_params, size_average=False):
    loss = F.nll_loss(y_params[0], utils.onehot.fromOneHot(y).long(), size_average=size_average)
    if not size_average:
        loss = loss / y.size(0)
    return loss

def log_density(in_dist):
    if in_dist in [dist.Bernoulli]:
        return log_bernoulli
#    elif in_dist.dist_class==dist.normal.dist_class or in_dist.dist_class==cust.spectral.dist_class:
    elif in_dist in [dist.Normal]:
        return log_normal
    elif in_dist in [dist.Categorical]:
        return log_categorical
    else:
        raise Exception("Cannot find a criterion for distribution %s"%in_dist)
