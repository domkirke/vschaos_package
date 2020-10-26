import torch, torch.nn as nn

from . import Criterion
from .criterion_divergence import MMD
from ..utils.misc import crossed_select
from ..distributions import Bernoulli


class TotalCovariance(Criterion):
    MMDLimit = 400
    def __repr__(self):
        return "TotalCovariance (%s)"%self.method

    def __init__(self, latent_params=None, method='mmd', whitening=False):
        assert method in ['mmd', 'adversarial']
        super(TotalCovariance, self).__init__()
        self.method = method
        if self.method == "adversarial":
            assert latent_params
            self.discriminator, self.optimizer =  self.get_discriminator(latent_params)


    def get_discriminator(self, latent_params):
        disc = self.Sequential(nn.Linear(latent_params['dim'], latent_params['dim']), nn.ReLU(),
                               nn.Linear(latent_params['dim'], latent_params['dim']) ,nn.ReLU(),
                               nn.Linear(latent_params['dim'], 1), nn.Softmax(1))
        optimizer = torch.optim.Adam(disc.params(), lr=1e-4)
        return disc, optimizer

    def loss(self, params1, *args, N=1, **kwargs):
        if self.method == 'mmd':
            if params1.batch_shape[0] > self.MMDLimit:
                n_segments = params1.batch_shape[0]//self.MMDLimit
                div_out = []
                for i in range(n_segments):
                    print('%d / %d'%(i, n_segments))
                    div_out.append(float(self.loss(params1[self.MMDLimit*i:(i+1)*self.MMDLimit],
                        *args, N=N, **kwargs)[0]))
                div_out = sum(div_out)/len(div_out)
                return div_out, (float(div_out),)

            z_permuted = params1.scramble(dim=-1)
            return MMD()(params1, z_permuted, N=N)
        elif self.method == 'adversarial':
            z_dist = params1.sample()
            z_permuted = params1.scramble(dim=-1).sample()
            mask = Bernoulli(0.5).sample(z_dist.shape[0])
            classifier_in = crossed_select(mask, z_permuted, z_dist)
            classifier_out = self.discriminator(classifier_in)
            loss = nn.functional.binary_cross_entropy(classifier_out)
            return loss, (float(loss),)

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_named_losses(self, losses):
        named_losses = {}
        for i, l in enumerate(losses):
            kld_name = 'tc_%i'%i if self._name is None else 'tc_%i_%s'%(i, self._name)
            if issubclass(type(l), (list, tuple)):
                named_losses[kld_name]=l[0]
                if len(l) > 1:
                    logdet_name = 'logdet_%i'%i if self._name is None else 'logdet_%i_%s'%(i, self._name)
                    named_losses[logdet_name] = l[1]
            else:
                named_losses = {kld_name: l}
        return named_losses



