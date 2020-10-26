import torch, torch.nn as nn, numpy as np
from .. import distributions as dist
import sklearn.decomposition as decomposition

from ..utils.misc import checklist

class DimRedBaseline(nn.Module):

    def __init__(self, input_params, latent_params, **kwargs):
        super(DimRedBaseline, self).__init__()
        self.pinput = input_params
        latent_params = checklist(latent_params)
        self.platent = checklist(latent_params[-1])
        self.dimred_module = self.dimred_class(n_components = self.platent[-1]['dim'], **kwargs)

    def encode(self, x, **kwargs):
        input_device = torch.device('cpu')
        if torch.is_tensor(x):
            input_device = x.device
        dimred_out = torch.from_numpy(self.dimred_module.fit_transform(x)).to(input_device).float()
        dimred_dist = dist.Normal(dimred_out, torch.zeros_like(dimred_out)+1e-12)
        return [{'out':dimred_out, 'out_params':dimred_dist}]

    def decode(self, z, squeezed=False, **kwargs):
        input_device = torch.device('cpu')
        if torch.is_tensor(z):
            input_device = z.device
            z = z.cpu().detach().numpy()
        dimred_out = torch.from_numpy(self.dimred_module.inverse_transform(z)).to(input_device).float()
        if squeezed:
            dimred_out = dimred_out.unsqueeze(1)
        dimred_dist = dist.Normal(dimred_out, torch.zeros_like(dimred_out)+1e-12)
        return [{'out':dimred_out, 'out_params':dimred_dist}]

    def forward(self, x, y=None, **kwargs):
        input_car = len(checklist(self.pinput['dim']))+self.pinput.get('conv')
        input_shape = x.shape[-input_car:]
        batch_shape = x.shape[:-input_car]
        x = x.view(np.cumprod(batch_shape)[-1], np.cumprod(input_shape)[-1])
        z = self.encode(x, **kwargs)
        reconstruction= self.decode(z[0]['out'], **kwargs)

        x_params = reconstruction[0]['out_params'].reshape(*batch_shape, *input_shape)
        z_params_enc = [z[0]['out_params'].reshape(*batch_shape, self.platent[0]['dim'])]
        z_enc = [z[0]['out'].reshape(*batch_shape, self.platent[0]['dim'])]
        return {'x_params':x_params, 'z_params_enc':z_params_enc, 'z_params_dec':[], 'z_enc':z_enc}


class PCABaseline(DimRedBaseline):
    dimred_class = decomposition.PCA

class ICABaseline(DimRedBaseline):
    dimred_class = decomposition.FastICA
