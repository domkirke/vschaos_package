import torch
from .criterion_criterion import Criterion, reduce
from .. import distributions as dist
from ..modules.modules_bottleneck import MLP
from ..modules.modules_distribution import CategoricalLayer1D

class InfoNCE(Criterion):

    def loss(self, model=None, out=None, out_negative=None, true_ids=None, epoch=None, n_preds=None, *args, **kwargs):
        # multi cpc?
        cpc_states = out.get('cpc_states_enc')
        if cpc_states is not None:
            cpc_states = cpc_states[0]
        else:
            return torch.tensor(0., requires_grad=False)
        n_preds = n_preds or out['z_enc'][-1].shape[1] - cpc_states.shape[1]
        z_real = out['z_enc'][-1][:, :-n_preds]
        z_predicted = out['z_enc'][-1][:,-n_preds:]
        density_ratio = model.prediction_module.get_density_ratio(z_predicted, z_real)

        cpcs = 0
        for i in range(density_ratio.shape[0]):
            cpcs = cpcs + torch.sum(torch.diag(torch.nn.LogSoftmax(dim=1)(density_ratio[i])))
        cpcs = - cpcs / (z_predicted.shape[0] * z_predicted.shape[1])

        return cpcs, (float(cpcs),)

    def get_named_losses(self, losses):
        return {'cpc': losses[0]}


class Classification(Criterion):
    def __repr__(self):
        return f"Classification(task={self.task}, mlp={self.mlp})"

    def __init__(self, pinput, task, plabel, layer=-1, hidden_params={'nlayers':2, 'dim':200}, optim_params={}, target_dims=None, **kwargs):
        super(Classification, self).__init__(**kwargs)
        self.task = task
        self.pinput = pinput
        self.plabel = plabel
        self.layer = layer
        self.phidden = hidden_params 
        self.target_dims = target_dims
        if target_dims is not None:
            pinput = dict(pinput)
            pinput['dim'] = len(target_dims)

        self.init_modules(pinput, self.phidden, plabel)
        self.init_optimizer(optim_params)

    def init_modules(self, pinput, phidden, plabel):
        self.mlp = MLP(pinput, phidden)
        self.out_module = CategoricalLayer1D({'dim': self.mlp.phidden['dim']}, plabel)

    def init_optimizer(self, optim_params={}):
        self.optim_params = optim_params
        alg = optim_params.get('optimizer', 'Adam')
        optim_args = optim_params.get('optim_args', {'lr':1e-3})
        optimizer = getattr(torch.optim, alg)([{'params':self.parameters()}], **optim_args)
        self.optimizer = optimizer

    def get_params(self, out, from_encoder=False, target_dims = None):
        if from_encoder:
            if isinstance(self.layer, tuple):
                z_out = out[self.layer[0]]['out'][self.layer[1]]
            else:
                z_out = out[self.layer]['out']
        else:
            if isinstance(self.layer, tuple):
                z_out = out['z_enc'][self.layer[0]][self.layer[1]]
            else:
                z_out = out['z_enc'][self.layer]
        if target_dims is not None:
            z_out = torch.index_select(z_out, -1, torch.Tensor(target_dims).to(z_out.device).long())
        return z_out

    def loss(self, model, out, y, from_encoder=False, **kwargs):
        if y.get(self.task) is None:
            print('{Warning} classification loss did not access metadata "%s"'%self.task)
            return 0., (0.)
        z_out = self.get_params(out, from_encoder=from_encoder, target_dims=self.target_dims)
        classif_out = self.out_module(self.mlp(z_out))
        label = y[self.task]
        if self.plabel['dist'] == dist.Categorical:
            if label.ndim == classif_out.probs.ndim - 2:
                label = label.unsqueeze(-1).repeat(1, classif_out.probs.shape[1])
        label = label.to(z_out.device)
        classif_loss = -reduce(classif_out.log_prob(label), self.reduction)
        return classif_loss, (classif_loss.item(),)

    def forward(self, out, from_encoder=False):
        z_out = self.get_params(out, from_encoder=from_encoder, target_dims=self.target_dims)
        classif_out = self.out_module(self.mlp(z_out))
        return classif_out

    def step(self, *args, retain=False, **kwargs):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_named_losses(self, losses):
        return {f'classif_{self.task}_{self.layer}':losses[0]}



    # def loss(self, out=None, **kwargs):
