import torch
import pdb
from torch.nn.parallel._functions import Scatter, Gather
from .. import distributions as dist


def concat_distrib(distrib_list, target_device,  unsqueeze=True, dim=1):
    def concat_normal(distrib_list):
        means = [d.mean for d in distrib_list]
        stds = [d.stddev for d in distrib_list]
        if unsqueeze:
            means = [m.unsqueeze(dim) for m in means]
            stds = [s.unsqueeze(dim) for s in stds]
        if target_device != torch.device('cpu'):
            means = Gather.apply(target_device, dim, *tuple(means))
            stds = Gather.apply(target_device, dim, *tuple(stds))
        else:
            means = torch.cat(means, dim=dim)
            stds = torch.cat(stds, dim=dim)
        return type(distrib_list[0])(means, stds)

    def concat_categorical(distrib_list):
        probs = [d.probs for d in distrib_list]
        if unsqueeze:
            probs = [p.unsqueeze(dim) for p in probs]
        probs = Gather.apply(target_device, dim, *tuple(probs))
        #probs = [p.unsqueeze(dim) for p in probs]
        #probs = torch.cat(probs, dim)
        return type(distrib_list[0])(probs=probs)

    def concat_flow(distrib_list):
        return dist.ConcatenatedSequenceFlowDistribution(distrib_list)

    assert(len(set([type(d) for d in distrib_list])) == 1)
    if type(distrib_list[0]) in (dist.Normal, dist.RandomWalk):
        return concat_normal(distrib_list)
    elif type(distrib_list[0]) in (dist.Categorical, dist.Bernoulli):
        return concat_categorical(distrib_list)
    elif type(distrib_list[0]) in (dist.FlowDistribution, dist.SequenceFlowDistribution):
        return concat_flow(distrib_list)
    else:
        raise Exception('cannot concatenate distribution of type %s'%type(distrib_list[0]))


def gather(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device
      (-1 means the CPU).
    """
    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None
        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)(((k, gather_map([d[k] for d in outputs]))
                              for k in out))
        if isinstance(out, torch.distributions.Distribution):
            return concat_distrib(outputs, target_device, dim=0, unsqueeze=False)
        return type(out)(map(gather_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        return gather_map(outputs)
    finally:
        gather_map = None
