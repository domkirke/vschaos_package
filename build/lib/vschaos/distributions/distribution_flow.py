import torch, pdb, bisect
from . import Distribution, TransformedDistribution
from torch.distributions.utils import _sum_rightmost
from torch.distributions import kl
from numpy import cumsum
# from ..modules.flow import flow

def checktuple(item, n=1, rev=False):
    if not issubclass(type(item), tuple):
        item = (item,)*n
    if rev:
        item = list(reversed(item))
    return item

class FlowDistribution(TransformedDistribution):
    requires_preflow = True
    in_selector = lambda _, x: x
    def __init__(self, base_distribution, flow, unwrap_blocks=False, in_select=lambda x: x, div_mode="post", validate_args=None):
        super(FlowDistribution, self).__init__(in_select(base_distribution), flow.transforms, validate_args=validate_args)
        self.base_distribution=base_distribution
        self.flow = flow
        self.in_select = in_select
        self.unwrap_blocks=unwrap_blocks
        assert div_mode in ["pre", "post"], "div_mod keyword must be pre or post"
        self.div_mode = div_mode or "post"

    def __repr__(self):
        return "FlowDistribution(%s, %s)"%(self.base_dist, self.flow)

    def __getitem__(self, item):
        return type(self)(self.base_distribution.__getitem__(item), self.flow,
                                unwrap_blocks=self.unwrap_blocks, in_select = self.in_select, div_mode = self.div_mode)

    # def __getattr__(self, item):
    #     if hasattr(self, item):
    #         return super(FlowDistribution, self).__getattr__(item)
    #     else:
    #         return self.base_dist.__getattr__(item)

    def sample(self, sample_shape=torch.Size(), x_0=None, aux_in = None, n=None):
        with torch.no_grad():
            if x_0 is None:
                x_0 = self.base_distribution.sample(sample_shape)
            x = self.in_selector(x_0)
            if self.unwrap_blocks:
                full_x = []
            self.flow.amortization(x_0, aux=aux_in)
            for i, flow in enumerate(self.flow.blocks):
                x = flow(x)
                if self.unwrap_blocks:
                    full_x.append(x.unsqueeze(1))
            if self.unwrap_blocks:
                return torch.cat(full_x, dim=1), x_0
            else:
                return x, x_0

    def rsample(self, sample_shape=torch.Size(), x_0=None, aux_in=None, n=None):
        if x_0 is None:
            x_0 = self.base_distribution.rsample(sample_shape)
        if issubclass(type(x_0), tuple):
            x_in = x_0[0]
        else:
            x_in = x_0
        x = self.in_selector(x_in)
        if self.unwrap_blocks:
            full_x = []
        self.flow.amortization(x_0, aux=aux_in)
        for i, flow in enumerate(self.flow.blocks):
            #pdb.set_trace()
            x = flow(x)
            if self.unwrap_blocks:
                full_x.append(x.unsqueeze(1))
        if self.unwrap_blocks:
            return torch.cat(full_x, dim=1), x_in
        else:
            return x, x_in


    def log_prob(self, value):
        """
        Scores the sample by inverting the transform(s) and computing the score
        using the score of the base distribution and the log abs det jacobian.
        """
        log_prob = self.base_dist.log_prob(value) - self.flow.bijectors.log_abs_det_jacobian(value)
        return log_prob


class Flow(object):

    def __init__(self, dist_type, flow_module, div_mode="pre"):
        super(Flow, self).__init__()
        self._dist = dist_type
        self._flow = flow_module
        self.div_mode = div_mode

    @property
    def dist(self):
        return self._dist

    @property
    def flow(self):
        return self._flow

    @property
    def has_rsample(self):
        return self._dist.has_rsample

    def __call__(self, out, *args, **kwargs):
        return FlowDistribution(out, self._flow, div_mode=self.div_mode, **kwargs)


class SequenceFlowDistribution(FlowDistribution):
    in_selector = lambda _, x: x[:, -1]

    def __getitem__(self, item):
        item = checktuple(item)
        dist = self
        for i, it in enumerate(item):
            if i == 0:
                dist = super(SequenceFlowDistribution, self).__getitem__(it)
            if i == 1:
                assert it.step is None, "indexing among sequence dimension has to be left continuous, got %s" % it
                end = it.stop
                if end < 0:
                    end = self.batch_shape[1] + end
                if end <= self.base_dist.batch_shape[1]:
                    dist = self.base_dist[:, :end]
                else:
                    n_flow = end - self.base_dist.batch_shape[1]
                    flows = self.flow[:n_flow]
                    dist = SequenceFlowDistribution(self.base_dist, flows, unwrap_blocks=self.unwrap_blocks,
                                                    in_select=self.in_select, div_mode = self.div_mode)
        return dist

    def __repr__(self):
        return "SequenceFlowDistribution(%s, %s)"%(self.base_dist, self.flow)
    
    @property
    def batch_shape(self):
        final_seq_length = self.base_distribution.batch_shape[1] + len(self.flow.bijectors)
        return (self.base_distribution.batch_shape[0], final_seq_length, *self.base_distribution.batch_shape[2:])

    def sample(self, *args, **kwargs):
        x, x_0 = super(SequenceFlowDistribution, self).sample(*args, **kwargs)
        x =  torch.cat([x_0, x], dim=1)
        return x, x

    def rsample(self, *args, **kwargs):
        x, x_0 = super(SequenceFlowDistribution, self).rsample(*args, **kwargs)
        x =  torch.cat([x_0, x], dim=1)
        return x, x

    def log_prob(self, value):
        """
        Scores the sample by inverting the transform(s) and computing the score
        using the score of the base distribution and the log abs det jacobian.
        """
        n_seq = self.base_distribution.batch_shape[1]
        log_prob_dist = self.base_distribution.log_prob(value[:, :n_seq])
        #log_prob_dist = self.base_distribution.log_prob(value)
        log_prob_flow = self.flow.bijectors.log_abs_det_jacobian(value[:, n_seq], matrix=True)
        if len(log_prob_flow.shape) < 3 or log_prob_flow.shape[-1] == 1:
            log_prob_flow = log_prob_flow.unsqueeze(-1) if len(log_prob_flow.shape) == 2 else log_prob_flow
            # trick to fit shape of log jacobians
            log_prob_flow = log_prob_flow.repeat(1, 1, value.shape[-1]) / value.shape[-1]
        # cumulative sum and adding to last step from base distribution
        log_prob_flow = - torch.cumsum(log_prob_flow, dim=1) + log_prob_dist[:, -1].unsqueeze(1)
        return torch.cat([log_prob_dist, log_prob_flow], dim=1)

class ConcatenatedSequenceFlowDistribution(SequenceFlowDistribution):
    def __init__(self, flows, x_0=None):
        self._flows = flows
        self.x_0 = x_0

    def __repr__(self):
        return "ConcatenatedSequenceFlowDistribution(len=%s)"%(len(self._flows))

    @property
    def batch_shape(self):
        seq_shape = int(cumsum([fl.batch_shape[1] for fl in self._flows])[-1])
        return torch.Size((self._flows[0].batch_shape[0], seq_shape))

    @property
    def event_shape(self):
        return torch.Size((*self._flows[-1].batch_shape[2:],))

    def __getitem__(self, item):
        item = checktuple(item)
        flows = list(self._flows); x_0 = self.x_0
        for i, it in enumerate(item):
            assert type(it) is slice, "Indexing ConcatenatedSequenceFlowDistribution only has slices"
            if i==0:
                flows = [fl[it] for fl in self._flows]
                if x_0 is not None:
                    x_0 = [x.__getitem__(it) for x in x_0]
            elif i==1:
                assert it.start is None and it.step is None, "indexing among sequence dimension has to be left continuous, got %s"%it
                shapes = cumsum([fl.batch_shape[1] for fl in self._flows])
                end = it.stop
                if end < 0:
                    end = shapes[-1] + end
                flow_idx = bisect.bisect_left(shapes, end)
                flows = flows[:flow_idx] + [flows[flow_idx][:, :end - shapes[flow_idx-1]]]
                if x_0 is not None:
                    x_0 = x_0[:flow_idx]+[None]
            else:
                flows = [fl.__getitem__(slice(None), slice(None), *item[2:]) for fl in self._flows]
                x_0 = [x.__getitem__(slice(None), slice(None), *item[2:]) for x in x_0]
                break
        return ConcatenatedSequenceFlowDistribution(flows, x_0=x_0)



    def rsample(self, *args, x_0=None, **kwargs):
        outs = []; outs_preflow = [None]*len(self._flows)
        x_0 = x_0 or self.x_0
        if x_0 is not None:
            assert issubclass(type(x_0), list)
            assert len(x_0) == len(self._flows)
        else:
            x_0 = [None]*len(self._flows)
        for i, flow in enumerate(self._flows):
            if issubclass(type(flow), FlowDistribution):
                out, out_preflow = flow.rsample(x_0=x_0[i], **kwargs)
            else:
                out = flow.rsample()
                out_preflow = out
            if x_0[i] is None:
                outs_preflow[i] = out_preflow
            else:
                outs_preflow[i] = x_0[i]
            outs.append(out)
        out = torch.cat(outs, dim=1)
        return out, outs_preflow

    def sample(self, *args, x_0=None, **kwargs):
        outs = [];
        outs_preflow = [None] * len(self._flows)
        x_0 = x_0 or self.x_0
        if x_0 is not None:
            assert issubclass(type(x_0), list)
            assert len(x_0) == len(self._flows)
        else:
            x_0 = [None] * len(self._flows)
        for i, flow in enumerate(self._flows):
            out, out_preflow = flow.sample(x_0=x_0[i], **kwargs)
            if x_0[i] is None:
                outs_preflow[i] = out_preflow
            else:
                outs_preflow[i] = x_0[i]
            outs.append(out)
        out = torch.cat(outs, dim=1)
        return out, outs_preflow

    def log_prob(self, value):
        dims = [0] + cumsum([fl.batch_shape[1] for fl in self._flows]).tolist()
        log_prob = torch.zeros_like(value, device=value.device)
        for i, fl in enumerate(self._flows):
            if issubclass(type(fl), FlowDistribution):
                base_shape = fl.base_distribution.batch_shape[1]
            else:
                base_shape = fl.batch_shape[1]
            log_prob_in = value[:, dims[i]:dims[i+1]]
            log_prob[:, dims[i]:dims[i+1]] = fl.log_prob(log_prob_in)
        return log_prob






"""
@kl.register_kl(FlowDistribution, Distribution)
def kl_flow_dist(p, q):
    if p.div_mode=="pre":
        return kl.kl_divergence(p.base_dist, q)
    else:
        raise NotImplementedError

@kl.register_kl(Distribution, FlowDistribution)
def kl_dist_flow(p, q):
    if p.div_mode=="pre":
        return kl.kl_divergence(p, q.base_dist)
    else:
        raise NotImplementedError


@kl.register_kl(FlowDistribution, FlowDistribution)
def kl_dist_flow(p, q):
    if p.div_mode=="pre" and q.div_mode=="pre":
        return kl.kl_divergence(p.base_dist, q.base_dist)
    else:
        raise NotImplementedError
"""

    


