import torch, pdb
from . import utils
from .. import distributions as dist
from ..data import OneHot
from ..distributions.distribution_priors import get_default_distribution
from . import reduce, Criterion, CriterionContainer, KLD, LogDensity, NaNError, MultiDivergence, Entropy
from ..utils import checklist, apply, print_stats


def scale_prior(prior, size):
    #TODO has to be implemented to prevent unscaled priors for KLD
    return prior

def regularize_logdets(logdets):
    if logdets is None:
        return 0
    if len(logdets[0].shape) == 1:
        logdets = torch.cat([l.unsqueeze(1) for l in logdets], dim = 1)
    elif len(logdets[0].shape)>=2:
        logdets = torch.cat(logdets, dim=-1)
    return logdets

def check_loss(loss, reduction=None):
    if issubclass(type(loss), type):
        loss = loss(reduction=reduction)
    return loss


class ELBO(CriterionContainer):
    reconstruction_class = LogDensity
    regularization_class = KLD

    def __repr__(self):
        return "ELBO(reconstruction=%s, regularization=%s, beta=%s, warmup=%s)"%(self.reconstruction_loss, self.regularization_loss, self.beta, self.warmup)

    def __add__(self, other):
        return Criterion.__add__(self, other)

    def __init__(self, warmup=100, beta=1.0, warmup_exp=1, reconstruction_loss=None, regularization_loss=None, reduction=None, *args, **kwargs):
        # losses
        reconstruction_loss = reconstruction_loss if reconstruction_loss is not None else self.reconstruction_class
        regularization_loss = regularization_loss if regularization_loss is not None else self.regularization_class
        criterions = []
        if issubclass(type(reconstruction_loss), list):
            reconstruction_loss = CriterionContainer(criterions=[check_loss(l, reduction=reduction) for l in reconstruction_loss])
            criterions.append(reconstruction_loss)
        else:
            reconstruction_loss = check_loss(reconstruction_loss, reduction=reduction)
            criterions.append(reconstruction_loss)
        if issubclass(type(regularization_loss), list):
            regularization_loss = MultiDivergence([check_loss(l, reduction=reduction) for l in regularization_loss])
            criterions.extend(regularization_loss)
        else:
            regularization_loss = check_loss(regularization_loss, reduction=reduction)
            criterions.append(regularization_loss)
        super(ELBO, self).__init__(criterions=criterions, reduction=reduction, **kwargs)
        self.reconstruction_loss = reconstruction_loss
        self.regularization_loss = regularization_loss
        # parameters
        self.warmup = warmup
        self.beta = beta
        self.warmup_exp = max(warmup_exp,0)


    def get_warmup_factors(self, latent_params, epoch=None, beta=None, warmup=None):
        # get warmup & beta sub-parameters
        warmup = warmup or self.warmup
        if issubclass(type(warmup), list):
            assert len(warmup) >= len(latent_params)
        else:
            warmup = [warmup]*len(latent_params)
        beta = beta or self.beta
        if issubclass(type(beta), list):
            assert len(beta) >= len(latent_params)
        else:
            beta = [beta]*len(latent_params)
        scaled_factors = [min(((epoch+1)/(warmup[i]-1))**self.warmup_exp, 1.0)*beta[i] if warmup[i] != 0 and epoch is not None else beta[i] for i in range(len(latent_params))]
        return scaled_factors

    def get_reconstruction_params(self, model, out, target, epoch=None, callback=None, **kwargs):
        if issubclass(type(self.reconstruction_loss), list):
            callback = callback or self.reconstruction_loss

            target = model.format_input_data(target)
            input_params = checklist(model.input_params); target = checklist(target)
            rec_params = []
            x_params = checklist(out['x_params'])
            for i, ip in enumerate(input_params):
                rec_params.append((callback[i], {'params1': x_params[i], 'params2': target[i], 'input_params': ip, 'epoch':epoch}, 1.0))
        else:
            callback = callback or self.reconstruction_loss
            x_params = out['x_params']
            rec_params = [(callback, {'params1': x_params, 'params2': target, 'input_params': model.input_params, 'epoch':epoch}, 1.0)]
        return rec_params

    def get_regularization_params(self, model, out, epoch=None, beta=None, warmup=None, callback=None, y=None, **kwargs):

        def parse_layer(latent_params, out, layer_index=0, y=None):
            if issubclass(type(latent_params), list):
                return [parse_layer(latent_params[i], utils.get_latent_out(out,i), layer_index=layer_index, y=y) for i in range(len(latent_params))]
            # encoder parameter
            params1 = out.get("z_params_enc")
            replaced_callback = None
            if out.get("z_preflow_enc") is not None:
                out1 = out['z_preflow_enc']
            else:
                out1 = out['z_enc']
            # decoder parameters
            prior = latent_params.get('prior') or None
            y = y or {}
            if latent_params.get('task') in y.keys():
                    # supervised case
                    replaced_callback = LogDensity(reduction=self.reduction)
                    params2 = y[latent_params['task']].to(out1.device)
                    if latent_params.get('dist') in [dist.Multinomial]:
                        params2 = OneHot(classes=latent_params['dim'])(params2)
                        if params2.ndim == params1.probs.ndim - 1:
                            params2 = params2.unsqueeze(-2).expand_as(params1.probs)
                    elif latent_params.get('dist') in [dist.Categorical]:
                        if params2.ndim == params1.probs.ndim - 2:
                            params2 = params2.unsqueeze(-1).expand_as(params1.probs)
                    else:
                        raise NotImplementedError(
                            "semi-supervision for task %s not implemented" % (latent_params['dist']))
                    out2 = None
            elif out.get('z_params_dec') is not None:
                params2 = out['z_params_dec']
                out2 = out["z_dec"]
            elif prior is not None:
                params2 = scale_prior(prior, out['z_enc'])(batch_size = params1.shape, device=out1.device, **kwargs)
                out2 = params2.rsample()
            else:
                prior_shape = out['z_enc'].shape
                params2 = get_default_distribution(latent_params['dist'], prior_shape, device=out['z_enc'].device)
                out2 = params2.rsample()

            #pdb.set_trace()
            return {"params1":params1, "params2":params2, "out1":out1, "out2":out2, 'callback':replaced_callback}

        # retrieve regularization parameters
        latent_params = checklist(model.latent_params)
        beta = beta or self.beta; beta = checklist(beta, n=len(latent_params))
        warmup = warmup or self.warmup
        factors = self.get_warmup_factors(latent_params, epoch=epoch, beta=beta, warmup=warmup)

        # parse layers
        reg_params = [];  
        regularization_loss = self.regularization_loss
        if not issubclass(type(regularization_loss), MultiDivergence):
           regularization_loss = checklist(regularization_loss, n=len(latent_params))
        for i, ip in enumerate(latent_params):
            parsed_layer = parse_layer(ip, utils.get_latent_out(out, i), y=y)
            if issubclass(type(ip), (list, MultiDivergence)):
                rp = []
                for pl in parsed_layer:
                    if pl.get('callback') is None:
                        rl = regularization_loss[i]
                    else:
                        rl = pl['callback']
                    rp.append((rl, pl, factors[i]))
                reg_params.append(rp)
            else:
                callback = parsed_layer.get('callback') or regularization_loss[i]
                reg_params.append((callback, parsed_layer, factors[i], (i,)))

        return reg_params

    def loss(self, model = None, out = None, target = None, epoch = None, beta=None, warmup=None, y=None, *args, **kwargs):
        assert not model is None and not out is None and not target is None, "ELBO loss needs a model, an output and an input"
        # parse loss arguments
        reconstruction_params = self.get_reconstruction_params(model, out, target, y=y, epoch=epoch, **kwargs)
        beta = beta or self.beta
        regularization_params = self.get_regularization_params(model, out, epoch=epoch, beta=beta, warmup=warmup, y=y, **kwargs)
        full_loss = 0; rec_errors=tuple(); reg_errors=tuple()
        # get reconstruction error
        for i, rec_args in enumerate(reconstruction_params):
            rec_loss, rec_losses = rec_args[0](**rec_args[1], reduction=self.reduction, is_sequence=model.take_sequences,**kwargs)
            full_loss = full_loss + rec_args[2]*rec_loss if rec_args[2] != 0. else full_loss
            rec_errors = rec_errors + rec_losses
        # get latent regularization error
        for i, reg_args in enumerate(regularization_params):
            if isinstance(reg_args, list):
                reg_losses_tmp = []
                for rg in reg_args:
                    reg_loss, reg_losses = rg[0](**rg[1], reduction=self.reduction, is_sequence=model.take_sequences, **kwargs)
                    full_loss = full_loss + rg[2]*reg_loss if rg[2] != 0. else full_loss
                    reg_losses_tmp.append(reg_losses)
                reg_errors = reg_errors + (sum(reg_losses_tmp, tuple()),)
            else:
                reg_loss, reg_losses = reg_args[0](**reg_args[1], reduction=self.reduction, is_sequence=model.take_sequences,
                                             **kwargs)
                full_loss = full_loss + reg_args[2] * reg_loss if reg_args[2] != 0. else full_loss
                reg_errors = reg_errors + reg_losses
        return full_loss, (rec_errors, reg_errors)
             
    def get_named_losses(self, losses):
        named_losses = {}
        if issubclass(type(self.reconstruction_loss), list):
            rec_losses = {}
            for i, rec_loss in enumerate(losses[0]):
                l = self.reconstruction_loss[i].get_named_losses(rec_loss)
                l = {k+'_%d'%i:v for k, v in l.items()}
                rec_losses = {**l, **rec_losses}
        else:
            named_losses = {**self.reconstruction_loss.get_named_losses(losses[0]), **named_losses}

        if issubclass(type(self.regularization_loss), list):
            reg_losses = {}
            for i, reg_loss in enumerate(losses[1]):
                if isinstance(reg_loss, list):
                    for j, rl in enumerate(losses[1][i]):
                        l = self.regularization_loss[i].get_named_losses(rl)
                        l = {k+'_%d'%j:v for k, v in l.items()}
                        reg_losses = {**l, **reg_losses}
                else:
                    named_loss = self.regularization_loss[i].get_named_losses(reg_loss)
                    reg_losses[named_loss+'_%d'%i] = named_loss
            named_losses = {**reg_losses, **named_losses}
        else:
            named_losses = {**named_losses, **self.regularization_loss.get_named_losses(losses[1])}
        return named_losses


class SemiSupervisedELBO(ELBO):

    def __repr__(self):
        return "SemiSupervisedELBO(semi_supervised_labels=%s)"%self.semi_supervised_labels

    def __init__(self, *args, semi_supervised_labels=None, **kwargs):
        kwargs['reduction'] = "batch_sum"
        super(SemiSupervisedELBO, self).__init__(*args, **kwargs)
        self.semi_supervised_labels = semi_supervised_labels

    def get_inference_distribs(self, model, out):
        distribs = {}; dims = {}
        for i ,latent_layer in enumerate(model.platent):
            if isinstance(latent_layer, list):
                for j, pl in enumerate(latent_layer):
                    if pl.get('task'):
                        distribs[pl['task']] = out['z_params_enc'][i][j]
                        dims[pl['task']] = pl['dim']
            else:
                if latent_layer.get('task'):
                    distribs[latent_layer['task']] = out['z_params_enc'][i]
                    dims[latent_layer]['task'] = latent_layer['dim']
        return distribs, dims

    def semi_supervised_loss(self, *args, model = None, out = None, y=None, **kwargs):
        assert y is not None and model is not None and out is not None, "mandatory keywords : model, out, y"
        inference_distribs, dims = self.get_inference_distribs(model, out)
        loss, losses = super(SemiSupervisedELBO, self).loss(*args, model=model, out=out, y=y, **kwargs)
        log_probs = []
        for k, v in inference_distribs.items():
            y_tmp = y.get(k).to(v.probs.device)
            if isinstance(v, dist.Multinomial):
                y_tmp = OneHot(classes=dims[k])(y_tmp)
                if y_tmp.ndim == v.probs.ndim - 1:
                    y_tmp = y_tmp.unsqueeze(-2).expand_as(v.probs)
            elif isinstance(v, dist.Categorical):
                if y_tmp.ndim == v.probs.ndim - 2:
                    y_tmp = y_tmp.unsqueeze(-1).expand_as(v.probs)

            log_prob = v.log_prob(y_tmp).exp()
            if log_prob.ndim == 2:
                log_prob = log_prob.mean(1)
            log_probs.append(log_prob)

        loss = torch.prod(torch.stack(log_probs, 1), 1) * loss
        entropies = sum([Entropy(reduction=self.reduction)(params1=dist)[0] for dist in inference_distribs.values()])

        loss = (loss + entropies).mean()
        losses = losses + (entropies.mean().item(), )
        return loss, losses

    def loss(self, *args, **kwargs):
        if self._supervised:
            loss, losses = super(SemiSupervisedELBO, self).loss(*args, **kwargs)
            loss = loss.mean()
            losses = losses + (0.,)
        else:
            loss, losses = self.semi_supervised_loss(**kwargs)
        return loss, losses

    def get_named_losses(self, losses):
        named_losses = super(SemiSupervisedELBO, self).get_named_losses(losses)
        if len(losses) > 2:
            named_losses = {**named_losses, 'total_entropy':losses[2]}
        return named_losses
                

