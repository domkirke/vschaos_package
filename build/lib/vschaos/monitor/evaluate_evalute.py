import torch, pdb, torch.nn as nn
from ..utils import merge_dicts, decudify
from .evaluate_baselines import PCABaseline, ICABaseline


baseline_hash={'PCA':PCABaseline, 'ICA':ICABaseline}

class Evaluation(nn.Module):
    def __init__(self, **kwargs):
        super(Evaluation, self).__init__()
        self.parse_params(**kwargs)
        self.output=None
        if kwargs.get('out'):
            self.output = kwargs.get('out')

    def parse_params(self, **kwargs):
        pass

    def forward_model(self, model, input, multi_decode=True, **kwargs):
        preprocessing = kwargs.get('preprocessing')
        preprocess = kwargs.get('preproccess')
        if preprocessing and preprocess:
            input = preprocessing(input)
        input = model.format_input_data(input)
        vae_out = model.forward(input, multi_decode=multi_decode, **kwargs)
        return vae_out

    def evaluate(self, outputs):
        pass

    def __call__(self, model, loader, baselines=[], multi_decode=True, *args, **kwargs):
        evals = []
        print('forwarding model...')
        with torch.no_grad():
            outs = []; xs=[]; ys=[];
            baseline_outs = []
            for x, y in loader:
                vae_out = self.forward_model(model, x, y=y, multi_decode=multi_decode, **kwargs)
                outs.append(decudify(vae_out))
                xs.append(decudify(x));  ys.append(decudify(y))
        if issubclass(type(outs[0]), list):
            outs = [merge_dicts([outs[i][j] for i in range(len(outs))]) for j in range(len(outs[0]))]
        else:
            outs = merge_dicts(outs)
        ys= merge_dicts(ys)
        xs = torch.cat(xs, dim=0)
        print(' evaluating...')
        if issubclass(type(outs), list):
            evaluation_out = {f'layer_{i}':self.evaluate(outs[i], target=xs, y=ys, model=model, **kwargs) for i in range(len(outs))}
        else:
            evaluation_out = self.evaluate(outs, target=xs, y=ys, model=model, **kwargs)
        evals.append(evaluation_out)

        # baselines
        baseline_outs = {}
        for baseline in baselines:
            print('baseline %s...'%baseline)
            baseline_module = baseline_hash[baseline](model.pinput, model.platent)
            baseline_out = baseline_module(xs)
            eval_baseline = self.evaluate(baseline_out, target=xs, y=ys, model=baseline_module)
            baseline_outs[baseline] = eval_baseline

        if self.output:
            torch.save(evals, self.output)
        if len(evals) > 1:
            evals = merge_dicts(evals)
        else:
            evals = evals[0]
        evals['baselines'] = baseline_outs
        return evals

    def __add__(self, c):
        if not issubclass(type(c), Evaluation):
            return TypeError("only Evaluation subclasses can be added")
        if issubclass(type(self), EvaluationContainer):
            self._evaluations.append(c)
            return self
        else:
            container = EvaluationContainer()
            container.append(self); container.append(c)
            return container


class EvaluationContainer(Evaluation):
    def __repr__(self):
        return "Evaluation : (%s)"%self._evaluations
    def __init__(self, evaluations=[], **kwargs):
        self._evaluations = evaluations
        self.output = None
        if kwargs.get('out'):
            self.output = kwargs.get('out')

    def append(self, evaluation):
        self._evaluations.append(evaluation)

    def parse_params(self, **kwargs):
        [eval.parse_params(**kwargs) for eval in self._evaluations]

    def evaluate(self, outputs, target=None, model=None, **kwargs):
        out = {}
        for e in self._evaluations:
            out = {**out, **e.evaluate(outputs, target=target, model=model, **kwargs)}
        return out

def print_evaluation(eval, level=0):
    if issubclass(type(eval), dict):
        return ''.join(['-'*level+' '+k+'\n'+print_evaluation(v, level=level+1) for k, v in eval.items()])
    else:
        return '-'*level + ' ' + str(eval) + '\n'


