from numpy import ceil, cumprod
from torch import sigmoid, tensor, eye
from ..distributions import Bernoulli, Normal, Categorical, MultivariateNormal, RandomWalk, Empirical
from ..distributions.distribution_flow import Flow
import torch.nn as nn

from . import init_module, DeconvolutionalLatent
from .. import distributions as dist
from ..utils import checklist, checktuple

def EmpiricalLayer(pinput, poutput, **kwargs):
    take_conv = 0
    if issubclass(type(poutput['dim']), list):
        take_conv = len(poutput['dim']) != 1
    if poutput.get('conv'):
        take_conv = poutput['conv']
    if take_conv:
        raise NotImplementedError()
        #return GaussianLayer2D(pinput, poutput, **kwargs)
    else:
        return EmpiricalLayer1D(pinput, poutput, **kwargs)

class EmpiricalLayer1D(nn.Sequential):
    def __init__(self, pinput, poutput, **kwargs):
        self.input_dim = pinput['dim']; self.output_dim = poutput['dim']
        self.nn_lin = poutput.get('nn_lin', None)
        modules = (nn.Linear(self.input_dim, self.output_dim),)
        if self.nn_lin is not None:
            modules = modules+(getattr(nn, self.nn_lin)(),)
        super(EmpiricalLayer1D, self).__init__(*modules)


########################################################################
####        Gaussian layers

def GaussianLayer(pinput, poutput, **kwargs):
    """
    GaussianLayer is used to parametrize the parameters of Gaussian-related distributions.
    Affiliated distributions : dist.Normal, dist.MultivariateNormal, proc.RandomWalk

    If input data dimensionality is greater than 1, or the keyword *conv* is specified in input
    parameters, a convolutional module is used. Otherwise, the layer relies on simple Linear modules.

    :param dict pinput: input parameters (specify *conv* keyword to enforce a convolutional module)
    :param poutput: output parameters
    :return: Gaussian layer
    """
    take_conv = 0
    if poutput.get('channels') or poutput.get('conv'):
        take_conv = True
    if take_conv:
        return GaussianLayer2D(pinput, poutput, **kwargs)
    else:
        return GaussianLayer1D(pinput, poutput, **kwargs)


class GaussianLayer1D(nn.Module):

    requires_deconv_indices = False
    '''Module that outputs parameters of a Gaussian distribution.'''
    def __init__(self, pinput, poutput, **kwargs):
        """
        GaussianLayer1D is a layer using simple nn.Linear modules to encode mean and variance parameters of gaussian
        affiliated distributions.

        pinput additional parameters :
        * *nn_lin* specifies a specific non-linearity for mean output (has to be submodule of torch.nn)
        * *nn_lin_args* specifies additional argument for non-linearity module
        :param dict pinput: input parameters
        :param dict poutput: output parameters
        """

        nn.Module.__init__(self)
        self.pinput = pinput; self.poutput = poutput

        self.nn_lin = poutput.get('nn_lin')
        if not issubclass(type(self.nn_lin), nn.Module) and self.nn_lin is not None:
            self.nn_lin = getattr(nn, poutput.get('nn_lin'))(**poutput.get('nn_lin_args', {}))

        # init mean module
        #input_dim = sum(cumprod(checktuple(p['dim']))[-1] for p in checklist(pinput));
        input_dim = sum(checklist(p['dim'])[-1] for p in checklist(pinput));
        output_dim = cumprod(checktuple(poutput['dim']))[-1]
        self.mean_module = nn.Linear(input_dim, output_dim)
        nn.init.xavier_normal_(self.mean_module.weight)
        nn.init.zeros_(self.mean_module.bias)

        # init variance module
        self.var_module = nn.Linear(input_dim, output_dim)
        nn.init.xavier_normal_(self.var_module.weight)
        nn.init.zeros_(self.var_module.bias)
        #init_module(self.var_module, 'Sigmoid')

        #TODO sequence shit
        self.is_seq = kwargs.get('is_seq', False)
        
    #@flatten_seq_method
    def forward(self, ins,  *args, **kwargs):
        '''Outputs parameters of a diagonal normal distribution.
        :param ins : input vector.
        :returns: (torch.Tensor, torch.Tensor)'''

        # format
        #pdb.set_trace()
        n_batch = ins.shape[0]; n_seq = None
        if self.is_seq:
            n_seq = ins.shape[1]
            ins = ins.view(n_batch*n_seq, *ins.shape[2:])
        else:
            ins = ins.view(n_batch, *ins.shape[1:])
        # forward
        if self.nn_lin:
            mu = self.nn_lin(self.mean_module(ins))
        else:
            mu = self.mean_module(ins)
        std = self.var_module(ins)
        # reshape
        if self.is_seq:
            mu = mu.reshape(n_batch, n_seq, *mu.shape[1:])
            std = std.reshape(n_batch, n_seq, *std.shape[1:])

        mu  = mu.reshape(*tuple(mu.shape[:-1]), *checktuple(self.poutput['dim']))
        std  = std.reshape(*tuple(std.shape[:-1]), *checktuple(self.poutput['dim']))

        dist = self.poutput['dist']
        dist_type = dist
        if issubclass(type(dist), Flow):
            dist_type = dist.dist

        if dist_type in [Normal, RandomWalk]:
            return Normal(mu, sigmoid(std))
        elif dist_type == MultivariateNormal:
            #TODO full covariance parametrization
            return dist(mu, covariance_matrix=sigmoid(std).unsqueeze(1) * eye(mu.shape[1]))
        # elif dist_type == RandomWalk:
        #     return dist(mu, sigmoid(std))
        #return Normal(mu, sqrt(exp(logvar)))
    

class GaussianLayer2D(nn.Module):
    requires_deconv_indices = True

    def __init__(self, pinput, poutput, hidden_module=None, **kwargs):
        nn.Module.__init__(self)
        self.pinput = pinput; self.poutput = poutput
        # retrieve convolutional parameters from hidden module
        """
        if hidden_module:
            self.separate_heads = hidden_module.separate_heads
            self.record_heads = pinput.get('record_heads', True)
            phidden = {k: [v[-1]] if issubclass(type(v), list) else [v] for k, v in hidden_module.phidden.items()}
            phidden['channels'].append(poutput['channels'])
        """
        self.phidden = self.get_phidden(self.pinput, self.poutput)

        #TODO pooling specific arguments
        self.pool_target_size = None; output_padding = None
        self.requires_deconv_indices = False
        if hidden_module:
            if hidden_module.has_pooling:
                self.pool_target_size = hidden_module.pool_target_sizes[-1]
                self.requires_deconv_indices = True
            self.phidden['output_padding'] = checklist(hidden_module.phidden[0].get('output_padding'))[-1]

        # non-linearity
        self.nn_lin = poutput.get('nn_lin')
        self.nn_lin_args = poutput.get('nn_lin_args', {})
        if self.nn_lin:
            if issubclass(type(self.nn_lin), str):
                self.nn_lin = getattr(nn, self.nn_lin)(**self.nn_lin_args)

        # separate_vars : if multi-head, defines if each head owns its own variance or if one global variance is defined
        self.separate_vars = kwargs.get('separate_vars', True)
        self.mean_modules, self.std_modules, weights = self.init_modules(self.phidden, self.separate_vars)
        if weights is not None:
            self.weights = nn.ParameterList(weights)

        self.is_seq = kwargs.get('is_seq', False)

    def get_phidden(self, pinput, poutput):
        # make convolutional arguments from input and output parameters
        in_channels = checklist(pinput.get('channels', 1))[0]
        out_channels = checklist(poutput.get('channels', 1))[0]
        if pinput.get('kernel_size') is None:
            raise TypeError('keyword kernel_size must be defined in pinput')
        kernel_size = checklist(pinput.get('kernel_size'))[0]
        padding = checklist(pinput.get('padding', int(ceil(kernel_size/2))))[0]
        dilation = checklist(pinput.get('dilation', 1))[0]
        stride = checklist(pinput.get('stride', 1))[0]
        conv_dim = pinput.get('conv_dim', len(checktuple(poutput['dim'])))
        windowed = poutput.get('windowed', False)
        output_padding = pinput.get('output_padding', [None])[0]
        phidden = {'channels':[in_channels, out_channels],
                       'kernel_size':kernel_size, 'conv_dim':conv_dim,
                       'padding':padding, 'dilation':dilation, 'stride':stride,
                       'windowed':windowed, 'output_padding':output_padding}
        if pinput.get('heads'):
            phidden['heads'] = pinput['heads']
        if pinput.get('pool'):
            phidden['pool'] = pinput['pool'][0]
        return phidden

    def get_mean_module(self, phidden):
        # make mean module of the layer
        deconv_class = phidden.get('class', [DeconvolutionalLatent])
        deconv_layer = deconv_class[-1].conv_layer
        mean_module = deconv_layer(phidden['channels'][0], phidden['channels'][1], phidden['conv_dim'],
                                       phidden['kernel_size'],
                                       pool=phidden.get('pool'),
                                       dilation=phidden['dilation'],
                                       dropout=None, padding=phidden['padding'],
                                       stride=phidden['stride'], windowed=phidden['windowed'],
                                       batch_norm=None, nn_lin=None, output_padding=phidden['output_padding'])
        nn.init.xavier_normal_(mean_module._modules['conv_module'].weight)
        return mean_module

    def get_variance_module(self, phidden):
        # make variance module of the layer
        deconv_class = phidden.get('class', [DeconvolutionalLatent])
        deconv_layer = deconv_class[-1].conv_layer
        std_module = deconv_layer(phidden['channels'][0], 1, phidden['conv_dim'],
                                      phidden['kernel_size'],
                                      pool=phidden.get('pool'), 
                                      dilation=phidden['dilation'],
                                      dropout=None, padding=phidden['padding'],
                                      stride=phidden['stride'],
                                      batch_norm=None, nn_lin=None, output_padding=phidden['output_padding'])
        nn.init.xavier_normal_(std_module._modules['conv_module'].weight)
        return std_module

    def init_modules(self, phidden, separate_vars = False):
        if phidden.get('heads'):
            mean_modules = nn.ModuleList([self.get_mean_module(phidden) for i in range(phidden['heads'])])
            if not separate_vars:
                # why? there should be just one module?
                #std_params = dict(phidden); std_params['channels'][0] = phidden['channels'][0] * phidden['heads'][0]
                var_modules = self.get_variance_module(phidden)
            else:
                var_modules = nn.ModuleList([self.get_variance_module(phidden) for i in range(phidden['heads'])])
            weights = [nn.Parameter(tensor([1.])) for i in range(phidden['heads'])]
        else:
            mean_modules = self.get_mean_module(phidden)
            var_modules = self.get_variance_module(phidden)
            weights = None
        return mean_modules, var_modules, weights


    #@flatten_seq_method
    def forward(self, ins,  *args, indices=None, output_size=None, output_heads=False, **kwargs):
        n_batch = ins.shape[0]; n_seq = None

        #if not any channel dimension, add it
        if len(ins.shape) <= self.phidden['conv_dim']+1:
            ins = ins.unsqueeze(-self.phidden['conv_dim']-1)

        #TODO verify this sequence thing...!
        is_seq = self.is_seq
        if is_seq:
            n_seq = ins.shape[1]
            if not ins.is_contiguous():
                ins = ins.contiguous()
            ins = ins.view(ins.shape[0]*ins.shape[1], *ins.shape[2:])

        if issubclass(type(self.mean_modules), nn.ModuleList):
            mu_out = []
            for i in range(len(self.mean_modules)):
                mu_out.append(self.mean_modules[i](ins[:, i], indices=indices) * self.weights[i])
            if output_heads:
                self.current_outs = [m.detach().cpu() for m in mu_out]
            mu_out = sum(mu_out)
        else:
            mu_out = self.mean_modules(ins, indices=indices)

        if issubclass(type(self.std_modules), nn.ModuleList):
            std_out = []
            for i in range(len(self.std_modules)):
                std_out.append(self.std_modules[i](ins[:, i], indices=indices) * self.weights[i])
            std_out = sum(std_out)
        else:
            if output_heads:
                std_out = self.std_modules(ins.reshape(ins.shape[0], ins.shape[1]*ins.shape[2], *ins.shape[3:]), indices=indices)
            else:
                std_out = self.std_modules(ins, indices=indices)

        if self.nn_lin:
            mu_out = self.nn_lin(mu_out)

        if is_seq:
            mu_out = mu_out.reshape(n_batch, n_seq, *mu_out.shape[1:])
            std_out = std_out.reshape(n_batch, n_seq, *std_out.shape[1:])
            if self.record_heads and self.separate_heads:
                self.current_outs = [c.reshape(n_batch, n_seq, *mu_out.shape[2:]) for c in self.current_outs]

        if self.poutput['dist'] == Normal:
            return Normal(mu_out, sigmoid(std_out))
        elif self.poutput['dist'] == MultivariateNormal:
            #TODO implement full-covariance encodings
            return MultivariateNormal(mu_out, covariance_matrix=sigmoid(std_out).unsqueeze(1) * eye(mu_out.shape[1]))
        elif self.poutput['dist'] == MultivariateNormal:
            return RandomWalk(mu_out, sigmoid(std_out))


        
        
        
        
########################################################################
####        Bernoulli layers

def BernoulliLayer(pinput, poutput, **kwargs):
    take_conv = poutput.get('channels')
    if take_conv:
        return BernoulliLayer2D(pinput, poutput, **kwargs)
    else:
        return BernoulliLayer1D(pinput, poutput, **kwargs)

class BernoulliLayer1D(nn.Module):
    requires_deconv_indices = False
    '''Module that outputs parameters of a Bernoulli distribution.'''
    def __init__(self, pinput, poutput, **kwargs):
        super(BernoulliLayer1D , self).__init__()
        self.pinput = pinput; self.poutput = poutput
        self.input_dim = sum(checklist(p['dim'])[-1] for p in checklist(pinput));
        self.output_dim = cumprod(checktuple(poutput['dim']))[-1]
        cum_output_dim = cumprod(self.output_dim)[-1]
        self.modules_list = nn.Sequential(nn.Linear(self.input_dim, cum_output_dim), nn.Sigmoid())
        init_module(self.modules_list, 'Sigmoid')
        
    def forward(self, ins,  *args, **kwargs):
        mu = self.modules_list(ins)
        mu = mu.view(*mu.shape[:-1], *checktuple(self.output_dim))
        return Bernoulli(probs=mu)

class BernoulliLayer2D(nn.Module):
    def __init__(self, pinput, poutput, hidden_module=None, **kwargs):
        nn.Module.__init__(self)
        self.pinput = pinput; self.poutput = poutput
        # retrieve convolutional parameters from hidden module
        """
        if hidden_module:
            self.separate_heads = hidden_module.separate_heads
            self.record_heads = pinput.get('record_heads', True)
            phidden = {k: [v[-1]] if issubclass(type(v), list) else [v] for k, v in hidden_module.phidden.items()}
            phidden['channels'].append(poutput['channels'])
        """
        self.phidden = self.get_phidden(self.pinput, self.poutput)

        #TODO pooling specific arguments
        self.pool_target_size = None;
        self.requires_deconv_indices = False
        if hidden_module:
            if hidden_module.has_pooling:
                self.pool_target_size = hidden_module.pool_target_sizes[-1]
                self.requires_deconv_indices = True
            self.phidden['output_padding'] = checklist(hidden_module.phidden[0].get('output_padding'))[-1]


        # non-linearity
        self.nn_lin = poutput.get('nn_lin')
        self.nn_lin_args = poutput.get('nn_lin_args', {})
        if self.nn_lin:
            if issubclass(type(self.nn_lin), str):
                self.nn_lin = getattr(nn, self.nn_lin)(**self.nn_lin_args)

        self.mean_modules, weights = self.init_modules(self.phidden)
        if weights is not None:
            self.weights = nn.ParameterList(weights)

        self.is_seq = kwargs.get('is_seq', False)

    def get_phidden(self, pinput, poutput):
        in_channels = checklist(pinput.get('channels', 1))[0]
        out_channels = checklist(poutput.get('channels', 1))[0]
        if pinput.get('kernel_size') is None:
            raise TypeError('keyword kernel_size must be defined in pinput')
        kernel_size = checklist(pinput.get('kernel_size'))[0]
        padding = checklist(pinput.get('padding', int(ceil(kernel_size / 2))))[0]
        dilation = checklist(pinput.get('dilation', 1))[0]
        stride = checklist(pinput.get('stride', 1))[0]
        conv_dim = pinput.get('conv_dim', len(checktuple(poutput['dim'])))
        windowed = poutput.get('windowed', False)
        output_padding = pinput.get('output_padding', [None])[0]
        phidden = {'channels': [in_channels, out_channels],
                   'kernel_size': kernel_size, 'conv_dim': conv_dim,
                   'padding': padding, 'dilation': dilation, 'stride': stride,
                   'windowed': windowed, 'output_padding': output_padding}
        if pinput.get('heads'):
            phidden['heads'] = pinput['heads']
        if pinput.get('pool'):
            phidden['pool'] = pinput['pool'][0]
        return phidden

    def get_mean_module(self, phidden):
        # make mean module of the layer
        deconv_class = phidden.get('class', [DeconvolutionalLatent])
        deconv_layer = deconv_class[-1].conv_layer
        mean_module = deconv_layer(phidden['channels'][0], phidden['channels'][1], phidden['conv_dim'],
                                       phidden['kernel_size'],
                                       pool=phidden.get('pool'),
                                       dilation=phidden['dilation'],
                                       dropout=None, padding=phidden['padding'],
                                       stride=phidden['stride'], windowed=phidden['windowed'],
                                       batch_norm=None, nn_lin=None, output_padding=phidden['output_padding'])
        nn.init.xavier_normal_(mean_module._modules['conv_module'].weight)
        return mean_module

    def init_modules(self, phidden):
        if phidden.get('heads'):
            mean_modules = nn.ModuleList([self.get_mean_module(phidden) for i in range(phidden['heads'])])
            weights = [nn.Parameter(tensor([1.])) for i in range(phidden['heads'])]
        else:
            mean_modules = self.get_mean_module(phidden)
            weights = None
        return mean_modules, weights


    def forward(self, ins,  *args, indices=None, output_heads=False, **kwargs):
        n_batch = ins.shape[0]; n_seq = None

        # if not any channel dimension, add it
        if len(ins.shape)<= self.phidden['conv_dim']+1:
            ins = ins.unsqueeze(-self.phidden['conv_dim']-1)

        #TODO verify this sequence thing...!
        is_seq = self.is_seq
        if is_seq:
            n_seq = ins.shape[1]
            if not ins.is_contiguous():
                ins = ins.contiguous()
            ins = ins.view(ins.shape[0]*ins.shape[1], *ins.shape[2:])

        if issubclass(type(self.mean_modules), nn.ModuleList):
            mu_out = []
            for i in range(len(self.mean_modules)):
                mu_out.append(self.mean_modules[i](ins[:, i], indices=indices) * self.weights[i])
            if output_heads:
                self.current_outs = [m.detach().cpu() for m in mu_out]
            mu_out = sum(mu_out)
        else:
            mu_out = self.mean_modules(ins, indices=indices)

        mu_out = sigmoid(mu_out)

        if is_seq:
            mu_out = mu_out.reshape(n_batch, n_seq, *mu_out.shape[1:])
            if self.record_heads and self.separate_heads:
                self.current_outs = [c.reshape(n_batch, n_seq, *mu_out.shape[2:]) for c in self.current_outs]

        return Bernoulli(probs=mu_out)


########################################################################
####        Categorical layers


def CategoricalLayer( pinput, poutput, **kwargs):
    if issubclass(type(poutput['dim']), list) or poutput.get('conv'):
        return CategoricalLayer2D(pinput, poutput, **kwargs)
    else:
        return CategoricalLayer1D(pinput, poutput, **kwargs)


class CategoricalLayer1D(nn.Module):
    requires_deconv_indices = False
    '''Module that outputs parameters of a Bernoulli distribution.'''
    def __init__(self, pinput, poutput, **kwargs):
        nn.Module.__init__(self)
        self.dist = poutput['dist']

        self.input_dim = sum(checklist(p['dim'])[-1] for p in checklist(pinput));
        self.output_dim = cumprod(checktuple(poutput['dim']))[-1]

        self.modules_list = nn.Sequential(nn.Linear(checklist(self.input_dim)[-1], self.output_dim), nn.Softmax(dim=1))
        init_module(self.modules_list, 'Softmax')
        
    def forward(self, ins,  *args, **kwargs):
        mu = self.modules_list(ins)
        return self.dist(probs=mu)


class CategoricalLayer2D(nn.Module):
    def __init__(self, pinput, poutput, hidden_module=None, **kwargs):
        nn.Module.__init__(self)
        self.pinput = pinput; self.poutput = poutput
        # retrieve convolutional parameters from hidden module
        """
        if hidden_module:
            self.separate_heads = hidden_module.separate_heads
            self.record_heads = pinput.get('record_heads', True)
            phidden = {k: [v[-1]] if issubclass(type(v), list) else [v] for k, v in hidden_module.phidden.items()}
            phidden['channels'].append(poutput['channels'])
        """
        self.phidden = self.get_phidden(self.pinput, self.poutput)

        self.pool_target_size = None;
        self.requires_deconv_indices = False
        if hidden_module:
            if hidden_module.has_pooling:
                self.pool_target_size = hidden_module.pool_target_sizes[-1]
                self.requires_deconv_indices = True

        # non-linearity
        self.nn_lin = poutput.get('nn_lin')
        self.nn_lin_args = poutput.get('nn_lin_args', {})
        if self.nn_lin:
            if issubclass(type(self.nn_lin), str):
                self.nn_lin = getattr(nn, self.nn_lin)(**self.nn_lin_args)

        self.mean_modules, weights = self.init_modules(self.phidden)
        if weights is not None:
            self.weights = nn.ParameterList(weights)

        self.is_seq = kwargs.get('is_seq', False)

    def get_phidden(self, pinput, poutput):
        # make convolutional arguments from input and output parameters
        in_channels = checklist(pinput.get('channels', 1))[-1]
        out_channels = checklist(poutput.get('channels', 1))[0]
        if pinput.get('kernel_size') is None:
            raise TypeError('keyword kernel_size must be defined in pinput')
        kernel_size = checklist(pinput.get('kernel_size'))[-1]
        padding = checklist(pinput.get('padding', 0))[-1]
        dilation = checklist(pinput.get('dilation', 1))[-1]
        stride = checklist(pinput.get('stride', 1))[-1]
        conv_dim = pinput.get('conv_dim', len(checktuple(pinput['dim'])))
        windowed = poutput.get('windowed', False)
        output_padding = checklist(pinput.get('output_padding', [None]))[-1]
        phidden = {'channels':[in_channels, out_channels],
                       'kernel_size':kernel_size, 'conv_dim':conv_dim,
                       'padding':padding, 'dilation':dilation, 'stride':stride,
                       'windowed':windowed, 'output_padding':output_padding}
        if pinput.get('heads'):
            phidden['heads'] = pinput['heads']
        if pinput.get('pool'):
            phidden['pool'] = pinput.get('pool', [None])[-1]
        return phidden

    def get_mean_module(self, phidden):
        # make mean module of the layer
        deconv_class = phidden.get('class', [DeconvolutionalLatent])
        deconv_layer = deconv_class[-1].conv_layer
        mean_module = deconv_layer(phidden['channels'][0], phidden['channels'][1], phidden['conv_dim'],
                                       phidden['kernel_size'],
                                       pool=phidden.get('pool'),
                                       dilation=phidden['dilation'],
                                       dropout=None, padding=phidden['padding'],
                                       stride=phidden['stride'], windowed=phidden['windowed'],
                                       batch_norm=None, nn_lin='Softmax', output_padding=phidden['output_padding'])
        nn.init.xavier_normal_(mean_module._modules['conv_module'].weight)
        return mean_module

    def init_modules(self, phidden):
        if phidden.get('heads'):
            mean_modules = nn.ModuleList([self.get_mean_module(phidden) for i in range(phidden['heads'])])
            weights = [nn.Parameter(tensor([1.])) for i in range(phidden['heads'])]
        else:
            mean_modules = self.get_mean_module(phidden)
            weights = None
        return mean_modules, weights


    def forward(self, ins,  *args, indices=None, output_heads=False, **kwargs):
        n_batch = ins.shape[0]; n_seq = None

        # if not any channel dimension, add it
        if len(ins.shape) <= self.phidden['conv_dim']+1:
            ins = ins.unsqueeze(-self.phidden['conv_dim']-1)

        #TODO verify this sequence thing...!
        is_seq = self.is_seq
        if is_seq:
            n_seq = ins.shape[1]
            if not ins.is_contiguous():
                ins = ins.contiguous()
            ins = ins.view(ins.shape[0]*ins.shape[1], *ins.shape[2:])

        if issubclass(type(self.mean_modules), nn.ModuleList):
            mu_out = []
            for i in range(len(self.mean_modules)):
                mu_out.append(self.mean_modules[i](ins[:, i], indices=indices) * self.weights[i])
            if output_heads:
                self.current_outs = [m.detach().cpu() for m in mu_out]
            mu_out = sum(mu_out)
        else:
            mu_out = self.mean_modules(ins, indices=indices)

        if is_seq:
            mu_out = mu_out.reshape(n_batch, n_seq, *mu_out.shape[1:])
            if self.record_heads and self.separate_heads:
                self.current_outs = [c.reshape(n_batch, n_seq, *mu_out.shape[2:]) for c in self.current_outs]

        return Categorical(probs=mu_out)


class FlowLayer(nn.Module):

    def __init__(self, pinput, poutput, *args, **kwargs):
        super(FlowLayer, self).__init__()
        if not issubclass(type(poutput.get('dist')), Flow):
            raise ValueError("FlowLayer should only be used with Flow distribution, not %s"%poutput.get('dist'))
        layer = get_module_from_density(poutput['dist'].dist)
        self.distrib_layer = layer(pinput, poutput, *args, **kwargs)
        self.poutput = poutput
        self.flow = poutput['dist']._flow

    def forward(self, *args, **kwargs):
        out = self.distrib_layer(*args, **kwargs)
        return self.poutput['dist'](out)


def get_module_from_density(distrib):
    if issubclass(type(distrib), Flow):
        return FlowLayer
    if distrib == Empirical:
        return EmpiricalLayer
    if distrib in (dist.Normal, dist.RandomWalk, dist.MultivariateNormal):
        return GaussianLayer
    elif distrib == dist.Bernoulli:
        return BernoulliLayer
    elif distrib == dist.Categorical:
        return CategoricalLayer
    elif distrib in [dist.Multinomial]:
        return CategoricalLayer
    else:
        raise TypeError('Unknown distribution type : %s'%distrib)
