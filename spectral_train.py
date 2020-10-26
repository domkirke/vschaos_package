#!/usr/bin/env python3
"""
# -*- coding: utf-8 -*-
Created on Mon Oct 22 14:26:52 2018

@author: chemla
"""

# -*- coding: utf-8 -*-
import matplotlib, pdb
matplotlib.use('agg')
import torch, argparse, numpy as np, copy, os

import vschaos
from vschaos import hashes
import vschaos.vaes as vaes

from vschaos.data.data_audio import DatasetAudio
from vschaos.data import data_transforms as dt
import vschaos.distributions as dist

from vschaos import DataParallel
from vschaos.modules.modules_convolution import *
from vschaos.modules.modules_bottleneck import *
from vschaos.modules import flow as flow
from vschaos.criterions.criterion_elbo import ELBO, SemiSupervisedELBO
from vschaos.criterions.criterion_logdensities import LogDensity
from vschaos.criterions.criterion_functional import MSE
from vschaos.criterions.criterion_divergence import KLD,MMD,RD,MultiDivergence
from vschaos.criterions.criterion_misc import InfoNCE, Classification
from vschaos.distributions.distribution_priors import ClassPrior
from vschaos.monitor.visualize_dimred import PCA, ICA
from vschaos.train.train_train import train_model, SimpleTrainer
from vschaos.utils.dataloader import DataLoader, RawDataLoader
from vschaos.utils.misc import checklist, choices, parse_filtered_classes
from vschaos.utils.schedule import Warmup

#%% Parsing arguments
parser = argparse.ArgumentParser()

valid_flows = ['PlanarFlow', 'RadialFlow', 'IAFlow', 'MAFlow', 'AffineFlow', 'AffineCouplingFlow', 'MaskedCouplingFlow']
# Dataset options
dataset_args = parser.add_argument_group(title='dataset', description="Arguments relative to dataset import")
dataset_args.add_argument('--dbroot', type=str, default='/Users/chemla/Datasets/acidsInstruments-ordinario', help='database root path (no / at end)')
dataset_args.add_argument('--transform', type=str, default='stft-1024', help='imported audio transform')
dataset_args.add_argument('--sequence', type=int, default=0, help="sequence length (leave empty for no sequence)")
dataset_args.add_argument('--random_start', type=int, default=0, help="random beginning of imported sequence")
dataset_args.add_argument('--offline', type=int, default=0, help="uses asynchronous import (default : 0)")
dataset_args.add_argument('--offline_selector', type=str, choices=['all', 'random', 'random_slice', 'sequence', 'sequence_batch'], default='all', help="type of asynchronous import")
dataset_args.add_argument('--save_dataset', type=int, default=0, help="save dataset after loading")
dataset_args.add_argument('--load_dataset', type=int, default=0, help="load dataset from pickle (has to be called first with --save 1")
dataset_args.add_argument('--tasks', type=str, nargs="*", default=None, help='tasks imported for plotting and conditioning')
dataset_args.add_argument('--files', type=int, default=None, help='number of files imported from dataset')

# Normalization options
pp_args = parser.add_argument_group(title='preprocessing', description="Arguments relative to pre-processing")
pp_args.add_argument('--preprocessing', type=str, default="magnitude", choices=["magnitude", "polar", "mag+inst-f"])
pp_args.add_argument('--mag_prenorm', type=str, choices=['std', 'max', 'none'], default="none", help='preprocessing : normalization applied before non-linearity')
pp_args.add_argument('--mag_nnlin', type=str, default='log1p', choices=['log1p', 'log', 'tanh', 'none'], help="preprocessing : type of non linearity used")
pp_args.add_argument('--mag_postnorm', type=str, choices=['std', 'max', 'none'], default="std",help='preprocessing : normalization applied after non-linearity')
pp_args.add_argument('--phase_normalize', type=str, choices=['none', 'unipolar', 'bipolar'], default="bipolar")
pp_args.add_argument('--preprocessing_batch', type=int, default=30, help="number of files used for preprocessing scaling")
pp_args.add_argument('--nn_lin', type=str, choices=['none', 'softplus', 'tanh', 'softsign'], default='softplus', help="non linearity used at generation layer")

# Architecture options
arch_args = parser.add_argument_group(title='architecture', description="Arguments relative to model architecture")
arch_args.add_argument('--model', type=str, default="vae", choices=['vae', 'shrub'], help="model class")
arch_args.add_argument('--dims', type=int, nargs='+', default = [64], help='layerwise latent dimensions')
arch_args.add_argument('--hidden_dims', type=int, nargs='+', default = [1000, 500, 200, 100], help = 'layer-wise hidden dimensions')
arch_args.add_argument('--hidden_num', type=int, nargs='+', default = [1, 1, 1, 1], help = 'layer-wise number of hidden layers')
arch_args.add_argument('--dropout', type=float, default=None, nargs="*", help = "layer-wise dropout")
arch_args.add_argument('--mlp_layer', type=str, choices=['linear', 'residual', 'gated', 'gated_residual', 'self-attention'], default="linear", help="layer-wise module classes")
arch_args.add_argument('--mlp_norm', type=str, choices=['none', 'batch'], default='batch')
arch_args.add_argument('--flow_blocks', type=str, nargs="*", default=['PlanarFlow'], choices=valid_flows, help='defining flow blocks')
arch_args.add_argument('--flow_length', type=int, default=None, help='layerwise length of corresponding normalizing flow')
arch_args.add_argument('--flow_amortization', type=str, default='input', choices=['input', 'none', 'latent'], help='amortization mode fo flows')
# convolutional encoder parameters
arch_args.add_argument('--conv_norm_enc', type=str, choices=['none', 'instance', 'batch'], default="batch" )
arch_args.add_argument('--channels', type=int, nargs='+', default = None, help = 'number of channels for convolution network (layer 1)')
arch_args.add_argument('--conv_enc', type=str, choices=['conv', 'gated', 'multi_conv', 'multi_gated'], default='conv', help="type of convolution module")
arch_args.add_argument('--kernel_size', type=int, nargs='+', default = [5, 3], help = 'size of convolution network\'s kernels')
arch_args.add_argument('--pool', type=int, nargs='+', default=None, help = 'size of pooling modules for convolution kernels')
arch_args.add_argument('--dilation', type=int, nargs='+', default=[1,1], help='dilation of each convolution layer')
arch_args.add_argument('--stride', type=int, nargs='+', default=[2,2], help="stride of convolution layer")
arch_args.add_argument('--mlp2conv', type=str, choices=['unflatten', 'conv1x1'], default='flattening mode for enconding module')
# convolutional decoder parameters
arch_args.add_argument('--conv_norm_dec', type=str, choices=['none', 'instance', 'batch'], default="batch")
arch_args.add_argument('--conv_dec', type=str, choices=['conv', 'gated', 'multi_conv', 'multi_gated'], default=None, help="type of deconvolution module (empty for same as encoder)")
arch_args.add_argument('--decoder_channels', type=int, nargs='+', default = None, help = 'number of channels for deconvolution module (empty for same as encoder)')
arch_args.add_argument('--decoder_kernel_size', type=int, nargs='+', default = None, help = 'size of deconvolution network\'s kernels (empty for same as encoder)')
arch_args.add_argument('--decoder_dilation', type=int, nargs='+', default=None, help='dilation of each deconvolution layer (empty for same as encoder)')
arch_args.add_argument('--decoder_stride', type=int, nargs='+', default=None, help='stride of each deconvolution layer (empty for same as encoder)')
arch_args.add_argument('--decoder_hidden_dims', type=int, nargs="*",  default=None, help='hidden dimension of decoder flattening module (empty for same as encoder)')
arch_args.add_argument('--decoder_hidden_num', type=int, nargs="*", default=None, help='hidden number of layers for decoder flattening module (empty for same as encoder)')
arch_args.add_argument('--windowed', type=str, default=None, help="enables windowed convolution")
arch_args.add_argument('--conv2mlp', type=str, choices=['unflatten', 'conv1x1'], default='flattening mode for decoding module')

# conditioning parameters
arch_args.add_argument('--conditioning', type=str, nargs="*", default=[], help="conditioning tasks")
arch_args.add_argument('--conditioning_target', type=str, nargs="*", choices=['encoder', 'decoder', 'both'], default='decoder', help="conditioned modules")
arch_args.add_argument('--conditioning_layer', type=int, nargs="*", default=None, help="conditioned latent layers (default: all)")
# semi-supervision
arch_args.add_argument('--semi_supervision', type=str, nargs="*", default=[], help="semi-supervised tasks")
arch_args.add_argument('--semi_supervision_dropout', type=int, default=0.5, help="semi-supervised tasks")
arch_args.add_argument('--semi_supervision_warmup', type=int, default=100, help="semi-supervised tasks")
# classification
arch_args.add_argument('--classification', type=str, nargs="*", default=[], help="classified tasks")
arch_args.add_argument('--classif_nlayers', type=int, default=1, help="classifiers number of layers")
arch_args.add_argument('--classif_dim', type=int, default=16, help="classifiers capacity")
arch_args.add_argument('--classif_lr', type=float, default=None, help="classifiers learning rate")
arch_args.add_argument('--classif_target', type=int, nargs="*", default=None, help="classifiers target dimensions")

# Training options
training_args = parser.add_argument_group(title='training', description="Arguments specific to training parameters")
training_args.add_argument('--alpha', type=float, default = 2.0, nargs='*', help="RD / JSD parameter")
training_args.add_argument('--learnable_alpha', type=int, default=0, help="is alpha learnable (RD/JSD)")
training_args.add_argument('--rec_weight', type=float, default = 1.0, help="reconstruciton weighting")
training_args.add_argument('--beta', type=float, default = 1.0, nargs='*', help="DKL weighting")
training_args.add_argument('--warmup', type=int, default = 100, nargs="*", help="warmup of DKL weighting")
training_args.add_argument('--reconstruction', type=str, choices=['ll' ,'mse', 'adversarial'], default='ll', help="reconstruction loss")
training_args.add_argument('--regularization', type=str, choices=['kld', 'mmd', 'renyi', 'adversarial'], nargs="*", default='kld', help="regularization loss")
training_args.add_argument('--priors', type=str, nargs="*", choices=['isotropic', 'wiener', 'none'], default=['isotropic'], help="regularization priors")
training_args.add_argument('--batch_size', type=int, default=64, help="batch size of training")
training_args.add_argument('--batch_split', type=int, default=None)
training_args.add_argument('--epochs', type=int, default=3000, help="nuber of training epochs")
training_args.add_argument('--lr', type=float, nargs="*",  default=1e-3, help="learning rate")
training_args.add_argument('--save_epochs', type=int, default=50, help="model save periodicity")
training_args.add_argument('--plot_epochs', type=int, default=20, help="plotting periodicity")
# Save options
training_args.add_argument('--name', type=str, default='vae_spectrum', help="model name")
training_args.add_argument('--savedir', type=str, default='saves', help='results directory')
training_args.add_argument('--cuda', type=int, default=[-1], nargs="*", help="cuda id (-1 for cpu)")
training_args.add_argument('--check', type=int, default=0, help="triggers breakpoint before training")
# Plot options
monitor_args = parser.add_argument_group(title='monitoring', description="Arguments specific to monitoring")
monitor_args.add_argument('--tensorboard', type=str, default=None, help="activates tensorboard support (please enter runs/ target folder)")
monitor_args.add_argument('--plot_reconstructions', type=int, default=1, help="make reconstruction plots (enter number of plotted examples)")
monitor_args.add_argument('--plot_latentspace', type=int, default=1, help="make latent space plots" )
monitor_args.add_argument('--plot_statistics', type=int, default=1, help="make statistics plots")
monitor_args.add_argument('--plot_losses', type=int, default=1, help="make losses plots")
monitor_args.add_argument('--plot_dims', type=int, default=1, help="make dimensions plots")
monitor_args.add_argument('--plot_confusion', type=int, default=1, help="make consistency plots (multi-layer)" )
monitor_args.add_argument('--audio_reconstructions', type=int, default=3, help="reconstruct audio samples (enter number of reconstructions)")
#monitor_args.add_argument('--audio_interpolate', type=int, default=0, help="make stat istics plots")
monitor_args.add_argument('--plot_npoints', type=int, default=500, help="number of points used for latent plots")
args = parser.parse_args()

scheduled_parameters = {}
#%% Loader & loss functionprint('[Info] Loading data...')
# import parameters
print('Importing dataset...')
# Create dataset object
target_transforms = None
if args.preprocessing == "magnitude":
    transforms = [dt.Magnitude(contrast='log1p', shrink=1, normalize={'mode': "minmax", 'scale': 'unipolar'})]
    # transforms = [Magnitude(preprocessing='log1p')]
if args.preprocessing == "polar":
    transforms = [dt.Polar(mag_options={'preprocessing': 'log1p', 'shrink':8, 'normalize': {'mode': "gaussian", 'scale': 'unipolar'}},
           phase_options={'normalize': {'mode': 'minmax', 'scale': 'bipolar'}})]
elif args.preprocessing == "mag+inst-f":
    transforms = [dt.PolarInst(mag_options={'preprocessing': 'log1p', 'shrink':8, 'normalize': {'mode': "gaussian", 'scale': 'unipolar'}},
           phase_options={'normalize': {'mode': 'minmax', 'scale': 'bipolar'}})]

if args.sequence != 0 and not args.offline:
    transforms.extend([dt.Sequence(args.sequence, dim=0)])
    target_transforms = dt.Repeat(args.sequence, dim=-1, unsqueeze=True)
transforms = dt.ComposeAudioTransform(transforms)

audioSet = DatasetAudio(args.dbroot, tasks=args.tasks, transforms=[dt.Mono(), dt.Squeeze(0), dt.STFT(2048)])
audioSet.import_data(scale=None)
audioSet.write_transform('stft-1024', scale=False)
# loading data

audioSet = DatasetAudio(args.dbroot, tasks=args.tasks, transforms=transforms, target_transforms=target_transforms)
if args.files is not None:
    audioSet = audioSet.random_subset(args.files)
if args.offline:
    if args.sequence == 0:
        assert args.offline_selector in ['all', 'random','random_slice'],\
            'if sequence == 0 offline_selector must be all, random or random_slice'
        offline_selector_args = {}
    else:
        assert args.offline_selector in ['sequence', 'sequence_batch'],\
            'if sequence > 0 offline_selector must be sequence, sequence_batch'
        offline_selector_args = {'sequence_length': args.sequence, 'random_idx': args.random_start, 'batches': 5}
    offline_selector = hashes.async_hash[args.offline_selector]
    audioSet, original_transforms = audioSet.load_transform(args.transform, offline=args.offline,
                                                            selector=offline_selector, selector_args=offline_selector_args)
else:
    audioSet, original_transforms = audioSet.load_transform(args.transform, offline=args.offline)

audioSet.scale_transforms(scale=args.preprocessing_batch)
full_transforms = original_transforms + transforms
audioSet.apply_transforms()
if not args.sequence:
    audioSet.flatten_data(dim=0)
if args.channels is not None:
    audioSet.transforms = dt.ComposeAudioTransform(dt.Unsqueeze(-2))
    full_transforms.append(audioSet.transforms[0])
audioSet.construct_partition(['train', 'test'], [0.8, 0.2], balancedClass=False)

#%% BUILDING VAE
if args.preprocessing in ['mag+inst-f', "polar"]:
    assert args.sequence, "polar and mag+inst-f transforms have to be sequences (--sequence > 0)"
    input_dim = audioSet[0][0][0].size(-1)
    mag_params =  {'dim':input_dim, 'dist':dist.Normal, 'nn_lin':hashes.nnlin_hash[args.nn_lin], 'nn_lin_args':hashes.nnlin_args_hash[args.nn_lin], 'conv':args.channels is not None}
    phase_nn_lin = {'none':None, 'unipolar':'Softplus', 'bipolar':None}[args.phase_normalize]
    phase_params =  {'dim':input_dim, 'dist':dist.Normal, 'nn_lin':phase_nn_lin, 'conv':args.channels is not None}
    input_params = [mag_params, phase_params]
elif args.preprocessing in ['magnitude']:
    input_dim = audioSet[0][0].size(-1)
    input_params =  {'dim':input_dim, 'dist':dist.Normal, 'nn_lin':'Softplus', 'conv':args.channels is not None}


# hidden & latent parameters
latent_params = []; hidden_params = []
if args.model == "shrub":
    assert len(args.dims) >= 2, "ShrubVAE needs at least two layers"
    assert args.sequence is not None, "ShrubVAE needs a sequence length"
assert len(args.hidden_dims) >= len(args.dims), "specification of hidden dimensions inconsistent"
assert len(args.hidden_num) >= len(args.dims), "specification of hidden layers inconsistent"
args.dropout = checklist(args.dropout, len(args.dims)); args.mlp_layer = checklist(args.mlp_layer, len(args.mlp_layer))
args.conditioning_layer = args.conditioning_layer or list(range(len(args.dims)))

if args.priors:
    assert len(args.priors) >= len(args.dims)
else:
    args.priors = ['none']*len(args.dims)

if args.flow_length:
    flow_blocks = [getattr(vschaos.modules.flow, f) for f in args.flow_blocks]
    flow_params = {'dim':args.dims[-1], 'blocks': flow_blocks, 'flow_length':args.flow_length, 'amortized':args.flow_amortization}

for i, dim in enumerate(args.dims):
    # parse hidden parameters
    current_hidden = {'dim':args.hidden_dims[i], 'nlayers':args.hidden_num[i], 'dropout':args.dropout[i],
            'layer':hashes.layer_hash[args.mlp_layer[i]], 'normalization':'batch', 'linked':False}
    hidden_params.append({'encoder':current_hidden, 'decoder':dict(current_hidden)})
    # parse latent parameters
    latent_dist = dist.RandomWalk if args.priors[i] == 'wiener' else dist.Normal
    latent_params.append({'dim': dim, 'dist': latent_dist, 'prior':hashes.prior_hash[args.priors[i]]})
    # parse flows
    if args.flow_length is not None:
        latent_params[-1]['flows'] = flow_params
    # parse convolutional parameters
    if i==0 and args.channels is not None:
        if issubclass(type(input_params), list):
            input_params[0]['channels'] = 1; input_params[1]['channels'] = 1
        else:
            input_params['channels'] = 1
        # encoder parameters
        encoder_params = {**hidden_params[0]['encoder'], 'channels': args.channels, 'kernel_size': args.kernel_size, 'pool': args.pool,
                      'class':hashes.conv_hash[args.conv_enc][0], 'dilation': args.dilation,
                      'batch_norm_conv':args.conv_norm_enc, 'dropout': args.dropout[0], 'dropout_conv':None,
                      'stride':args.stride, 'conv_dim': len(checklist(input_dim)), 'normalization': 'batch', 'conv2mlp':args.conv2mlp}
        # decoder parameters
        decoder_params = dict(encoder_params)
        decoder_params['channels'] = args.decoder_channels if args.decoder_channels else decoder_params['channels']
        decoder_params['kernel_size'] = args.decoder_kernel_size if args.decoder_kernel_size else decoder_params['kernel_size']
        decoder_params['stride'] = args.decoder_stride if args.decoder_stride else decoder_params['stride']
        decoder_params['dilation'] = args.decoder_dilation if args.decoder_dilation else decoder_params['dilation']
        decoder_params['dim'] = args.decoder_hidden_dims[0] if args.decoder_hidden_dims else decoder_params['dim']
        decoder_params['nlayers'] = args.decoder_hidden_num[0] if args.decoder_hidden_num else decoder_params['nlayers']
        decoder_params['class'] = hashes.conv_hash[args.conv_dec][1] if args.conv_dec else hashes.conv_hash[args.conv_enc][1]
        decoder_params['mlp2conv'] = args.mlp2conv
        decoder_params['batch_norm_conv'] = args.conv_norm_dec
        decoder_params['linked'] = args.preprocessing == "magnitude"
        hidden_params[0] = {'encoder': encoder_params, 'decoder':decoder_params}
    else:
        hidden_params[0]['decoder']['normalization']=None
    if issubclass(type(input_params), list):
        hidden_params[0]['encoder'] = checklist(dict(hidden_params[0]['encoder']), n=len(input_params), copy=True)
        hidden_params[0]['decoder'] = checklist(dict(hidden_params[0]['decoder']), n=len(input_params), copy=True)
        #hidden_params = [{'encoder':hidden_params[i], 'decoder':decoder_params[i]} for i in range(len(hidden_params))]
    # parse conditioning
    if len(args.conditioning) > 0:
        if i in args.conditioning_layer:
            if args.conditioning_target in ['encoder', 'both']:
                hidden_params[i]['encoder']['label_params'] = {k:{'dim': audioSet.classes[k]['_length'], 'dist':dist.Categorical} for k in args.conditioning}
            elif args.conditioning_target in ['decoder', 'both']:
                hidden_params[i]['decoder']['label_params'] = {k:{'dim': audioSet.classes[k]['_length'], 'dist':dist.Categorical} for k in args.conditioning}

if len(args.semi_supervision) > 0:
    latent_params[-1] = checklist(latent_params[-1])
    for meta in args.semi_supervision:
        label_dim = audioSet.classes[meta]['_length']
        latent_params[-1].append({'dim': label_dim, 'dist': dist.Multinomial, 'task': meta})
semi_supervision_dropout = Warmup(range=[0, args.semi_supervision_warmup], values=[0.0, args.semi_supervision_dropout])
scheduled_parameters['semi_supervision_dropout'] = semi_supervision_dropout

# creating vae
vae = hashes.model_hash[args.model](input_params, latent_params, hidden_params, device=args.cuda[0])

# setting up optimizer
optim_params = {'optimizer':'Adam', 'optim_args':{'lr':args.lr}, 'scheduler':'ReduceLROnPlateau'}
vae.init_optimizer(optim_params)

# cuda handling
if len(args.cuda) > 1:
    vae = DataParallel(vae, args.cuda, output_device=args.cuda[0])

# creating elbo loss
reduction = "seq" if args.sequence > 0 else None
if args.preprocessing == "mag+inst-f":
    rec_loss = [MSE(), MSE()]
else:
    rec_loss = args.rec_weight * hashes.rec_hash[args.reconstruction]()
reg_loss = []; args.regularization = checklist(args.regularization, n=len(latent_params))
for i, reg in enumerate(args.regularization):
    reg_args = {}
    if reg == 'renyi':
        reg_args = {'alpha':args.alpha[i], 'learnable_alpha':args.learnable_alpha}
    reg_loss.append(hashes.reg_hash[reg](**reg_args, reduction='none'))
reg_loss = MultiDivergence(reg_loss)
if args.semi_supervision:
    loss = SemiSupervisedELBO(reconstruction_loss=rec_loss, regularization_loss=reg_loss, warmup=args.warmup,
                beta=args.beta, reduction=reduction)
else:
    loss = ELBO(reconstruction_loss=rec_loss, regularization_loss=reg_loss, warmup=args.warmup, beta=args.beta, reduction=reduction)

# creating classifiers
classifiers = []
for task in args.classification:
    params = latent_params[-1][0] if args.semi_supervision else latent_params[-1]
    layer = (-1, 0) if args.semi_supervision else -1
    args.classif_target = args.classif_target if args.classif_target != [] else None
    # arbitrary 100 here, but otherwise classification loss to tiny to compete with elbo
    classifier = 100*Classification(params, task, {'dim':audioSet.classes[task]['_length'], 'dist':dist.Categorical},
                                layer=layer, hidden_params={'dim':args.classif_dim, 'nlayers':args.classif_nlayers},
                                optim_params={'lr':args.classif_lr or args.lr}, target_dims=args.classif_target)
    if args.cuda[0] >= 0:
        classifier = classifier.to(torch.device('cuda:%d'%args.cuda[0])) 
    loss = loss + classifier
    classifiers.append(classifier)

# plot arguments
plots = {}; synth = {}
loader = DataLoader
if args.preprocessing in ['polar', 'mag+inst-f']:
    rec_transforms = [transforms[0][0], transforms[0][1]]
else:
    rec_transforms = [transforms[0]]
if args.plot_reconstructions:
    plots['reconstructions'] = [{'plot_multihead':True, 'preprocess':False, 'preprocessing':rec_transforms, 'as_sequence': False, 'n_points':5, 'name':'with_pp', 'use_external_loader':loader},
                                {'plot_multihead':True, 'preprocess':False, 'preprocessing':None, 'as_sequence': False, 'n_points':5, 'name':'without_pp', 'use_external_loader':loader}]
if args.plot_latentspace:
    plots['latent_space'] = {'transformation':PCA, 'tasks':audioSet.tasks, 'balanced':False, 'n_points':args.plot_npoints, 'batch_size':args.batch_size}
if args.plot_dims:
    plots['latent_dims'] = {'reductions':[None, PCA, ICA], 'tasks':audioSet.tasks, 'balanced':False, 'n_points':args.plot_npoints, 'batch_size':args.batch_size}
if args.plot_statistics:
    plots['statistics'] = {'preprocess':False, 'tasks':audioSet.tasks, 'n_points':args.plot_npoints, 'batch_size':args.batch_size}
if args.plot_losses:
    plots['losses'] = {'loss':loss}
if args.plot_confusion:
    plots['confusion'] = {'classifiers':classifiers, 'n_points':args.plot_npoints, 'batch_size':args.batch_size}
if args.audio_reconstructions:
    if not args.preprocessing in ['magnitude', 'polar', 'mag+inst-f']:
        print('[Warning] preprocessing %s is not invertible, audio reconstruction is disabled.'%args.preprocessing)
    else:
        ids = torch.randperm(len(audioSet.files))[:args.audio_reconstructions]
        files = [audioSet.files[i] for i in ids]
        meta = {k: v[ids] for k, v in audioSet.metadata.items()}
        synth['audio_reconstructions'] = {'transforms':full_transforms, 'sequence':args.sequence, 'norm':True, "files":files, 'take_sequences':args.sequence != 0, 'metadata':meta}

trainer = SimpleTrainer(vae, audioSet, loss, name=args.name, plots=plots, synth=synth, split=args.batch_split,
                        tasks=args.conditioning+args.classification,
                        semi_supervision=args.semi_supervision,
                        semi_supervision_dropout=semi_supervision_dropout,
                        scheduled_params=scheduled_parameters,
                        use_tensorboard=f"{args.tensorboard}/{args.name}" if args.tensorboard else None)


if args.check:
    pdb.set_trace()
log_file = args.savedir+'/'+args.name+'/log.txt'

device = torch.device(args.cuda[0]) if args.cuda[0] >= 0 else -1
scheduled_parameters={'semi_supervision_dropout':semi_supervision_dropout}
with torch.cuda.device(device):
    train_model(trainer, options={'epochs':args.epochs,
                                  'save_epochs':args.save_epochs,
                                  'results_folder':args.savedir+'/'+args.name,
                                  'batch_size':args.batch_size,
                                  'batch_split':args.batch_split,
                                  'plot_epochs':args.plot_epochs},
                        save_with={'files':audioSet.files,
                                   'script_args':args,
                                   'full_transforms':full_transforms,
                                   'target_transforms':target_transforms,
                                   'transform':args.transform},
                        **scheduled_parameters)

