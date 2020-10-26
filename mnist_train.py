import argparse, torch, pdb
import vschaos
from vschaos import distributions as dist, hashes
from vschaos.data import dataset_from_torchvision, Binary, Normalize, Unsqueeze, Flatten
from vschaos.vaes import VanillaVAE
from vschaos.utils import checklist, Warmup, DataLoader
from vschaos.criterions import ELBO, SemiSupervisedELBO, MultiDivergence, Classification
from vschaos.train import SimpleTrainer, train_model


parser = argparse.ArgumentParser()
valid_flows = ['PlanarFlow', 'RadialFlow', 'IAFlow', 'MAFlow', 'AffineFlow', 'AffineCouplingFlow', 'MaskedCouplingFlow']
parser.add_argument('--binarized', type=int, default=0, help="binarize dataset")
parser.add_argument('--dims', type=int, nargs='+', default = [64], help='layerwise latent dimensions')
parser.add_argument('--hidden_dims', type=int, nargs='+', default = [800, 500, 200, 100], help = 'layer-wise hidden dimensions')
parser.add_argument('--hidden_num', type=int, nargs='+', default = [2, 1, 1, 1], help = 'layer-wise number of hidden layers')
parser.add_argument('--linked', type=int, default=0, help="linked encoders/decoders for semi-supervision")
parser.add_argument('--dropout', type=float, default=None, nargs="*", help = "layer-wise dropout")
parser.add_argument('--mlp_layer', type=str, choices=['linear', 'residual', 'gated', 'gated_residual', 'self-attention'], default="linear", help="layer-wise module classes")
parser.add_argument('--flow_blocks', type=str, nargs="*", default=['PlanarFlow'], choices=valid_flows, help='defining flow blocks')
parser.add_argument('--flow_length', type=int, default=None, help='layerwise length of corresponding normalizing flow')
parser.add_argument('--flow_amortization', type=str, default='input', choices=['input', 'none', 'latent'], help='amortization mode fo flows')

# convolutional encoder parameters
parser.add_argument('--conv_norm_enc', type=str, choices=['none', 'instance', 'batch'], default="batch" )
parser.add_argument('--channels', type=int, nargs='+', default = None, help = 'number of channels for convolution network (layer 1)')
parser.add_argument('--conv_enc', type=str, choices=['conv', 'gated', 'multi_conv', 'multi_gated'], default='conv', help="type of convolution module")
parser.add_argument('--kernel_size', type=int, nargs='+', default = [5, 3], help = 'size of convolution network\'s kernels')
parser.add_argument('--pool', type=int, nargs='+', default=None, help = 'size of pooling modules for convolution kernels')
parser.add_argument('--dilation', type=int, nargs='+', default=[1,1], help='dilation of each convolution layer')
parser.add_argument('--stride', type=int, nargs='+', default=[2,2], help="stride of convolution layer")
parser.add_argument('--mlp2conv', type=str, choices=['unflatten', 'conv1x1'], default='flattening mode for enconding module')

# convolutional decoder parameters
parser.add_argument('--conv_norm_dec', type=str, choices=['none', 'instance', 'batch'], default="batch")
parser.add_argument('--conv_dec', type=str, choices=['conv', 'gated', 'multi_conv', 'multi_gated'], default=None, help="type of deconvolution module (empty for same as encoder)")
parser.add_argument('--decoder_channels', type=int, nargs='+', default = None, help = 'number of channels for deconvolution module (empty for same as encoder)')
parser.add_argument('--decoder_kernel_size', type=int, nargs='+', default = None, help = 'size of deconvolution network\'s kernels (empty for same as encoder)')
parser.add_argument('--decoder_dilation', type=int, nargs='+', default=None, help='dilation of each deconvolution layer (empty for same as encoder)')
parser.add_argument('--decoder_stride', type=int, nargs='+', default=None, help='stride of each deconvolution layer (empty for same as encoder)')
parser.add_argument('--decoder_hidden_dims', type=int, nargs="*",  default=None, help='hidden dimension of decoder flattening module (empty for same as encoder)')
parser.add_argument('--decoder_hidden_num', type=int, nargs="*", default=None, help='hidden number of layers for decoder flattening module (empty for same as encoder)')
# parser.add_argument('--model', type=str, choices=['vae', 'dlgm', 'lvae'], default='vae', help='model type')
parser.add_argument('--windowed', type=str, default=None, help="enables windowed convolution")
parser.add_argument('--conv2mlp', type=str, choices=['unflatten', 'conv1x1'], default='flattening mode for decoding module')

# conditioning parameters
parser.add_argument('--conditioning_target', type=str, choices=['encoder', 'decoder', 'both'], default='decoder', help="conditioned modules")
parser.add_argument('--conditioning_layer', type=int, nargs="*", default=[], help="conditioned latent layer (default: no)")
# semi-supervision
parser.add_argument('--semi_supervision', type=str, nargs="*", default=[], help="semi-supervised tasks")
parser.add_argument('--semi_supervision_dropout', type=int, default=0.5, help="semi-supervised tasks")
parser.add_argument('--semi_supervision_warmup', type=int, default=100, help="semi-supervised tasks")
# classification
parser.add_argument('--classification', type=str, nargs="*", default=[], help="classified tasks")
parser.add_argument('--classif_nlayers', type=int, default=2, help="classifiers number of layers")
parser.add_argument('--classif_dim', type=int, default=200, help="classifiers capacity")
parser.add_argument('--classif_lr', type=float, default=None, help="classifiers learning rate")
parser.add_argument('--classif_weight', type=float, default=10.0, help="classifiers learning rate")


# Save options
parser.add_argument('--name', type=str, default='mnist', help="model name")
parser.add_argument('--savedir', type=str, default='saves', help='results directory')
# training parameters
parser.add_argument('--priors', type=str, nargs="*", choices=['isotropic', 'wiener', 'none'], default=None, help="regularization priors")
parser.add_argument('--cuda', type=int, default=[-1], nargs="*", help="cuda id (-1 for cpu)")
parser.add_argument('--alpha', type=float, default = 2.0, nargs='*', help="RD / JSD parameter")
parser.add_argument('--rec_weight', type=float, default = 1.0, help="reconstruciton weighting")
parser.add_argument('--beta', type=float, default = 1.0, nargs='*', help="DKL weighting")
parser.add_argument('--warmup', type=int, default = 100, nargs="*", help="warmup of DKL weighting")
parser.add_argument('--reconstruction', type=str, choices=['ll' ,'mse', 'adversarial'], default='ll', help="reconstruction loss")
parser.add_argument('--regularization', type=str, choices=['kld', 'mmd', 'renyi', 'adversarial'], nargs="*", default='kld', help="regularization loss")
parser.add_argument('--batch_size', type=int, default=64, help="batch size of training")
parser.add_argument('--batch_split', type=int, default=None)
parser.add_argument('--epochs', type=int, default=3000, help="nuber of training epochs")
parser.add_argument('--lr', type=float, nargs="*",  default=1e-3, help="learning rate")
parser.add_argument('--save_epochs', type=int, default=10, help="model save periodicity")
parser.add_argument('--plot_epochs', type=int, default=5, help="plotting periodicity")

# monitoring arguments
parser.add_argument('--tensorboard', type=str, default=None, help="activates tensorboard support (please enter runs/ target folder)")
parser.add_argument('--plot_reconstructions', type=int, default=1, help="make reconstruction plots (enter number of plotted examples)")
parser.add_argument('--plot_latentspace', type=int, default=1, help="make latent space plots" )
parser.add_argument('--plot_statistics', type=int, default=1, help="make statistics plots")
parser.add_argument('--plot_losses', type=int, default=1, help="make losses plots")
parser.add_argument('--plot_dims', type=int, default=1, help="make dimensions plots")
parser.add_argument('--plot_consistency', type=int, default=0, help="make consistency plots (multi-layer)" )
parser.add_argument('--plot_confusion', type=int, default=1, help="make consistency plots (multi-layer)" )
parser.add_argument('--plot_npoints', type=int, default=500, help="number of points used for latent plots")
parser.add_argument('--check', type=int, default=0, help="triggers breakpoint before training")

args = parser.parse_args()

scheduled_parameters = {}

# Data
if args.binarized:
    transforms = [Binary()]
else:
    transforms = [Normalize(mode="minmax", scale="unipolar")]
if args.channels is not None:
    transforms.append(Unsqueeze(-3))
else:
    transforms.append(Flatten(-2))
dataset = dataset_from_torchvision('MNIST', transforms=transforms)
dataset.scale_transforms()

if args.binarized:
    input_dist = dist.Bernoulli
    input_dim = dataset.data.shape[-2]*dataset.data.shape[-1] if args.channels is None else dataset.data.shape[-2:]
    input_params = {'dim': input_dim, 'dist': input_dist, 'nn_lin':'Sigmoid', 'channels': (args.channels is not None)}
else:
    input_dist = dist.Normal
    input_dim = dataset.data.shape[-2]*dataset.data.shape[-1] if args.channels is None else dataset.data.shape[-2:]
    input_params = {'dim': input_dim, 'dist': input_dist, 'channels': (args.channels is not None)}

# Hidden & latent parameters
latent_params = []; hidden_params = [];
assert len(args.hidden_dims) >= len(args.dims), "specification of hidden dimensions inconsistent"
assert len(args.hidden_num) >= len(args.dims), "specification of hidden layers inconsistent"
args.dropout = checklist(args.dropout, len(args.dims)); args.mlp_layer = checklist(args.mlp_layer, len(args.mlp_layer))

if args.priors:
    assert len(args.priors) >= len(args.dims)
else:
    args.priors = ['none']*len(args.dims)

if args.flow_length:
    flow_blocks = [getattr(vschaos.modules.flow, f) for f in args.flow_blocks]
    flow_params = {'dim': args.dims[-1], 'blocks': flow_blocks, 'flow_length': args.flow_length,
                   'amortized': args.flow_amortization}


for i, dim in enumerate(args.dims):
    # parse hidden parameters
    current_hidden = {'dim':args.hidden_dims[i], 'nlayers':args.hidden_num[i], 'dropout':args.dropout[i],
            'layer':hashes.layer_hash[args.mlp_layer[i]], 'normalization':'batch', 'linked':args.linked}
    # parse shrub parameters in case
    hidden_params.append({'encoder':current_hidden, 'decoder':dict(current_hidden)})

    # parse latent parameters
    latent_dist = dist.RandomWalk if args.priors[i] == 'wiener' else dist.Normal
    prior = None if args.priors is None else hashes.prior_hash[args.priors[i]]
    latent_params.append({'dim': dim, 'dist': latent_dist, 'prior':hashes.prior_hash[args.priors[i]]})

    # parse flows
    if args.flow_length is not None:
        latent_params[-1]['flows'] = flow_params

    # parse convolutional parameters
    if i==0 and args.channels is not None:
        if issubclass(type(input_params), list):
            input_params[0]['channels'] = 1; input_params[1]['channels'] = 1;
        else:
            input_params['channels'] = 1
        # encoder parameters
        encoder_params = {**hidden_params[0]['encoder'], 'channels': args.channels, 'kernel_size': args.kernel_size, 'pool': args.pool,
                      'class':hashes.conv_hash[args.conv_enc][0], 'dilation': args.dilation, 'linked':args.linked,
                      'batch_norm_conv':args.conv_norm_enc, 'dropout': args.dropout[0], 'dropout_conv':None,
                      'stride':args.stride, 'conv_dim':2, 'normalization': 'batch', 'conv2mlp':args.conv2mlp}
        # decoder parameters (could be coded better, but... i'm an artist)
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
        decoder_params['linked'] = True
        hidden_params[0] = {'encoder': encoder_params, 'decoder':decoder_params}
    else:
        hidden_params[0]['decoder']['normalization']=None
    if issubclass(type(input_params), list):
        hidden_params[0]['encoder'] = checklist(dict(hidden_params[0]['encoder']), n=len(input_params), copy=True)
        hidden_params[0]['decoder'] = checklist(dict(hidden_params[0]['decoder']), n=len(input_params), copy=True)
        #hidden_params = [{'encoder':hidden_params[i], 'decoder':decoder_params[i]} for i in range(len(hidden_params))]
    # parse conditioning

    if i in args.conditioning_layer:
        if args.conditioning_target in ['encoder', 'both']:
            hidden_params[i]['encoder']['label_params'] = {'class':{'dim': dataset.classes['class']['_length'], 'dist':dist.Categorical}}
        elif args.conditioning_target in ['decoder', 'both']:
            hidden_params[i]['decoder']['label_params'] = {'class':{'dim': dataset.classes['class']['_length'], 'dist':dist.Categorical}}

if len(args.semi_supervision) > 0:
    latent_params[-1] = checklist(latent_params[-1])
    latent_params[-1].append({'dim': dataset.classes['class']['_length'], 'dist': dist.Multinomial, 'task': 'class'})
semi_supervision_dropout = Warmup(range=[0, args.semi_supervision_warmup], values=[0.0, args.semi_supervision_dropout])
scheduled_parameters['semi_supervision_dropout'] = semi_supervision_dropout

# creating vae
device = torch.device(args.cuda[0]) if args.cuda[0] >= 0 else -1
vae = VanillaVAE(input_params, latent_params, hidden_params, device=args.cuda[0])
if device != -1:
    vae.cuda(device)

# setting up optimizer
optim_params = {'optimizer':'Adam', 'optim_args':{'lr':args.lr}, 'scheduler':'ReduceLROnPlateau'}
vae.init_optimizer(optim_params)

# creating loss
rec_loss = args.rec_weight * hashes.rec_hash[args.reconstruction]()
reg_loss = []; args.regularization = checklist(args.regularization, n=len(latent_params))
for i, reg in enumerate(args.regularization):
    reg_args = {}
    if reg == 'renyi':
        reg_args = {'alpha':args.alpha[i]}
    elif reg == 'adversarial':
        reg_args = {'latent_params':latent_params, 'optim_params':optim_params}
    reg_loss.append(hashes.reg_hash[reg](**reg_args, reduction='none'))
reg_loss = MultiDivergence(reg_loss)
if args.semi_supervision:
    loss = SemiSupervisedELBO(reconstruction_loss=rec_loss, regularization_loss=reg_loss, warmup=args.warmup,
                beta=args.beta)
else:
    loss = ELBO(reconstruction_loss=rec_loss, regularization_loss=reg_loss, warmup=args.warmup, beta=args.beta)

classifier = None
if args.classification:
    params = latent_params[-1][0] if args.semi_supervision else latent_params[-1]
    layer = (-1, 0) if args.semi_supervision else -1
    classifier = Classification(params, 'class', {'dim':dataset.classes['class']['_length'], 'dist':dist.Categorical},
                                layer=layer, hidden_params={'dim':args.classif_dim, 'nlayers':args.classif_nlayers},
                                optim_params={'lr':args.classif_lr or args.lr})
    if device != -1:
        classifier = classifier.to(device)
    loss = loss + args.classif_weight*classifier


# plot arguments
plots = {}; synth = {}
loader = DataLoader
if args.plot_reconstructions:
    plots['reconstructions'] = {'plot_multihead':True, 'preprocess':False, 'preprocessing':dataset.transforms, 'as_sequence': False, 'n_points':5}
if args.plot_latentspace:
    plots['latent_space'] = {'tasks':dataset.tasks, 'balanced':False, 'n_points':args.plot_npoints, 'batch_size':args.batch_size}
if args.plot_statistics:
    plots['statistics'] = {'preprocess':False, 'tasks':dataset.tasks, 'n_points':args.plot_npoints, 'batch_size':args.batch_size}
if args.plot_losses:
    plots['losses'] = {'loss':loss}
if args.plot_confusion and classifier is not None:
    plots['confusion'] = {'classifiers':classifier, 'n_points':args.plot_npoints, 'batch_size':args.batch_size}

trainer = SimpleTrainer(vae, dataset, loss, name=args.name, plots=plots, synth=synth, split=args.batch_split,
                        tasks=["class"],
                        semi_supervision=["class"] if args.semi_supervision else None,
                        semi_supervision_dropout=semi_supervision_dropout,
                        scheduled_params=scheduled_parameters,
                        use_tensorboard=f"{args.tensorboard}/{args.name}" if args.tensorboard else None)

if args.check:
    pdb.set_trace()
log_file = args.savedir+'/'+args.name+'/log.txt'

scheduled_parameters={'semi_supervision_dropout':semi_supervision_dropout}
with torch.cuda.device(device):
    train_model(trainer, options={'epochs':args.epochs,
                                  'save_epochs':args.save_epochs,
                                  'results_folder':args.savedir+'/'+args.name,
                                  'batch_size':args.batch_size,
                                  'batch_split':args.batch_split,
                                  'plot_epochs':args.plot_epochs},
                        save_with={'files':dataset.files,
                                   'script_args':args,
                                   'transforms':transforms},
                        **scheduled_parameters)

