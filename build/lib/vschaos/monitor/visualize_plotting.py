import torch, os, gc
import matplotlib
import matplotlib.pyplot as plt
import librosa

from sklearn.metrics import confusion_matrix
from . import visualize_dimred as dr

from ..distributions.distribution_priors import get_default_distribution
from ..distributions import Normal
from ..utils import decudify, merge_dicts, CollapsedIds, check_dir, recgetitem, apply_method
from ..utils.dataloader import DataLoader
from ..utils import get_flatten_meshgrid
import matplotlib.patches as mpatches
from matplotlib.ticker import StrMethodFormatter
from torchvision.utils import make_grid, save_image

from . import visualize_core as core
from ..modules.modules_convolution import *

################################################
########        DECORATORS
####

def partition_dependent(func):
    func.partition_dependent = True
    return func
def partition_independent(func):
    func.partition_dependent = False
    return func


################################################
########        RECONSTRUCTION PLOTS
####

eps=1e-7

def get_plotting_loader(dataset, partition=None, n_points=None, ids=None, is_sequence=None, tasks=None, loader=None, batch_size=None):
    if partition is not None:
        dataset = dataset.retrieve(partition)
    if ids is None:
        full_id_list = np.arange(len(dataset))
        ids = full_id_list[np.random.permutation(len(full_id_list))[:n_points]] if ids is None else ids
    # get data
    Loader = loader if loader else DataLoader
    loader = Loader(dataset, batch_size, ids=ids, is_sequence=is_sequence, tasks=tasks)
    return loader

@partition_dependent
def plot_reconstructions(dataset, model, label=None, n_points=10, out=None, preprocess=True, preprocessing=None, partition=None, use_external_loader=None,
                         epoch=None, name=None, loader=None, ids=None, plot_multihead=False, reinforcers=None, multilayer=False, forward_hook=None, **kwargs):

    # get plotting ids
    if not use_external_loader:
        loader = get_plotting_loader(dataset, partition=partition, n_points=n_points,
                                     ids=ids, is_sequence=model.take_sequences, tasks=label, loader=loader)
    else:
        if partition is not None:
            dataset = dataset.retrieve(partition)
        ids = ids or np.random.permutation(len(dataset))[:n_points]
        loader = use_external_loader(dataset, None,  ids=ids, is_sequence=model.take_sequences, tasks=label)

    data, metadata = next(loader.__iter__())

    # forward
    add_args = {}
    if hasattr(model, 'prediction_params'):
        add_args['n_preds'] = model.prediction_params['n_predictions']
    add_args['epoch'] = kwargs.get('epoch')

    outputs = []
    layers = range(len(model.platent)) if multilayer else [-1]
    with torch.no_grad():
        for l in layers:
            if forward_hook is None:
                outputs.append(model.forward(data, y=metadata, from_layer=l, teacher_prob=0, **kwargs, **add_args))
            else:
                data, output = forward_hook(model=model, reinforcers=reinforcers, x=data, y=metadata, period="train", **kwargs, **add_args)
                outputs.append(output)
            if reinforcers is not None:
                outputs[-1] = reinforcers.forward(outputs[-1])
    if out:
        out += '/reconstruction/'
        check_dir(out)

    #pdb.set_trace()
    for l, vae_out in enumerate(outputs):
        data = checklist(data)
        vae_out['x_params'] = checklist(vae_out['x_params'])
        preprocessing = checklist(preprocessing, n=len(vae_out['x_params']))
        if vae_out.get('x_reinforced') is not None:
            vae_out['x_reinforced'] = checklist(vae_out['x_reinforced'])

        figs = {}; axes = {}
        suffix = ""
        suffix = suffix+f"_layer{layers[l]}" if multilayer else suffix
        suffix = suffix+"_"+str(partition) if partition is not None else suffix
        suffix = suffix+"_%d"%epoch if epoch is not None else suffix

        for i in range(len(vae_out['x_params'])):
            multihead_outputs = None
            if plot_multihead and hasattr(model.decoders[0].out_modules[0], 'current_outs'):
                multihead_outputs = model.decoders[0].out_modules[0].current_outs

            if vae_out.get('x_params') is not None:
                fig, ax = core.plot_distribution(vae_out['x_params'][i], target=data[i], preprocessing=preprocessing[i], preprocess=preprocess, multihead=multihead_outputs, **kwargs)
                if not out is None:
                    for k, f in fig.items():
                        fig_path = f"{name}_{k}"
                        if len(vae_out['x_params']) > 1:
                            fig_path += "_%d"%i
                        figs[os.path.basename(fig_path)] = f
                        fig_path_tmp = f"{out}/{fig_path}{suffix}"
                        f.savefig(fig_path_tmp+".pdf", format="pdf")

            fig_reinforced = None
            if vae_out.get('x_reinforced') is not None:
                fig_reinforced, ax_reinforced = core.plot_distribution(vae_out['x_reinforced'][i], target=data[i], preprocessing=preprocessing, preprocess=preprocess, out=fig_path, multihead=multihead_outputs, **kwargs)
                fig_path = out+'_%s_reinforced'%name if len(vae_out) == 1 else out+'%s_%d_reinforced'%(name, i)
                figs[os.path.basename(fig_path)+'_reinforced'] = fig; axes[os.path.basename(fig_path)+'_reinforced'] = ax
                if not out is None:
                    fig_path = f"{out}/{name}{suffix}.pdf"
                    fig.savefig(fig_path, format="pdf")

            if vae_out.get('cpc_states_enc') is not None:
                cmap_cpc = matplotlib.cm.get_cmap('plasma')
                for layer, cpc in enumerate(vae_out['cpc_states_enc']):
                    fig_cpc, ax_cpc = plt.subplots(n_points, 1)
                    for ex in range(n_points):
                        cpc_num_dim =  vae_out['cpc_states_enc'][layer].shape[-1]
                        for cpc_dim in range(cpc_num_dim):
                            ax_cpc[ex].plot(vae_out['cpc_states_enc'][layer][ex, ..., cpc_dim].detach().cpu(), linewidth = 0.6, c=cmap_cpc(cpc_dim/cpc_num_dim))
                    if not out is None:
                        figs[os.path.basename(fig_path)+'_cpc'] = fig_cpc
                        axes[os.path.basename(fig_path)+'_cpc'] = ax_cpc
                        fig_cpc.savefig(f"{out}/{fig_path}{suffix}_cpc.pdf", format="pdf")

    del data; del outputs 
    return figs, axes

@partition_independent
def plot_samples(dataset, model, label=None, n_points=10, layer=None, priors=None, out=None, preprocess=False, preprocessing=None, partition=None,
                         epoch=None, name=None, loader=None, ids=None, plot_multihead=False, reinforcers=None, **kwargs):

    layer = layer or range(len(model.platent))
    priors = priors or [None]*len(model.platent)
    for l in layer:
        prior = priors[l] or model.platent[l].get('prior') or get_default_distribution(model.platent[l]['dist'], batch_shape=(model.platent[l]['dim'],))
        current_z = model.format_input_data(prior.sample((n_points,)))
        current_y = label
        # forward
        with torch.no_grad():
            vae_out = model.decode(current_z, y=current_y, from_layer=l)
        vae_out = {'x_params':vae_out[0]['out_params']}
        if reinforcers is not None:
            vae_out = reinforcers.forward(vae_out)

        vae_out['x_params'] = checklist(vae_out['x_params'])
        if vae_out.get('x_reinforced') is not None:
            vae_out['x_reinforced'] = checklist(vae_out['x_reinforced'])

        figs = {}; axes = {}
        if out:
            out += '/samples/'
            check_dir(out)

        suffix = ""
        suffix = "_"+str(partition) if partition is not None else suffix
        suffix = suffix+"_%d"%epoch if epoch is not None else suffix
        for i in range(len(vae_out['x_params'])):
            multihead_outputs = None
            if plot_multihead and hasattr(model.decoders[0].out_modules[0], 'current_outs'):
                multihead_outputs = model.decoders[0].out_modules[0].current_outs

            current_name = f"{name}_{l}"
            if vae_out.get('x_params') is not None:
                fig_path = current_name if len(vae_out) == 1 else '%s_%d'%(current_name, i)
                fig, ax = core.plot_distribution(vae_out['x_params'][i], preprocessing=preprocessing, preprocess=preprocess, multihead=multihead_outputs, out=fig_path, **kwargs)
                figs[os.path.basename(fig_path)] = fig; axes[os.path.basename(fig_path)] = ax
                if not out is None:
                    fig_path = f"{out}/{fig_path}{suffix}.pdf"
                    fig.savefig(fig_path, format="pdf")

            fig_reinforced = None
            if vae_out.get('x_reinforced') is not None:
                fig_reinforced, ax_reinforced = core.plot_distribution(vae_out['x_reinforced'][i], target=data_pp[i], preprocessing=preprocessing, preprocess=preprocess, out=fig_path, multihead=multihead_outputs, **kwargs)
                fig_path = out+'%d_%s_reinforced'%(l, name) if len(vae_out) == 1 else out+'_%d_%s_%d_reinforced'%(l, name, i)
                figs[os.path.basename(fig_path)+'_reinforced'] = fig; axes[os.path.basename(fig_path)+'_reinforced'] = ax
                if not out is None:
                    fig_path = f"{out}/{name}{suffix}.pdf"
                    fig.savefig(fig_path, format="pdf")

    return figs, axes


def import_axes(fig, ax_to_remove, ax_to_import, position=None):
    ax_to_import.remove();
    position = position or ax_to_remove.get_position();
    ax_to_import.set_position(position);
    subplotspec = ax_to_remove.get_subplotspec()
    ax_to_import.set_subplotspec(subplotspec)
    ax_to_import.figure = fig;
    fig.axes.append(ax_to_import)
    fig.add_axes(ax_to_import)
    ax_to_remove.remove()
    ax_to_import.set_title('')


@partition_dependent
def plot_prediction(dataset, model, label=None, n_points=10, out=None, preprocess=True, preprocessing=None, partition=None,
                         epoch=None, name=None, loader=None, ids=None, plot_multihead=False, reinforcers=None, **kwargs):

    loader = get_plotting_loader(dataset, partition=partition, n_points=n_points,
                                 ids=ids, is_sequence=model.take_sequences, tasks=label, loader=loader)
    data, metadata = next(loader.__iter__())
    if preprocess:
        if issubclass(type(model.pinput), list):
            preprocessing = preprocessing if preprocessing else [None]*len(dataset.data)
            data_pp = [None]*len(dataset.data)
            if not issubclass(type(preprocessing), list):
                preprocessing = [preprocessing]*len(dataset.data)
            for i, pp in enumerate(preprocessing):
                if not pp is None:
                    data_pp[i] = preprocessing(data[i])
        else:
            data_pp = preprocessing(data) if preprocessing is not None else data
    else:
        data_pp = data

    fig_params = [{'name':'prediction', 'predict':True, 'interp_dropout':0.0, 'teacher_prob':0.0},
                  {'name':'interp_05', 'predict':False, 'interp_dropout':0.5, 'teacher_prob':0.0},
                  {'name':'prednint', 'predict':True, 'interp_dropout':0.5, 'teacher_prob':0.0},
                  {'name':'baseline', 'predict':False, 'interp_dropout':0.0, 'teacher_prob':0.0}]
    # forward
    # add_args = {}
    # if hasattr(model, 'prediction_params'):
    #     add_args['n_preds'] = model.prediction_params['n_predictions']
    # add_args['epoch'] = kwargs.get('epoch')

    results = {v['name']: None for v in fig_params}

    with torch.no_grad():
        for params in fig_params:
            results[params['name']] = model.forward(data_pp, y=metadata, **params)
        #encoder_out = {'z_params_enc':[encoder_out[i]['out_params'] for i in range(len(encoder_out))],
        #               'z_enc':[encoder_out[i]['out'] for i in range(len(encoder_out))]}

    n_examples = results['baseline']['z_enc'][-1].size(0)
    n_dims = model.platent[-1]['dim']
    cmap = matplotlib.cm.get_cmap('hsv')

    out += "/prediction"
    check_dir(out)
    figs = {}; axes = {};

    legend_elements = []
    fontP = matplotlib.font_manager.FontProperties()
    fontP.set_size('xx-small')
    for i, params in enumerate(fig_params):
        legend_elements.append(matplotlib.lines.Line2D([0], [0], color=cmap(i / len(fig_params)), lw=3, label=params['name']))

    for n in range(n_examples):
        n_rows, n_columns = core.get_divs(n_dims)
        fig, ax = plt.subplots(n_rows, n_columns)
        for t_ind, t in enumerate(fig_params):
            for i in range(n_rows):
                for j in range(n_columns):
                    n_prd = results[t['name']]['z_enc'][-1].size(1); x_range = np.arange(n_prd)
                    mean = results[t['name']]['z_params_enc'][-1].mean[n, :, i*n_columns+j].cpu()
                    stddev = results[t['name']]['z_params_enc'][-1].stddev[n, :, i * n_columns + j].cpu()
                    ax[i, j].plot(mean, c=cmap(t_ind/len(fig_params)), linewidth=0.5)
                    ax[i, j].fill_between(x_range, mean+stddev, mean-stddev, alpha=0.1, color=cmap(t_ind/len(fig_params)))
                    ax[i, j].set_xticks([])
                    ax[i, j].set_yticklabels(ax[i, j].get_yticks(), fontsize="xx-small")
                if j==n_columns-1 and i==0:
                    ax[i, j].legend(handles=legend_elements, bbox_to_anchor=(0.9, 0.9), loc='upper left', title="", prop=fontP)

        current_name = f"pred_{partition or str()}_{n}"
        if out is not None:
            fig.savefig(out+'/%s_%d.pdf'%(current_name,epoch), format="pdf")
        figs[current_name] = fig; axes[current_name] = ax
        plt.close('all')

        for t_ind, t in enumerate(fig_params):
            x_out = checklist(results[t['name']]['x_params']); data_pp = checklist(data_pp)
            for x_ind in range(len(checklist(model.pinput))):
                fig, ax = core.plot_distribution(x_out[x_ind], target=data_pp[x_ind],
                                                 preprocessing=preprocessing[x_ind],
                                                 preprocess=preprocess, is_sequence=True, **kwargs)
                if out is not None:
                    fig = checklist(fig)
                    for fig_idx, fig in enumerate(fig):
                        current_name = f"rec_{partition or str()}_{fig_idx}"
                        fig.savefig(out + '/%s_%d.pdf' % (current_name, epoch), format="pdf")
                figs[current_name] = fig
                axes[current_name] = ax
        plt.close('all')

    return figs, axes




def get_plot_subdataset(dataset, n_points=None, partition=None, ids=None):
    if ids is None:
        if partition:
            current_partition = dataset.partitions[partition]
            ids = current_partition[np.random.permutation(len(current_partition))[:n_points]]
        else:
            ids = np.random.permutation(dataset.data.shape[0])[:n_points]
    dataset = dataset.retrieve(ids)
    return dataset, ids

@partition_independent
def grid_latent(dataset, model, layer=-1, reduction=dr.PCA, n_points=None, ids=None, grid_shape=10, scales=[-3.0, 3.0], batch_size=None, loader=None, label=None, out=None, epoch=None, partition=None, **kwargs):
    n_dims = 2 #n_dims or model.platent[layer]['dim']
    zs, meshgrids, idxs = get_flatten_meshgrid(n_dims, scales, grid_shape); 
    idxs_hash = {i:idxs[i] for i in range(len(idxs))}

    latent_dims = model.platent[layer]['dim']
    if latent_dims > 2:
        dataset = get_plot_subdataset(dataset, n_points=n_points, partition=partition, ids=ids)
        Loader = loader if loader else DataLoader
        loader = Loader(dataset[0], batch_size, is_sequence=model.take_sequences, tasks=label)
        outs = []
        with torch.no_grad():
            for x, y in loader:
                outs.append(model.encode(model.format_input_data(x)))
        outs = merge_dicts(outs) 
        #TODO general sampling
        data_zs = outs[layer]['out_params'].mean
        dimred = reduction(n_components=n_dims)
        dimred.fit_transform(data_zs.cpu().numpy())
        zs = dimred.inverse_transform(zs)

    with torch.no_grad():
        outs = model.decode(model.format_input_data(zs), layer=layer)[0]['out_params']
    
    grid = torch.zeros(*meshgrids[0].shape, 1, *outs.mean.shape[1:])
    grid_std = torch.zeros(*meshgrids[0].shape, 1, *outs.mean.shape[1:])

    for n, idx in enumerate(idxs):
        x, y = idx
        grid[x, y, 0, :] = outs[n].mean.cpu()
        grid_std[x, y, 0, :] = outs[n].stddev.cpu()

    grid = grid.view(grid.shape[0]*grid.shape[1], *grid.shape[2:])
    grid_std = grid_std.view(grid_std.shape[0]*grid_std.shape[1], *grid_std.shape[2:])

    grid_img = make_grid(grid, nrow=grid_shape)
    grid_std_img = make_grid(grid_std, nrow=grid_shape)
    
    epoch = "" if epoch is None else "_%d"%epoch
    layer = len(model.platent) + layer if layer < 0 else layer
    if not os.path.isdir(out+'/grid'):
        os.makedirs(out+'/grid')
    save_image(grid_img, out+'/grid/grid_%d%s.png'%(layer, epoch))
    save_image(grid_std_img, out+'/grid/std_grid_%d%s.png'%(layer, epoch))


    fig = plt.figure()
    plt.imshow(grid_img.transpose(0,2), aspect="auto")

    return [fig], [fig.axes]


def image_export(dataset, model, label=None, n_rows=None, ids=None, out=None, partition=None, n_points=10, **kwargs):
    if ids is None:
        if partition:
            current_partition = dataset.partitions[partition]
            ids = current_partition[np.random.permutation(len(current_partition))[:n_points]]
        else:
            ids = np.random.permutation(dataset.data.shape[0])[:n_points]

    # get item ids
    if not ids is None:
        n_rows = int(np.sqrt(len(ids)))
        reconstruction_ids = ids
    else:
        n_rows = n_rows or 5
        reconstruction_ids = np.random.permutation(dataset.data.shape[0])[:n_rows**2]
    
    # get corresponding images
    images = dataset.data[reconstruction_ids]
    if not label is None:
        if not issubclass(type(label), list):
            label = [label]
        metadata = {t: dataset.metadata[t][reconstruction_ids] for t in label} 
    else:
        metadata = None
        
    # forward
    out_image = model.forward(images, y=metadata)['x_params'].mean
    out_image = out_image.reshape(out_image.size(0), 1, 28, 28)
    out_grid = make_grid(out_image, nrow=n_rows)
    if out:
        save_image(out_grid, out+'_grid.png')
        
    fig = plt.figure()
    plt.imshow(np.transpose(out_grid.cpu().detach().numpy(), (1,2,0)), aspect='auto')
    del out_image
    return [fig], [fig.axes]


def plot_tf_reconstructions(dataset, model, label=None, n_points=10, out=None, preprocess=True, preprocessing=None, partition=None,
                         name=None, loader=None, ids=None, plot_multihead=False, reinforcers=None, **kwargs):

    if partition is not None:
        dataset = [dataset[0].retrieve(partition), dataset[1].retrieve(partition)]

    if ids is None:
        if issubclass(type(dataset[0].data), list):
            full_id_list = np.array(range(len(dataset[0])))
            ids = full_id_list[np.random.permutation(len(full_id_list))[:n_points]] if ids is None else ids
        else:
            full_id_list = np.array(range(len(dataset[0])))
            ids = full_id_list[np.random.permutation(len(full_id_list))[:n_points]] if ids is None else ids

    # get audio data
    Loader = loader if loader else DataLoader
    loader_audio = Loader(dataset[0], None, ids=ids, is_sequence=model[0].take_sequences, tasks=label)
    loader_symbol = Loader(dataset[1], None, ids=ids, is_sequence=model[1].take_sequences, tasks=label)
    data_audio, metadata_audio = next(loader_audio.__iter__())
    #TODO why
    data_symbol = [np.array(x)[ids] for x in dataset[1].data]
    if preprocess:
        if preprocessing[0]:
            data_audio_pp = preprocessing[0](data_audio)
        if preprocessing[1]:
            data_symbol_pp = preprocessing[1](data_symbol)

    else:
        data_audio_pp = data_audio
        data_symbol_pp = data_symbol


    # forward
    add_args = {}
    if hasattr(model, 'prediction_params'):
        add_args['n_preds'] = model.prediction_params['n_predictions']
    
    with torch.no_grad():
        audio_out = model[0].forward(data_audio_pp, y=metadata_audio, **add_args)
        symbol_out = model[1].forward(data_symbol_pp, y=metadata_audio, **add_args)
        z_audio_tf = symbol_out['z_params_enc'][-1].mean
        if model[0].take_sequences:
            z_audio_tf = z_audio_tf.unsqueeze(1)
        z_audio_tf = audio_out['z_enc'][:-1] + [z_audio_tf]
        audio_tf = model[0].decode(z_audio_tf)
        symbol_tf = model[1].decode(audio_out['z_params_enc'][-1].mean)
        #audio_tf = model[0].decode(symbol_out['z_enc'][0])
        #symbol_tf = model[1].decode(audio_out['z_enc'][0])

    if reinforcers is not None:
        if reinforcers[0] is not None:
            audio_out = reinforcers[0].forward(audio_out)

    # compute transfers
    # WARNING here the conv module should work without *[0]

    n_examples = n_points
    n_symbols = len(model[1].pinput)
    grid = plt.GridSpec(n_examples*2, 3 * len(model[1].pinput))
    fig = plt.figure(figsize=(14,8))

    data_audio = data_audio_pp.cpu().detach().numpy()
    audio_out = audio_out['x_params'].mean.cpu().detach().numpy()
    audio_tf_out =audio_tf[0]['out_params'].mean.cpu().detach().numpy()
    is_image = False

    if preprocessing[0] is not None:
        data_audio = preprocessing[0].invert(data_audio)
        audio_out = preprocessing[0].invert(audio_out)
        audio_tf_out = preprocessing[0].invert(audio_tf_out)

    if len(audio_out.shape) > 2:
        if (audio_out.shape[1] == 1 and not model[0].take_sequences): 
            audio_out = np.squeeze(audio_out)
            audio_tf_out = np.squeeze(audio_tf_out)
        elif  (audio_out.shape[2]==1 and model[0].take_sequences):
            audio_out = np.squeeze(audio_out)
            audio_tf_out = np.squeeze(audio_tf_out)
            is_image = True
        else:
            is_image = True
    # data_symbol = [d.squeeze().cpu().detach().numpy() for d in data_symbol]

    for i in range(n_examples):
        # plot original signal
        ax1 = fig.add_subplot(grid[2*i, :n_symbols])
        if not is_image:
            ax1.plot(data_audio[i])
        else:
            ax1.imshow(data_audio[i], aspect='auto')
        # plot original symbols
        for l in range(n_symbols):
            current_ax = fig.add_subplot(grid[2*i+1, l])
            current_ax.plot(data_symbol[l][i])

        # reconstructed signals
        ax2 = fig.add_subplot(grid[2*i, n_symbols:2*n_symbols])
        if is_image:
            ax2.imshow(audio_out[i], aspect='auto')
        else:
            ax2.plot(data_audio[i], linewidth=0.5)
            ax2.plot(audio_out[i])
        # reconstructed labels
        for l in range(n_symbols):
            current_ax = fig.add_subplot(grid[2*i+1, n_symbols+l])
            current_ax.plot(data_symbol[l][i], linewidth=0.5)
            current_ax.plot(symbol_out['x_params'][l].probs[i].cpu().detach().numpy())
        # transferred data
        ax3 = fig.add_subplot(grid[2*i+1, 2*n_symbols:])
        if is_image:
            ax3.imshow(audio_tf_out[i], aspect='auto')
        else:
            ax3.plot(data_audio[i], linewidth=0.5)
            ax3.plot(audio_tf_out[i])
        for l in range(n_symbols):
            current_ax = fig.add_subplot(grid[2*i, 2*n_symbols+l])
            current_ax.plot(data_symbol[l][i], linewidth=0.5)
            current_ax.plot(symbol_tf[0]['out_params'][l].probs[i].cpu().detach().numpy())

    if out:
        fig.savefig(out+'.pdf', format='pdf')

    del audio_out; del symbol_out
    gc.collect(); gc.collect();

    return [fig], fig.axes


def plot_mx_reconstructions(datasets, model, solo_models=None, n_examples=3, out=None, random_mode='uniform'):
    ids = []; datas = []
    for d in datasets:
        ids.append( np.random.permutation(d.data.shape[0])[:n_examples] )
        datas.append(d.data[ids[-1]])
    if random_mode == 'uniform':
        random_weights = np.random.random((len(datasets), n_examples))
    elif random_mode == 'constant':
        random_weights = np.ones((len(datasets), n_examples))
    datas = np.array(datas)
    mixtures = np.sum( np.expand_dims(random_weights, -1) * datas, 0)

    vae_out = model.forward(mixtures)

    cmap = core.get_cmap(n_examples)
    fig, ax = plt.subplots(n_examples+1, len(datasets)+1, figsize=(20,10))
    if ax.ndim == 1:
        ax = np.array([ax])

    for i in range(n_examples):
        for j in range(len(datasets)):
            ax[i, j].plot(datas[j, i])
            ax[i, j].plot(vae_out['x_solo_params'][j].mean[i].detach().cpu().numpy(), linewidth=0.5)
            ax[i, j].set_title('weight : %f'%random_weights[j,i])

    for j in range(len(datasets)):
        z_projs = vae_out['z_params_enc'][0][j].mean.cpu().detach().numpy()
        ax[n_examples, j].scatter(z_projs[:, 0], z_projs[:, 1], c=cmap(np.arange(n_examples)))
        """
        if solo_models:
            z_projs_solo = solo_outs[j]['z_params_enc'][0][0].cpu().detach().numpy()
            ax[n_examples, j].scatter(z_projs_solo[:, 0], z_projs_solo[:, 1], marker = 'v', c=cmap(np.arange(n_examples)))
        """

    for i in range(n_examples):
        ax[i, len(datasets)].plot(mixtures[i])
        ax[i, len(datasets)].plot(vae_out['x_params'][i].detach().cpu().numpy())
        ax[i, len(datasets)].set_title('inferred weigths : %s'%vae_out['mixture_coeff'].mean[i, :])
#        if solo_models:
    if out:
        fig.savefig(out+'.pdf', format='pdf')

    del vae_out
    gc.collect(); gc.collect()
    return fig, ax


def plot_mx_latent_space(datasets, model, n_points=None, out=None, tasks=None):
    ids = []; datas = []
    for d in datasets:
        n_examples = n_points
        if n_examples is None:
            n_examples = d.data.shape[0]
        ids.append( np.random.permutation(d.data.shape[0])[:n_examples] )
        datas.append(d.data[ids[-1]])

    min_size = min([i.shape[0] for i in ids])
    ids = [i[:min_size] for i in ids]

    random_weights = np.random.random((len(datasets), min_size))
    datas = np.array(datas)
    mixtures = np.sum(np.expand_dims(random_weights, -1) * datas, 0)

    vae_out = model.forward(mixtures)

    z_out = vae_out['z_params_enc'][0]
    tasks = [None] if tasks is None else tasks

    for t in tasks:
        figs = []; axes = []
        fig, ax = plt.subplots(len(z_out), figsize=(10,10))
        if not issubclass(type(ax), np.ndarray):
            ax = (ax,)
        for i, z in enumerate(z_out):
            handles = []
            if not t is None:
                y = datasets[i].metadata[t].astype('int')[ids[i]]
                cmap = core.get_cmap(len(set(y)))
                colors = cmap(y)
                classes = datasets[i].classes.get(t)
                if not classes is None:
                    reverse_classes = {v:k for k,v in classes.items()}
                    for u in set(y):
                        patch = mpatches.Patch(color=cmap(u), label=reverse_classes[u])
                        handles.append(patch)

            else:
                colors = None
            z_tmp = z.mean.detach().cpu().numpy()
            s_tmp = np.max(z.variance.detach().cpu().numpy(), 1)
            ax[i].scatter(z_tmp[:, 0], z_tmp[:, 1], c=colors, s=15.0*s_tmp)

            if not len(handles) == 0:
                ax[i].legend(handles=handles)

        if not out is None:
            fig.savefig(out+'_%s.pdf'%t, format='pdf')
        figs.append(fig); axes.append(ax)

    return fig, ax


def get_spectral_transform(x, transform='fft', window_size=4096, mel_filters=256):
    if len(x.shape) == 2:
        x = x.unsqueeze(1)
    if transform in ('stft', 'stft-mel'):
        x_fft = torch.stft(x.squeeze(), window=torch.hann_window(window_size, device=x.device), center=True, pad_mode='constant')
        x_fft_real = x_fft[:,:,:, 0]; x_fft_imag = x_fft[:,:,:, 1];
    elif transform in ('fft', 'fft-mel'):
        x_fft = torch.fft(torch.cat([x.squeeze().unsqueeze(-1), torch.zeros_like(x.squeeze().unsqueeze(-1))], dim=-1), 2)
        x_fft_real = x_fft.select(-1, 0); x_fft_imag = x_fft.select(-1, 1);
        x_fft_real = x_fft_real[:, :int(x_fft_real.shape[1]/2+1)];
        x_fft_imag = x_fft_imag[:, :int(x_fft_imag.shape[1]/2+1)];
        window_size = x_fft_real.shape[1]*2

    x_radius = torch.sqrt(x_fft_real.pow(2) + x_fft_imag.pow(2))
    x_angle = torch.atan2(x_fft_real, x_fft_imag+eps)
    if transform in ("stft-mel", 'fft-mel'):
        mel_w = librosa.filters.mel(22050, window_size-1, n_mels = min(mel_filters, window_size))
        mel_weights = torch.from_numpy(mel_w).float().to(x_fft).detach()
        x_radius = torch.bmm(mel_weights.unsqueeze(0).repeat(x_radius.shape[0],1,1), x_radius.unsqueeze(-1)).transpose(1,2)
    return x_radius, x_angle, x_fft_real, x_fft_imag

def plot_spectrograms(dataset, model, label=None, n_points=10, out=None, preprocessing=None, partition=None, ids=None,
                      transform="fft", window_size=2048, mel_filters=256, sample=False, plot_multihead=False):
    # get plotting ids

    if not torch.backends.mkl.is_available():
        print('Error in plot spectrograms : MKL backend not available.')
        return

    n_rows, n_columns = core.get_divs(n_points)
    if ids is None:
        if issubclass(type(dataset.data), list):
            full_id_list = np.array(range(dataset.data[0].shape[0])) if partition is None else np.array(range(len(dataset.partitions[partition])))
            ids = full_id_list[np.random.permutation(len(full_id_list))[:n_points]] if ids is None else ids
        else:
            full_id_list = np.array(range(dataset.data.shape[0])) if partition is None else np.array(range(len(dataset.partitions[partition])))
            ids = full_id_list[np.random.permutation(len(full_id_list))[:n_points]] if ids is None else ids

    # get data
    loader = DataLoader(dataset, None, ids=ids, is_sequence=model.take_sequences)
    data, _ = next(loader.__iter__())
    if issubclass(type(model.pinput), list):
        # if not issubclass(type(dataset.data), list):
        #     data = [dataset.data]
        # data = [d[ids] for d in data]
        preprocessing = preprocessing if not preprocessing is None else [None]*len(dataset.data)
        if not issubclass(type(preprocessing), list):
            preprocessing = [preprocessing]
        for i, pp in enumerate(preprocessing):
            if not pp is None:
                data[i] = preprocessing(data[i])
        else:
            data = [dataset.data[ids]]
            preprocessing = [preprocessing]
    else:
        if preprocessing is not None:
            data = preprocessing(data)

    # in case, get metadata
    metadata = None
    if not label is None:
        if not issubclass(type(label), list):
            label = [label]
        if not label is None:
            metadata = {l: dataset.metadata.get(l)[ids] for l in label}
    else:
        metadata = None

    # forward
    add_args = {}
    if hasattr(model, 'prediction_params'):
        add_args['n_preds'] = model.prediction_params['n_predictions']
    with torch.no_grad():
        vae_out = model.forward(data, y=metadata, **add_args)['x_params']


    if not issubclass(type(vae_out), list):
        vae_out = [vae_out]
    if not issubclass(type(data), list):
        data = [data]

    # plot
    figs = []; axes = []
    nrows, ncolumns = core.get_divs(n_points)
    for i in range(len(vae_out)):
        # multihead_outputs = None
        # if plot_multihead and hasattr(model.decoders[0].out_modules[0], 'current_outs'):
        #     multihead_outputs = model.decoders[0].out_modules[0].current_outs
        if sample:
            current_out = vae_out[i].sample()
        else:
            current_out = vae_out[i].mean
        if preprocessing:
            current_out = preprocessing.invert(current_out)

        spec_orig = get_spectral_transform(data[i], window_size=window_size, mel_filters=mel_filters)
        spec_rec = get_spectral_transform(current_out)
        fig, ax = plt.subplots(nrows, ncolumns)
        if transform in ('fft', 'fft-mel'):
            for j in range(nrows):
                for k in range(ncolumns):
                    ax[j, k].plot(spec_orig[j*ncolumns+k])
                    ax[j, k].plot(spec_rec[j*ncolumns+k])

        if not out is None:
            fig.suptitle('output %d'%i)
            name = out+'.pdf' if len(vae_out) == 1 else out+'_%d.pdf'%i
            fig.savefig(name, format="pdf")
        figs.append(fig); axes.append(ax)

    del data; del vae_out
    return figs, axes

        
################################################
########        LATENT PLOTS
####


@partition_dependent
def plot_latent(dataset, model, transformation=None, n_points=None, preprocessing=None, label=None, tasks=None, ids=None, balanced=False, batch_size=None, partition=None, epoch=None,
                   preprocess=True, loader=None, sample = False, layers=None, target_dim=3, zoom=10, out=None, name=None, legend=True, centroids=False, prediction=None, trainer=None, *args, **kwargs):
    '''
    3-D plots the latent space of a model
    :param dataset: `vschaos.data.Dataset` object containing the data
    :param model: `vschaos.vaes.AbstractVAE` child
    :param transformation: `vschaos.visualize_dimred.Embedding` object used to project on 3 dimensions (if needed)
    :param n_points: number of points plotted
    :param preprocessing: preprocessing used before encoding
    :param label: conditioning data
    :param tasks: tasks plotted (None for no task-related coloring)
    :param ids: plot given ids from dataset (overrides n_points and balanced options)
    :param classes:
    :param balanced: balance data for each task
    :param batch_size: batch size of the loader
    :param preprocess:
    :param loader: class of the data loader used
    :param sample: sample from distribution (if `False`: takes the mean)
    :param layers: latent layers to plot (default : all)
    :param color_map: color map used for coloring (default: plasma)
    :param zoom: weight of each point radius (default: 10)
    :param out: if given, save plots at corresponding places
    :param legend: plots legend if `True`
    :param centroids: plots centroids for each class
    :return: list(figure), list(axis)
    '''

    ### prepare data IDs
    tasks = checklist(tasks)
    # check as list
    if len(tasks) == 0 or tasks is None:
        tasks = [None] 
    full_ids = CollapsedIds()
    if ids is not None:
        dataset = dataset.retrieve(ids)
    # retrieve partition in case
    if partition:
        dataset = dataset.retrieve(partition)
    # retrieve points
    if n_points is not None:
        dataset = dataset.retrieve(np.random.permutation(len(dataset.data))[:n_points])
     # if no tasks are given, add indices as None
    if tasks == [None]:
        full_ids.add(None, ids if ids is not None else range(len(dataset)))
        nclasses = {None:[None]}; class_ids = {None:{None:full_ids.get_full_ids()}}
    if tasks != [None] and tasks != None:
        # fill full_ids objects with corresponding classes, get classes index hash
        class_ids = {}; nclasses = {}
        for t in tasks:
            class_ids[t], nclasses[t] = core.get_class_ids(dataset, t, balanced=balanced, ids=ids, split=True)
            full_ids.add(t,np.concatenate(list(class_ids[t].values())))

    ### forward
    if not issubclass(type(label), list) and not label is None:
        label = [label]
    # preparing dataloader
    Loader = loader or DataLoader
    loader_ids = full_ids.get_full_ids()
    loader = Loader(dataset, batch_size, ids=loader_ids, tasks = label, shuffle=False)
    # forward!
    output = []; full_y = []
    with torch.no_grad():
        for x,y in loader:
            full_y.append(y)
            output.append(decudify(model.encode(model.format_input_data(x), y=y, *args, **kwargs)))
    full_y = merge_dicts(full_y)
    torch.cuda.empty_cache()
    vae_out = merge_dicts(output)

    ### plot!
    figs = {}; axes = {}
    layers = layers or range(len(model.platent))
    if out:
        out += "/latent"
        check_dir(out)
    # iterate over layers
    suffix = ""
    suffix = "_"+str(partition) if partition is not None else suffix
    suffix = suffix+"_%d"%epoch if epoch is not None else suffix

    for layer in layers:
        # iteration over tasks
        for task in tasks:
            print('-- plotting task %s'%task)
            is_multiclass = None

            task_ids = {k: full_ids.transform(v) for k, v in class_ids[task].items()}
            if task:
                meta = full_y[task][full_ids.get_ids(task)]
                is_multiclass = hasattr(meta[0], '__iter__')
            else:
                meta = None

            # if is_multiclass:
            #     for k, v in class_ids[task].items():
            #         class_names = {v:k for k, v in dataset.classes[task].items()}
            #         current_ids = full_ids.transform(v)
            #         ghost_ids = np.array(list(filter(lambda x: x not in current_ids, range(full_z.shape[0]))))
            #         fig, ax = core.plot_latent(full_z[current_ids], meta=[k]*len(v), var=full_var[full_ids.transform(v)], classes=nclasses[task], class_ids={k:v}, class_names=class_names[k], centroids=centroids,legend=False, sequence=sequence, shadow_z =full_z[ghost_ids])
            #         # register and export
            #         fig_name = 'layer %d / task %s / class %s'%(layer, task, class_names[k]) if task else 'layer%d'%layer
            #         fig.suptitle(fig_name)
            #         name = name or 'latent'
            #         title = '%s_layer%d_%s_%s'%(name,layer, task, class_names[k])
            #         figs[title] = fig; axes[title] = ax
            #         if not out is None:
            #             title = '%s%s.pdf'%(out, title)
            #             fig.savefig(title, format="pdf")
            # else:
            class_names = {} if task is None else {v:k for k, v in dataset.classes[task].items()}
            legend = False if (task is None) or len(class_names.keys())>20 else legend
            current_transform = transformation if not isinstance(transformation, list) else transformation[layer]

            if isinstance(vae_out[layer]['out_params'], list):
                for i in range(len(vae_out[layer]['out_params'])):
                    out_tmp = {'out':vae_out[layer]['out'][i][full_ids.get_ids(task)],
                               'out_params':vae_out[layer]['out_params'][i][full_ids.get_ids(task)]}
                    sequence = out_tmp['out'].ndim > 2
                    fig, ax = core.plot_latent(out_tmp, meta=meta, classes=nclasses[task], class_ids=task_ids, class_names=class_names, centroids=centroids, legend=legend, sequence=sequence, target_dim=target_dim, sample=sample, transformation=current_transform)
                    title = '%slayer%d_%d_%s'%(name or 'latent', layer, i, task)
                    figs[title] = fig; axes[title] = ax
                    if not out is None:
                        fig.savefig(f"{out}/{title}{suffix}.pdf", format="pdf")
            else:
                out_tmp = {'out': vae_out[layer]['out'][full_ids.get_ids(task)],
                           'out_params': vae_out[layer]['out_params'][full_ids.get_ids(task)]}
                sequence = out_tmp['out'].ndim > 2
                fig, ax = core.plot_latent(out_tmp, meta=meta, classes=nclasses[task], class_ids=task_ids, class_names=class_names, centroids=centroids, legend=legend, sequence=sequence, target_dim=target_dim, sample=sample, transformation=current_transform)
                title = '%slayer%d_%s'%(name or 'latent', layer, task)
                figs[title] = fig; axes[title] = ax
                if not out is None:
                    fig.savefig(f"{out}/{title}{suffix}.pdf", format="pdf")
    gc.collect(); gc.collect()
    return figs, axes

@partition_dependent
def plot_latent_dim(dataset, model, label=None, tasks=None, n_points=None, layers=None, legend=True, out=None, ids=None, transformation=None, name=None,
                    partition=None, epoch=None, preprocess=True, loader=None, batch_size=None, balanced=True, preprocessing=None, sample=False, *args, **kwargs):
    ### prepare data IDs
    tasks = checklist(tasks)
    if len(tasks) == 0 or tasks is None:
        tasks = [None]
    full_ids = CollapsedIds()
    if partition:
        dataset = dataset.retrieve(partition)
    if n_points is not None:
        dataset = dataset.retrieve(np.random.permutation(len(dataset.data))[:n_points])

    if tasks == [None]:
        full_ids.add(None, ids if ids is not None else np.random.permutation(len(dataset.data))[:n_points])
        class_ids = {None:{None:full_ids.get_full_ids()}}; nclasses = {None:[]}
    else:
        # if n_points is not None:
        #    ids = np.random.permutation(len(dataset.data))[:n_points]
        class_ids = {}
        nclasses = {}
        for t in tasks:
            class_ids[t], nclasses[t] = core.get_class_ids(dataset, t, balanced=balanced, ids=ids, split=True)
            full_ids.add(t, np.concatenate(list(class_ids[t].values())))

    ### forwarding
    if not issubclass(type(label), list) and not label is None:
        label = [label]
    # preparing dataloader
    Loader = loader or DataLoader
    loader = Loader(dataset, batch_size, ids=full_ids.get_full_ids(), tasks=label, shuffle=False)
    # forward!
    output = []; full_y = []
    with torch.no_grad():
        for x, y in loader:
            full_y.append(y)
            output.append(
                decudify(model.encode(model.format_input_data(x), y=y)))
    torch.cuda.empty_cache()
    vae_out = merge_dicts(output)
    full_y = merge_dicts(full_y)

    ### plot!
    figs = {}; axes = {}
    layers = layers or list(range(len(model.platent)))
    transformation = checklist(transformation)
    if out:
        out += "/dims"
        check_dir(out)
    # iterate over layers
    for layer in layers:
        # sample = True -> sample distirbution; sample = False -> take mean of distribution
        vae_out[layer]['out'] = checklist(vae_out[layer]['out'])
        vae_out[layer]['out_params'] = checklist(vae_out[layer]['out_params'])
        if sample:
            full_z = np.concatenate([x.cpu().detach().numpy() for x in vae_out[layer]['out']], axis=-1)
        else:
            full_z = np.concatenate([x.mean.cpu().detach().numpy() for x in vae_out[layer]['out_params']], axis=0)
        full_var = np.concatenate([None if not hasattr(x, "stddev") else x.stddev for x in vae_out[layer]['out_params']], axis=0)
        # transform in case
        for reduction in transformation:
            full_z_t = full_z
            if reduction:
                if full_z.shape[-1] > 3:
                    if issubclass(type(reduction), list):
                        reduction = reduction[layer]
                    if issubclass(type(reduction), type):
                        full_z_t = reduction(n_components=3).fit_transform(full_z)
                    else:
                        full_z_t = reduction.transform(full_z)

            # iteration over tasks
            for task in tasks:
                if task:
                    meta = full_y[task][full_ids.ids[task]]
                    class_names = {v: k for k, v in dataset.classes[task].items()}
                else:
                    meta = None
                    class_ids = {None: None}; class_names = None
                fig, ax = core.plot_dims(full_z_t[full_ids.get_ids(task)], meta=meta, classes=nclasses[task],
                                         class_ids=class_ids, class_names=class_names, legend=legend, var=full_var[full_ids.get_ids(task)])

                # register and export
                fig_name = 'layer %d / task %s' % (layer, task) if task else 'layer%d' % layer
                fig.suptitle(fig_name)

                name = name or 'dims'
                suffix = ""
                suffix = "_"+str(partition) if partition is not None else suffix
                suffix = suffix+"_%d"%epoch if epoch is not None else suffix

                title = '%s_layer%d_%s_%s' % (name, layer, str(reduction), task)
                if not out is None:
                    fig.savefig(f'{out}/{title}{suffix}.pdf', format="pdf")
                figs[title] = fig; axes[title] = ax

    return figs, axes


@partition_dependent
def plot_latent_consistency(dataset, model, label=None, tasks=None, n_points=None, layers=None, legend=True, out=None, ids=None, transformation=None, name=None,
                     epoch=None, preprocess=True, loader=None, batch_size=None, partition=None, preprocessing=None, sample=False, *args, **kwargs):

    assert len(model.platent) > 1, "plot_latent_consistency is only made for hierarchical models"
    # get plotting ids
    if partition is not None:
        dataset = dataset.retrieve(partition)
    n_rows, n_columns = core.get_divs(n_points)
    if ids is None:
        full_id_list = np.arange(len(dataset))
        ids = full_id_list[np.random.permutation(len(full_id_list))[:n_points]] if ids is None else ids

    ### forwarding
    if not issubclass(type(label), list) and not label is None:
        label = [label]
    # preparing dataloader
    Loader = loader or DataLoader
    loader = Loader(dataset, batch_size, ids=ids, tasks=label, shuffle=False)
    # forward!
    output = []
    with torch.no_grad():
        for x, y in loader:
            if not preprocessing is None and preprocess:
                x = preprocessing(x)
            out_tmp = model.forward(x, y=y, **kwargs)
            output.append(decudify(out_tmp))
    torch.cuda.empty_cache()
    vae_out = merge_dicts(output)

    ### plot!
    figs = {}; axes = {}
    layers = layers or list(range(len(model.platent)))
    transformation = checklist(transformation)
    if out:
        out += "/consistency"
        check_dir(out)
    # iterate over layers
    for layer in layers:
        # sample = True -> sample distirbution; sample = False -> take mean of distribution
        # transform in case
        for reduction in transformation:
            if layer >= len(vae_out['z_params_dec']):
                continue
            z_enc_params = vae_out['z_params_enc'][layer]; z_dec_params = vae_out['z_params_dec'][layer]
            full_z_enc = z_enc_params.mean; full_z_dec = z_dec_params.mean
            full_z_var_enc = None if not hasattr(z_enc_params, "stddev") else z_enc_params.stddev.detach().cpu().numpy()
            full_z_var_dec = None if not hasattr(z_dec_params, "stddev") else z_dec_params.stddev.detach().cpu().numpy()
            if reduction:
                if issubclass(type(reduction), list):
                    reduction = reduction[layer]
                if issubclass(type(reduction), type):
                    reduction = reduction(n_components=3).fit(np.concatenate([full_z_enc, full_z_dec], axis=0))
                    full_z_enc = reduction(n_components=3).transform(full_z_enc)
                    full_z_dec = reduction(n_components=3).transform(full_z_dec)
                else:
                    full_z_enc = reduction.transform(full_z_enc)
                    full_z_dec = reduction.transform(full_z_dec)

            full_z_enc = full_z_enc.detach().cpu().numpy(); full_z_dec = full_z_dec.detach().cpu().numpy()
            # iteration over tasks
            fig, ax = core.plot_pairwise_trajs([full_z_enc, full_z_dec], var=[full_z_var_enc, full_z_var_dec])

            # register and export
            for i in range(len(ids)):
                fig_name = dataset.files[ids[i]] or "consist_%d"%i
                fig[i].suptitle(fig_name)
                name = name or 'dims'
                title = '/%s_%s_%s_%s'%(name, str(reduction), layer, i)
                suffix = ""
                suffix = "_"+str(partition) if partition is not None else suffix
                suffix = suffix+"_%d"%epoch if epoch is not None else suffix
                if not out is None:
                    fig[i].savefig(f"{out}/{title}{suffix}.pdf", format="pdf")
                figs[title] = fig[i]; axes[title] = ax[i]

    return figs, axes


valid_stats_dists = (Normal,)
def get_statistics_figs(zs, task_ids, class_names, legend=True, title="latent statistics"):
    plt.figure()
    fig, ax = plt.subplots(2, 1)
    ax[0].set_title('variance of latent positions')
    ax[1].set_title('mean of variances per axis')
    # get data
    pos_var = []; var_mean= [];
    n_classes = len(task_ids.keys())
    width = 1/n_classes
    cmap = core.get_cmap(n_classes)
    handles = []; counter=0

    id_range = np.arange(zs.mean.shape[-1])
    for i, c in task_ids.items():
        reg_axis = tuple(np.arange(len(zs.mean.shape)-1))
        pos_var.append(np.std(zs.mean[c].cpu().detach().numpy(), reg_axis))
        var_mean.append(np.mean(zs.variance[c].cpu().detach().numpy(), reg_axis))
        ax[0].bar(id_range+counter*width, pos_var[-1], width, color=cmap(counter))
        ax[1].bar(id_range+counter*width, var_mean[-1], width, color=cmap(counter))
        if legend:
            patch = mpatches.Patch(color=cmap(counter), label=str(class_names[counter]))
            handles.append(patch)
        counter += 1
    if legend:
        fig.legend(handles=handles)
    return fig


@partition_dependent
def plot_latent_stats(dataset, model, label=None, tasks=None, n_points=None, layers=None, legend=True, out=None, preprocess=True, ids=None,
                      loader=None, epoch=None, partition=None, batch_size=None, balanced=False, preprocessing=None, *args, **kwargs):

    tasks = checklist(tasks)
    # check as list
    if len(tasks) == 0 or tasks is None:
        tasks = [None]
    full_ids = CollapsedIds()
    if ids is not None:
        dataset = dataset.retrieve(ids)
    # retrieve partition in case
    if partition:
        dataset = dataset.retrieve(partition)
    # retrieve points
    if n_points is not None:
        dataset = dataset.retrieve(np.random.permutation(len(dataset.data))[:n_points])
     # if no tasks are given, add indices as None
    if tasks == [None]:
        full_ids.add(None, ids if ids is not None else range(len(dataset)))
        nclasses = {None:[None]}; class_ids = {None:{0:full_ids.get_full_ids()}}
    else:
        # fill full_ids objects with corresponding classes, get classes index hash
        class_ids = {}; nclasses = {}
        for t in tasks:
            class_ids[t], nclasses[t] = core.get_class_ids(dataset, t, balanced=balanced, ids=ids, split=True)
            full_ids.add(t,np.concatenate(list(class_ids[t].values())))

    ### forward
    if not issubclass(type(label), list) and not label is None:
        label = [label]
    # preparing dataloader
    Loader = loader or DataLoader
    loader_ids = full_ids.get_full_ids()
    loader = Loader(dataset, batch_size, ids=loader_ids, tasks = label, shuffle=False)
    # forward!
    output = []; full_y = []
    with torch.no_grad():
        for x,y in loader:
            full_y.append(y)
            output.append(decudify(model.encode(model.format_input_data(x), y=y)))
    torch.cuda.empty_cache()
    vae_out = merge_dicts(output)
    full_y = merge_dicts(full_y)

    figs = {}; axes = {}
    if out:
        out += "/stats/"
        check_dir(out)

    layers = layers or range(len(model.platent))
    for layer in layers:
        for task in tasks:
            task_ids = {k:full_ids.transform(v) for k, v in class_ids[task].items()}
            zs = vae_out[layer]
            if isinstance(zs['out_params'], list):
                for sublayer in range(len(zs['out_params'])):
                    if not issubclass(type(zs['out_params'][sublayer]), valid_stats_dists):
                        continue
                    title = 'stats_layer%d_%s.pdf'%(layer, task)
                    fig = get_statistics_figs(zs['out_params'][sublayer], task_ids, nclasses[task], legend=legend, title=title)
                    title = 'stats_layer%d_%d_%s.pdf'%(layer, sublayer, task)
                    figs[title] = fig; axes[title] = fig.axes
                    suffix = ""
                    suffix = "_"+str(partition) if partition is not None else suffix
                    suffix = suffix+"_%d"%epoch if epoch is not None else suffix
                    if not out is None:
                        fig.savefig(f"{out}/{title}{suffix}.pdf", format="pdf")
            else:
                if not issubclass(type(zs['out_params']), valid_stats_dists):
                    continue
                title = 'stats_layer%d_%s.pdf'%(layer, task)
                fig = get_statistics_figs(zs['out_params'], task_ids, nclasses[task], legend=legend, title=title)
                title = 'stats_layer%d_%s.pdf'%(layer, task)
                figs[title] = fig; axes[title] = fig.axes
                suffix = ""
                suffix = "_"+str(partition) if partition is not None else suffix
                suffix = suffix+"_%d"%epoch if epoch is not None else suffix
                if not out is None:
                    fig.savefig(f"{out}/{title}{suffix}.pdf", format="pdf")
            plt.close('all')

    # plot histograms
    return figs, axes
        
@partition_dependent
def plot_latent_dists(dataset, model, label=None, tasks=None, bins=20, layers=[0], n_points=None, dims=None, legend=True, split=False, out=None, relief=True, ids=None, **kwargs):
    # get data ids
    if n_points is None:
        ids = np.arange(dataset.data.shape[0]) if ids is None else ids
        data = dataset.data
        y = dataset.metadata.get(label)
    else:
        ids = np.random.permutation(dataset.data.shape[0])[:n_points] if ids is None else ids
        data = dataset.data[ids]
        y = dataset.metadata.get(label)
        if not y is None:
            y = y[ids]
    y = model.format_label_data(y)
    data = model.format_input_data(data);
    
    if dims is None:
        dims = list(range(model.platent[layer]['dim']))
        
    # get latent space
    with torch.no_grad():
        vae_out = model.encode(data, y=y)
        # get latent means of corresponding parameters
    
    # get  
    figs = []
    
    for layer in layers:
        zs = model.platent[layer]['dist'](*vae_out[0][layer]).mean.cpu().detach().numpy()
        if split:
            if tasks is None:
                for dim in dims:
                    fig = plt.figure('dim %d'%dim, figsize=(20,10))
                    hist, edges = np.histogram(zs[:, dim], bins=bins)
                    plt.bar(edges[:-1], hist, edges[1:] - edges[:-1], align='edge')
                    if not out is None:
                        prefix = out.split('/')[:-1]
                        fig.savefig(prefix+'/dists/'+out.split('/')[-1]+'_%d_dim%d.svg'%(layer, dim))
                    figs.append(fig)
            else:
                if not os.path.isdir(out+'/dists'):
                    os.makedirs(out+'/dists')
                for t in range(len(tasks)):
                    class_ids, classes = get_class_ids(dataset, tasks[t], ids=ids)
                    cmap = get_cmap(len(class_ids))
                    for dim in dims:
                        fig = plt.figure('dim %d'%dim, figsize=(20, 10))
                        ax = fig.gca(projection='3d') if relief else fig.gca()
                        for k, cl in enumerate(class_ids):
                            hist, edges = np.histogram(zs[cl, dim], bins=bins)
                            colors = cmap(k)
                            if relief:
                                ax.bar3d(edges[:-1], k*np.ones_like(hist), np.zeros_like(hist), edges[1:]-edges[:-1], np.ones_like(hist), hist, color=colors)
                                ax.view_init(30,30)
                            else:
                                ax.bar(edges[:-1], hist, edges[1:] - edges[:-1], align='edge')
                        if legend and not dataset.classes.get(tasks[t]) is None:
                            handles = []
                            class_names = {v: k for k, v in dataset.classes[tasks[t]].items()}
                            for i in classes:
                                patch = mpatches.Patch(color=cmap(i), label=class_names[i])
                                handles.append(patch)
                            fig.legend(handles=handles)
                        if not out is None:
                            prefix = out.split('/')[:-1]
                            fig.savefig('/'.join(prefix)+'/dists/'+out.split('/')[-1]+'_%d_%s_dim%d.svg'%(layer,tasks[t], dim))
    #                    plt.close('all')
                        figs.append(fig)
        else:
            if tasks is None:
                dim1, dim2 = get_divs(len(dims))
                fig, axes = plt.subplots(dim1, dim2, figsize=(20,10))
                for i in range(axes.shape[0]):
                    for j in range(axes.shape[1]):
                        current_id = i*dim2 + j
                        hist, edges = np.histogram(zs[:, dims[current_id]], bins=bins)
                        axes[i,j].bar(edges[:-1], hist, edges[1:] - edges[:-1], align='edge')
                        axes[i,j].set_title('axis %d'%dims[current_id])
                if not out is None:
                    prefix = out.split('/')[:-1]
                    fig.savefig(out+'_0.svg'%layer)
                figs.append(fig)
            else:
                dim1, dim2 = get_divs(len(dims))
                for t in range(len(tasks)):
                    class_ids, classes = get_class_ids(dataset, tasks[t], ids=ids)
                    cmap = get_cmap(len(class_ids))
                    if relief:
                        fig, axes = plt.subplots(dim1, dim2, figsize=(20,10), subplot_kw={'projection':'3d'})
                    else:
                        fig, axes = plt.subplots(dim1, dim2, figsize=(20,10))
                        
    #                pdb.set_trace()
                    for i in range(axes.shape[0]):
                        dim_y = 0 if len(axes.shape)==1 else axes.shape[1]
                        for j in range(dim_y):
                            current_id = i*dim2 + j
                            for k, cl in enumerate(class_ids):
                                hist, edges = np.histogram(zs[cl, dims[current_id]], bins=bins)
                                colors = cmap(k)
                                if relief:
                                    axes[i,j].bar3d(edges[:-1], k*np.ones_like(hist), np.zeros_like(hist), edges[1:]-edges[:-1], np.ones_like(hist), hist, color=colors, alpha=0.1)
                                    axes[i,j].view_init(30,30)
                                else:
                                    axes[i,j].bar(edges[:-1], hist, edges[1:] - edges[:-1], align='edge')
                                axes[i,j].set_title('axis %d'%dims[current_id])
                            
                    if legend and not dataset.classes.get(tasks[t]) is None:
                        handles = []
                        class_names = {v: k for k, v in dataset.classes[tasks[t]].items()}
                        for i in classes:
                            patch = mpatches.Patch(color=cmap(i), label=class_names[i])
                            handles.append(patch)
                        fig.legend(handles=handles)
    
                    if not out is None:
                        prefix = out.split('/')[:-1]
                        fig.savefig(out+'_%d_%s.svg'%(layer, tasks[t]))
                    figs.append(fig)
    return figs


@partition_dependent
def plot_latent_trajs(dataset, model, n_points=None, preprocessing=None, label=None, tasks=None, balanced=False, batch_size=None, ids=None,
                   partition=None, epoch=None, preprocess=True, loader=None, sample = False, layers=None, out=None, name=None, plot_var=True, legend=True, centroids=False, *args, **kwargs):
    '''
    3-D plots the latent space of a model
    :param dataset: `vschaos.data.Dataset` object containing the data
    :param model: `vschaos.vaes.AbstractVAE` child
    :param transformation: `vschaos.visualize_dimred.Embedding` object used to project on 3 dimensions (if needed)
    :param n_points: number of points plotted
    :param preprocessing: preprocessing used before encoding
    :param label: conditioning data
    :param tasks: tasks plotted (None for no task-related coloring)
    :param ids: plot given ids from dataset (overrides n_points and balanced options)
    :param classes:
    :param balanced: balance data for each task
    :param batch_size: batch size of the loader
    :param preprocess:
    :param loader: class of the data loader used
    :param sample: sample from distribution (if `False`: takes the mean)
    :param layers: latent layers to plot (default : all)
    :param color_map: color map used for coloring (default: plasma)
    :param zoom: weight of each point radius (default: 10)
    :param out: if given, save plots at corresponding places
    :param legend: plots legend if `True`
    :param centroids: plots centroids for each class
    :return: list(figure), list(axis)
    '''

    tasks = checklist(tasks)
    if len(tasks) == 0 or tasks is None:
        raise TypeError('tasks keyword must be given for function plot_latent_traj')
    full_ids = CollapsedIds()
    if partition:
        dataset = dataset.retrieve(partition)
    if n_points is not None:
        dataset = dataset.retrieve(np.random.permutation(len(dataset.data))[:n_points])

    if tasks == [None]:
        full_ids.add(None, ids if ids is not None else np.random.permutation(len(dataset.data))[:n_points])
        class_ids = {None:None}; nclasses = {None:[]}
    else:
        # if n_points is not None:
        #    ids = np.random.permutation(len(dataset.data))[:n_points]
        class_ids = {};
        nclasses = {}
        for t in tasks:
            class_ids[t], nclasses[t] = core.get_class_ids(dataset, t, balanced=balanced, ids=ids, split=True)
            full_ids.add(t, np.concatenate(list(class_ids[t].values())))

    ### forwarding
    if not issubclass(type(label), list) and not label is None:
        label = [label]
    # preparing dataloader
    Loader = loader or DataLoader
    loader = Loader(dataset, batch_size, ids=full_ids.get_full_ids(), tasks=label, shuffle=False)
    # forward!
    output = []
    with torch.no_grad():
        for x, y in loader:
            if not preprocessing is None and preprocess:
                x = preprocessing(x)
            output.append(
                decudify(model.encode(model.format_input_data(x), y=y, return_shifts=False, *args, **kwargs)))
    torch.cuda.empty_cache()
    vae_out = merge_dicts(output)

    ### plot!
    figs = {}; axes = {}
    layers = layers or range(len(model.platent))
    if out:
        out += "/latent_trajs"
        check_dir(out)

    # iterate over layers
    for layer in layers:
        # sample = True -> sample distirbution; sample = False -> take mean of distribution
        vae_out[layer]['out'] = checklist(vae_out[layer]['out'])
        vae_out[layer]['out_params'] = checklist(vae_out[layer]['out_params'])
        if sample:
            full_z = np.concatenate([x.cpu().detach().numpy() for x in vae_out[layer]['out']], axis=-1)
            full_var = np.ones_like(full_z) * 1e-3
        else:
            full_z = np.concatenate([x.mean.cpu().detach().numpy() for x in vae_out[layer]['out_params']], axis=0)
            full_var = np.concatenate([x.variance.cpu().detach().numpy() for x in vae_out[layer]['out_params']], axis=0)

        # iteration over tasks
        for task in tasks:
            print('-- plotting task %s'%task)
            if task is None:
                class_names = {None:None}
            else:
                class_names = {v: k for k, v in dataset.classes[task].items()}

            full_z_sorted = [full_z[full_ids.transform(class_ids[task][i])] for i in nclasses[task]]
            full_var_sorted = [full_var[full_ids.transform(class_ids[task][i])] for i in nclasses[task]]
            if len(full_z_sorted[0].shape) > 2:
                full_z_sorted = [fs.reshape(fs.shape[0]*fs.shape[1], fs.shape[2]) for fs in full_z_sorted]
                full_var_sorted = [fs.reshape(fs.shape[0]*fs.shape[1], fs.shape[2]) for fs in full_var_sorted]
            full_z_sorted = np.array(full_z_sorted)
            full_var_sorted = np.array(full_var_sorted)

            n_rows, n_columns = core.get_divs(full_z_sorted.shape[-1])
            fig, axis = plt.subplots(n_rows, n_columns, figsize=(10,10))
            if n_rows == 1:
                axis = axis[np.newaxis, :]
            if n_columns==1:
                axis = axis[:, np.newaxis]
            plt.gca()
            for i in range(n_rows):
                for j in range(n_columns):
                    x = np.arange(len(nclasses[task]))

                    # draw means
                    if plot_var:
                        current_var = full_var_sorted.transpose(0, -1, 1).mean(-1)
                        axis[i,j].bar(x, current_var[:, i*n_columns+j], alpha=0.3, color='r')
                        axis[i,j].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0e}'))
                        axis[i,j].yaxis.set_ticks_position('right')
                        axis[i,j].yaxis.set_tick_params(labelsize='x-small')
                        axis[i,j].yaxis.tick_right()
                        if i < n_rows - 1:
                            axis[i,j].xaxis.set_ticks_position('none')
                        else:
                            plt.xticks(x, [class_names[c] for c in nclasses[task]])
                            axis[i,j].xaxis.set_tick_params(labelsize="x-small", rotation=45)
                    # draw variances
                    axis_b = axis[i,j].twinx()
                    current_z = full_z_sorted.transpose(0, -1, 1).mean(-1)
                    current_z_var = full_z_sorted.transpose(0, -1, 1).std(-1)
                    axis_b.plot(x, current_z[:, i*n_columns+j]+current_var[:, i*n_columns+j], linewidth=0.3, color='r')
                    axis_b.plot(x, current_z[:, i*n_columns+j]-current_var[:, i*n_columns+j], linewidth=0.3, color='r')
                    axis_b.fill_between(x, current_z[:, i*n_columns+j]-current_z_var[:, i*n_columns+j],
                                           current_z[:, i*n_columns+j]+current_z_var[:, i*n_columns+j], color='r', alpha=0.3)
                    axis_b.plot(x, current_z[:, i*n_columns+j], color='r')
                    axis_b.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
                    axis_b.yaxis.set_tick_params(labelsize='x-small')
                    axis_b.yaxis.tick_left()
                    if i < n_rows - 1:
                        axis_b.xaxis.set_ticks_position('none')
                    else:
                        plt.xticks(x, [class_names[c] for c in nclasses[task]])
                        axis_b.xaxis.set_tick_params(labelsize="x-small", rotation=45)

            # register and export
            fig_name = 'layer %d / task %s'%(layer, task)  if task else 'layer%d'%layer
            fig.suptitle(fig_name)

            name = name or 'trajs'
            title = "%s_layer%d_%s"%(name,layer, task)
            figs[title] = fig; axes[title] = axis
            suffix = ""
            suffix = "_"+str(partition) if partition is not None else suffix
            suffix = suffix+"_%d"%epoch if epoch is not None else suffix
            if not out is None:
                title = f"{out}/{title}{suffix}.pdf"
                fig.savefig(title, format="pdf")

    gc.collect(); gc.collect()
    return figs, axes





################################################
########        LOSSES PLOTS
####

@partition_independent
def plot_losses(*args, loss=None, out=None, separated=False, axis="time", partition=None, epoch=None, **kwargs):
    assert loss
    assert axis in ['time', 'epochs']
    # get set and loss names
    set_names = loss.loss_history.keys()
    loss_names = list(set(sum([list(f.keys()) for f in loss.loss_history.values()],[])))
    # get number of graphs

    n_rows, n_columns = core.get_divs(len(loss_names))
    figs = {}; axes = {}
    if not separated:
        fig, ax = plt.subplots(n_rows, n_columns, figsize=(15,10))
        figs['losses'] = fig; axes['losses'] = ax
    else:
        fig = [plt.Figure() for i in range(len(loss_names))]
        ax = [f.add_subplot(1,1,1) for f in fig]
        ax = np.array(ax).reshape(n_rows, n_columns)
        for i in range(len(fig)):
            figs[loss_names[i]] = fig[i]; axes[loss_names[i]] = ax[i]

    if n_columns == 1:
        ax = np.expand_dims(ax, 1)
    elif n_rows == 1:
        ax = np.expand_dims(ax, 0)
    for i in range(n_rows):
        for j in range(n_columns):
            current_idx = i*n_columns + j
            current_loss = loss_names[current_idx]
            plots = []
            for k in set_names:
                values = loss.loss_history[k].get(current_loss)['values']
                times = loss.loss_history[k].get(current_loss).get('time')
                x = times if axis == 'time' and times is not None else range(len(values))
                if values is not None:
                    plot = ax[i,j].plot(np.array(x), values, label=k)
                    plot = plot[0]
                    plots.append(plot)
            ax[i,j].legend(handles=plots)
            ax[i,j].set_title(current_loss)

    name = kwargs.get('name', 'losses')
    out += '/losses'
    if not os.path.isdir(out+'/losses'):
        check_dir(out)
    suffix = ""
    suffix = "_"+str(partition) if partition is not None else suffix
    suffix = suffix+"_%d"%epoch if epoch is not None else suffix
    if separated:
        title = "%s_%s"%(name, loss_names[i])
        if out is not None:
            [fig[i].savefig(f"{out}/{title}{suffix}.pdf", format='pdf') for i in range(len(fig))]
    else:
        title = name
        if out is not None:
            fig.savefig(f"{out}/{title}{suffix}.pdf", format='pdf')
    return figs, axes


def plot_class_losses(dataset, model, evaluators, tasks=None, batch_size=512, partition=None, n_points=None, ids=None, epoch=None,
                      label=None, loader=None, balanced=True, preprocess=False, preprocessing=None, out=None, name=None, **kwargs):
    assert tasks
     # get plotting ids
    if partition is not None:
        dataset = dataset.retrieve(partition)
    if ids is not None:
        dataset = dataset.retrieve(ids)
    if n_points is not None:
        dataset = dataset.retrieve(np.random.permutation(len(dataset.data))[:n_points])

    classes = {}
    class_ids = {}
    full_ids = CollapsedIds()
    for t in tasks:
        class_ids[t], classes[t] = core.get_class_ids(dataset, t, balanced=balanced, split=True)
        full_ids.add(t, np.concatenate(list(class_ids[t].values())))

    Loader = loader if loader else DataLoader
    loader = Loader(dataset, batch_size, ids=full_ids.get_full_ids(), tasks = label, shuffle=False)
    # forward
    add_args = {}
    if hasattr(model, 'prediction_params'):
        add_args['n_preds'] = model.prediction_params['n_predictions']
    add_args['epoch'] = kwargs.get('epoch')

    outs = []; targets = []
    for x,y in loader:
        if preprocess and preprocessing:
            x = preprocessing(x)
        targets.append(x)
        vae_out = model.forward(x, y=y, **add_args)
        outs.append(decudify(vae_out))
    outs = merge_dicts(outs)
    targets = torch.from_numpy(np.concatenate(targets, axis=0)).float()
    #plt.figure()
    #plt.plot(targets[10]); plt.plot(outs['x_params'].mean[10])


    # forward!
    figs = {}; axes = {}
    for t in tasks:
        eval_dict = {}
        # obtain evaluations
        for class_name, i in class_ids[t].items():
            t_ids = full_ids.transform(i)
            out_class = recgetitem(outs, t_ids)
            eval_results = [e.evaluate(out_class, target=targets[t_ids], model=model) for e in evaluators]
            eval_dict[class_name] = eval_results
        # plot
        n_evals = len(eval_dict[class_name])
        n_rows, n_columns = core.get_divs(n_evals)
        fig, ax = plt.subplots(n_rows, n_columns)
        if n_rows == 1:
            ax = ax[np.newaxis]
        if n_columns == 1:
            ax = ax[:, np.newaxis]

        sorted_class_ids = sorted(eval_dict.keys())
        zoom = 0.9

        for i in range(n_rows):
            for j in range(n_columns):
                print(t, i,j)
                # get hash for losses
                loss_axis = {k: ax[i,j].twinx() for k in eval_results[i*n_columns+j].keys()}
                # get current losses
                values = {l: [] for l in eval_dict[list(eval_dict.keys())[0]][i*n_columns+j].keys()}
                for q, id in enumerate(sorted_class_ids):
                    current_result = eval_dict[id][i*n_columns+j]
                    # print(i, j, current_result)
                    ordered_keys = sorted(list([str(k) for k in current_result.keys()]))
                    color_map = core.get_cmap(len(ordered_keys))
                    for s, k in enumerate(ordered_keys):
                        origin = s
                        height = current_result[k]
                        # print(np.isnan(height), np.isinf(height), height)
                        if np.isnan(height) or np.isinf(height):
                            height = 0
                        # print(loss_axis.keys())
                        try:
                            loss_axis[k].bar(origin + zoom*(2*q+1)/(2*len(sorted_class_ids)), height, width = zoom/len(sorted_class_ids), color=color_map(q))
                        except KeyError:
                            pdb.set_trace()
                        loss_axis[k].set_xlim([-0.5, len(ordered_keys)])
                        loss_axis[k].set_yticks([])
                        if type(current_result[k]) in [list, tuple]:
                            for w in range(len(current_result[k])):
                                if not str(k)+"_%d"%w in values.keys():
                                    values[str(k)+'_%d'%w] = []
                                    del values[str(k)]
                                values[str(k)+'_%d'%w].append(float(current_result[k][w]))

                            loss_index = ordered_keys.index(str(k))
                            del ordered_keys[loss_index]
                            [ordered_keys.insert(loss_index+w, str(k)+"_%d"%w) for w in range(len(current_result[k]))]
                        else:
                            if not str(k) in values.keys():
                                values[str(k)] = []
                            values[k].append(current_result[k])
                ax[i,j].set_xticks([s for s in range(len(ordered_keys))])
                ax[i,j].set_xticklabels(ordered_keys, rotation=-0, ha='left', fontsize=6, minor=False)
                ax[i,j].set_yticks([])
                for loss_name, losses in values.items():
                    min_idx = np.argmin(np.array(losses)); max_idx = np.argmax(np.array(losses))
                    min_x = ordered_keys.index(loss_name) + zoom * (2 * min_idx + 1) / (2 * len(sorted_class_ids))
                    max_x = ordered_keys.index(loss_name) + zoom * (2 * max_idx + 1) / (2 * len(sorted_class_ids))
                    min_y = losses[min_idx]
                    max_y = losses[max_idx]
                    if np.isnan(min_y):
                        min_y = 0
                    if np.isnan(max_y):
                        max_y = 0
                    # print(min_idx, min_x, min_y, losses)
                    if np.log10(min_y) > 2:
                        min_y_str = "%.1e"%min_y
                    else:
                        min_y_str = "%.2f"%min_y
                    if np.log10(max_y) > 2:
                        max_y_str = "%.1e"%max_y
                    else:
                        max_y_str = "%.2f"%max_y
                    print(min_y_str, max_y_str)
                    loss_axis[loss_name].text(float(min_x), float(min_y), min_y_str,  fontsize=4, horizontalalignment='center', color=color_map(min_idx))
                    loss_axis[loss_name].text(float(max_x), float(max_y), max_y_str,  fontsize=4, horizontalalignment='center', color=color_map(max_idx))

            title = '/class_losses_%s_%s'%(t, partition)
            suffix = ""
            suffix = "_"+str(partition) if partition is not None else suffix
            suffix = suffix+"_%d"%epoch if epoch is not None else suffix
            if not out is None:
                out_dir = out + '/class_losses'
                check_dir(out_dir)
                fig.savefig(f"{out_dir}/{title}{epoch}.pdf", format="pdf")

    return figs, axes


@partition_dependent
def plot_confusion(dataset, model, classifiers, partition=None, n_points=None, batch_size=None, out=None, epoch=None, *args, **kwargs):
    loader = get_plotting_loader(dataset, is_sequence=model.take_sequences, partition=partition, n_points=n_points, batch_size=batch_size)
    classifiers = checklist(classifiers)
    outputs = [list()]*len(classifiers)
    full_y = []
    figs = {}; axes = []
    apply_method(classifiers, 'eval')
    with torch.no_grad():
        for x, y in loader:
            vae_out = model.encode(model.format_input_data(x), y=y)
            for i, c in enumerate(classifiers):
                outputs[i].append(c.forward(out=vae_out, from_encoder=True))
            full_y.append(y)
    classifier_out = [merge_dicts(o) for o in outputs]
    full_y = merge_dicts(full_y)

    if out is not None:
        if not os.path.isdir(out+'/classification'):
            os.makedirs(out+'/classification')
    # for c in classifiers:
    for c, c_out in enumerate(classifier_out):
        true_y = full_y.get(classifiers[c].task).int().cpu().reshape(-1).numpy()
        label_dim = classifier_out[c].probs.shape[-1]
        false_y = classifier_out[c].probs.reshape(-1, label_dim).argmax(1).cpu().numpy()
        fig, ax = core.make_confusion_matrix(confusion_matrix(true_y, false_y))
        name = f"classification_{classifiers[c].task}_{classifiers[c].layer}"
        if epoch is not None:
            name += "_%d"%epoch
        if partition is not None:
            name += "_%s"%partition
        fig.savefig(f'{out}/classification/{name}.pdf')
        figs[name] = fig; axes.append(ax)

    return figs, axes














################################################
########        MODEL ANALYSIS PLOTS
####


def plot_conv_weights(dataset, model, out=None, *args, **kwargs):
    weights = {}
    for i, current_encoder in enumerate(model.encoders):
        hidden_module = current_encoder.hidden_modules
        if issubclass(type(hidden_module), (ConvolutionalLatent, DeconvolutionalLatent)):
            layers = hidden_module.conv_module.conv_encoders
            for l in range(len(layers)):
                if issubclass(type(layers[l]), ConvLayer):
                    weights['encoder.%d.%d.weight'%(i,l)] = layers[l].conv_module.weight.data
                elif issubclass(type(layers[l]), GatedConvLayer):
                    weights['encoder.%d.%d.tanh_weight'%(i,l)] = layers[l].conv_module.weight.data
                    weights['encoder.%d.%d.sig_weight'%(i,l)] = layers[l].conv_module_sig.weight.data
                    weights['encoder.%d.%d.residual_weight'%(i,l)] = layers[l].conv_module_residual.weight.data
                    weights['encoder.%d.%d.1x1_weight'%(i,l)] = layers[l].conv_module_1x1.weight.data

    for i, current_decoder in enumerate(model.decoders):
        hidden_module = current_decoder.hidden_modules
        if issubclass(type(hidden_module), (MultiHeadConvolutionalLatent, MultiHeadDeconvolutionalLatent)):
            layers = hidden_module.conv_module.conv_encoders
            for h, head in enumerate(layers):
                layers = head.conv_encoders
                for l in range(len(layers)):
                    if issubclass(type(layers[l]), DeconvLayer):
                        weights['decoder.%d.%d.%d.weight'%(h,i,l)] = layers[l].conv_module.weight.data
                    elif issubclass(type(layers[l]), GatedDeconvLayer):
                        weights['decoder.%d.%d.%d.tanh_weight'%(h,i,l)] = layers[l].conv_module.weight.data
                        weights['decoder.%d.%d.%d.sig_weight'%(h,i,l)] = layers[l].conv_module_sig.weight.data
                        weights['decoder.%d.%d.%d.residual_weight'%(h,i,l)] = layers[l].conv_module_residual.weight.data
                        weights['decoder.%d.%d.%d.1x1_weight'%(h,i,l)] = layers[l].conv_module_1x1.weight.data
        elif issubclass(type(hidden_module), (ConvolutionalLatent, DeconvolutionalLatent)):
            layers = hidden_module.conv_module.conv_encoders
            for l in range(len(layers)):
                if issubclass(type(layers[l]), DeconvLayer):
                    weights['decoder.%d.%d.weight'%(i,l)] = layers[l].conv_module.weight.data
                elif issubclass(type(layers[l]), GatedDeconvLayer):
                    weights['decoder.%d.%d.tanh_weight'%(i,l)] = layers[l].conv_module.weight.data
                    weights['decoder.%d.%d.sig_weight'%(i,l)] = layers[l].conv_module_sig.weight.data
                    weights['decoder.%d.%d.residual_weight'%(i,l)] = layers[l].conv_module_residual.weight.data
                    weights['decoder.%d.%d.1x1_weight'%(i,l)] = layers[l].conv_module_1x1.weight.data
        out_modules = current_decoder.out_modules
        # for o in range(len(out_modules)):
        #     if issubclass(type(current_decoder.out_modules[o]), Deconvolutional):
        #     print('coucou')

    figs = []; axis = []
    if not os.path.isdir(out):
        os.makedirs(out)
    for name, kernel in weights.items():
        fig = plt.figure()
        kernel_grid = make_grid(kernel.transpose(0,2).transpose(1,2).unsqueeze(1))
        plt.imshow(kernel_grid.detach().transpose(0,2).transpose(0,1).numpy())
        figs.append(fig); axis.append(fig.axes)
        if out is not None:

            fig.savefig(out+'/%s.pdf'%name)

    return figs, axis


