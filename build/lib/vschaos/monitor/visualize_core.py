#######!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 17:00:26 2018

@author: chemla
"""
import  torch

from ..utils.onehot import oneHot
from ..utils.misc import checklist
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
from .visualize_dimred import PCA

from .. import distributions as dist

#%% Various utilities for plotting


def get_cmap(n, color_map='plasma'):
    return plt.cm.get_cmap(color_map, n)

def get_class_ids(dataset, task, balanced=False, ids=None, split=False):
    '''
    returns per-ids data index relatively to given task

    :param dataset: dataset object
    :param task: task name
    :param ids: only among given ids ( default: all )
    :return:
    '''
    if dataset.classes[task].get('_length'):
        n_classes = dataset.classes[task]['_length']
    else:
        n_classes = len(list(dataset.classes.get(task).values()))
    class_ids = dict(); n_class = set()
    for meta_id in range(n_classes):
        current_ids = np.array(list(dataset.get_ids_from_class([meta_id], task, ids=ids)))
        if len(current_ids) == 0:
            continue
        class_ids[meta_id] = np.array(current_ids).reshape(-1)
        current_meta = dataset.metadata.get(task)[current_ids]
        if current_meta is not None:
            n_class |= set(current_meta.tolist())

    bound = np.inf
    if balanced:
        bound = min([len(a) for a in class_ids.values()])

    n_class = list(n_class)
    for i in n_class:
        if len(class_ids[i]) >  bound:
            retained_ids = torch.sort(torch.randperm(len(class_ids[i]))[:bound]).values
            class_ids[i] = np.array(class_ids[i][retained_ids]).reshape(-1)

    if not split:
        class_ids = np.concatenate(list(class_ids.values())).tolist()
    return class_ids, n_class






def get_divs(n):
    primfac = []
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            primfac.append(d)  # supposing you want multiple factors repeated
            n //= d
        d += 1
    if n > 1:
       primfac.append(n)
    primfac = np.array(primfac)
    dims = (int(np.prod(primfac[0::2])), int(np.prod(primfac[1::2])))
    if dims[0] < dims[1]:
        dims = (dims[1], dims[0])
    return dims




def get_tensors_from_dist(distrib):
    if issubclass(type(distrib), dist.Normal):
        return {'mean':distrib.mean, 'std':distrib.stddev}
    else:
        raise NotImplementedError


# Plotting functions

def plot_mean_1d(dist, x=None, std=None, multihead=None, out=None, *args, **kwags):
    n_examples = dist.shape[0]
    # create axes
    if len(dist.shape) <= 2:
        n_rows, n_columns = get_divs(n_examples)
    elif len(dist.shape) == 3:
        # is sequence
        n_rows = n_examples
        n_columns = dist.shape[1]

    fig, axes = plt.subplots(n_rows, n_columns, figsize=(10,10))
    if n_columns == 1:
        axes = axes[:, np.newaxis]
    if n_rows == 1:
        axes = axes[np.newaxis, :]


    # get distributions
    dist_mean = dist.cpu().detach().numpy(); dist_mean_inv=None
    if std is not None:
        dist_variance = std.cpu().detach().numpy()

    if x is None:
        x = np.zeros_like(dist_mean)
    if torch.is_tensor(x):
        x = x.cpu().detach().numpy()

    seq_stride = 1
    for i in range(n_rows):
        for j in range(n_columns):
            if len(dist.shape) == 2:
                x_orig = x[i*n_columns+j]; x_dist = dist_mean[i*n_columns+j]
            else:
                x_orig = x[i, j*seq_stride]; x_dist = dist_mean[i, j*seq_stride]

            if x is not None:
                axes[i,j].plot(x_orig, linewidth=0.6)
            axes[i,j].plot(x_dist, linewidth=0.4)

    # multihead plot
    if multihead is not None:
        for k in range(len(multihead)):
            fig_m, axes_m = plt.subplots(n_rows, n_columns, figsize=(10,10))
            if len(axes_m.shape) == 1:
                axes_m = axes_m[:, np.newaxis]
            for i in range(n_rows):
                for j in range(n_columns):
                    axes_m[i, j].plot(multihead[k][i*n_columns+j].squeeze().numpy(), linewidth=0.6)

            fig.append(fig_m)
            axes.append(axes_m)

    if out is not None:
        fig.savefig(out+'.pdf', format="pdf")

    return fig, axes

def plot_mean_2d(dist, x=None, std=None, out=None, fig=None, *args, **kwargs):
    n_examples = dist.shape[0]
    n_rows, n_columns = get_divs(n_examples)
    if x is None:
        fig, axes = plt.subplots(n_rows, n_columns)
        if std is not None:
            fig_std, axes_std = plt.subplots(n_rows, n_columns)
    else:
        fig, axes = plt.subplots(n_rows, 2 * n_columns)
        if std is not None:
            fig_std, axes_std = plt.subplots(n_rows, 2*n_columns)

    if n_rows == 1:
        axes = axes[np.newaxis, :]
        if std is not None:
            axes_std = axes_std[:, np.newaxis]
    if n_columns == 1 and x is None:
        axes = axes[:, np.newaxis]
        if std is not None:
            axes_std = axes_std[:, np.newaxis]

    dist_mean = dist.cpu().detach().numpy()
    if std is not None:
        dist_std = std.cpu().detach().numpy()

    for i in range(n_rows):
        for j in range(n_columns):
                if x is not None:
                    axes[i,2*j].imshow(x[i*n_columns+j], aspect='auto')
                    axes[i,2*j+1].imshow(dist_mean[i*n_columns+j], aspect='auto')
                else:
                    axes[i,j].imshow(dist_mean[i*n_columns+j], aspect='auto')
                if hasattr(dist, "stddev"):
                    if x is not None:
                        axes_std[i,2*j].imshow(x[i*n_columns+j], aspect='auto')
                        axes_std[i,2*j+1].imshow(dist_std[i*n_columns+j],vmin=0, vmax=1, aspect='auto')
                    else:
                        axes_std[i,j].imshow(dist_std[i*n_columns+j], vmin=0, vmax=1, aspect='auto')

    if out is not None:
        fig.savefig(out+".pdf", format="pdf")

    return fig, axes


def plot_dirac(x, *args, **kwargs):
    x = dist.Normal(x, torch.zeros_like(x))
    return plot_mean(x, *args, **kwargs)


def plot_mean(x, target=None, preprocessing=None, axes=None, *args, is_sequence=False, **kwargs):
    is_sequence = is_sequence and x.mean.shape[1] > 1
    if type(x) == dist.Normal:
        x, std = x.mean, x.stddev
    else:
        x, std = x.mean, None

    if preprocessing:
        x = preprocessing.invert(x.detach().cpu())
        target = preprocessing.invert(target.detach().cpu())

    x = x.squeeze()
    if std is not None:
        std = std.detach().cpu().squeeze()
    if target is not None:
        target = target.squeeze()
    if is_sequence:
        figs = [None]*x.shape[0]; axes = [None]*x.shape[0]
        if std is None:
            std = [None]*x.shape[0]
        for ex in range(x.shape[0]):
            if target is not None:
                target_tmp = target[ex]
            if len(x.shape) <= 3:
                figs[ex], axes[ex] = plot_mean_1d(x[ex], x=target_tmp, *args, **kwargs)
            elif len(x.shape) == 4:
                kwargs['out'] = None if kwargs.get('out') is None else kwargs.get('out')+'_%d'%ex
                figs[ex], axes[ex] = plot_mean_2d(x[ex], x=target_tmp, std=std[ex], *args, **kwargs)
    else:
        figs = {}; axes = {}
        if len(x.shape) == 2:
            figs['mean'], axes['mean'] = plot_mean_1d(x, x=target, axes=axes, std=std, *args, **kwargs)
            if std is not None:
                figs['std'], axes['std'] = plot_mean_1d(std, x=target, axes=axes, std=std, *args, **kwargs)
        elif len(x.shape) == 3:
            figs['mean'], axes['mean'] = plot_mean_2d(x, x=target, axes=axes, std=std, *args, **kwargs)
            if std is not None:
                figs['std'], axes['std'] = plot_mean_2d(std, x=target, axes=axes, std=std, *args, **kwargs)
    return figs, axes

def plot_empirical(x, *args, **kwargs):
    return plot_mean(dist.Normal(x.mean, x.stddev), *args, **kwargs)

def plot_probs(x, target=None, preprocessing=None, *args, **kwargs):
    n_examples = x.batch_shape[0]
    n_rows, n_columns = get_divs(n_examples)
    fig, axes = plt.subplots(n_rows, n_columns) if len(target.shape) == 2 else plt.subplots(n_rows, 2*n_columns)
    if not issubclass(type(axes), np.ndarray):
        axes = np.array(axes)[np.newaxis]
    if axes.ndim == 1:
        axes = axes[:, np.newaxis]
    for i in range(n_rows):
        for j in range(n_columns):
            if target is not None:
                if len(target.shape) < 2:
                    target = oneHot(target, dist.batch_shape[1])
                if torch.is_tensor(target):
                    target = target.cpu().detach().numpy()
                axes[i,j].plot(target[i*n_columns+j])
            if len(target.shape) == 2:
                probs = x.probs[i*n_columns+j]
                if torch.is_tensor(probs):
                    probs = probs.cpu().detach().numpy()
                axes[i,j].plot(probs, linewidth=0.5)
                plt.tick_params(axis='y',  which='both',  bottom='off')
            elif len(x.shape) == 3:
               raise NotImplementedError
    return fig, axes


plotting_hash = {torch.Tensor: plot_dirac,
                 dist.Normal: plot_mean,
                 dist.Bernoulli: plot_mean,
                 dist.Categorical: plot_probs, 
                 dist.Empirical: plot_empirical}

def plot_distribution(dists, *args, **kwargs):
    if issubclass(type(dists), list):
        return [plot_distribution(dists[i], *args, **kwargs) for i in range(len(dists))]
    if not type(dists) in plotting_hash.keys():
        raise TypeError("error in plot_distribution : don't have a plot callback for type %s"%type(dists))
    fig, ax = plotting_hash[type(dists)](dists, *args, **kwargs)
    return fig, ax


def plot_latent_path(zs, data, synth, reduction=None):
    fig = plt.figure()
    grid = plt.GridSpec(1, 4, hspace=0.2, wspace=0.2)
        
    if not reduction is None:
        zs = reduction.transform(zs)
        
    gradient = get_cmap(zs.shape[0])
    if zs.shape[1] == 2:
        ax = fig.add_subplot(grid[:2])
        ax.plot(zs[:,0],zs[:,1], c=gradient(np.arange(zs.shape[0])))
    elif zs.shape[1] == 3:
        ax = fig.add_subplot(grid[:2], projection='3d', xmargin=1)
        for i in range(zs.shape[0]-1):
            ax.plot([zs[i,0], zs[i+1, 0]], [zs[i,1], zs[i+1, 1]], [zs[i,2], zs[i+1, 2]], c=gradient(i))

    ax = fig.add_subplot(grid[2])
    ax.imshow(data, aspect='auto')
    ax = fig.add_subplot(grid[3])
    ax.imshow(synth, aspect='auto')
    return fig, fig.axes


def plot_latent(current_z, *args, target_dim = 3, sample=False, transformation=PCA, class_ids=None, **kwargs):
    if isinstance(current_z['out_params'], dist.Normal):
        if sample:
            full_z = current_z['out'].cpu().detach().numpy()
            full_var = None
        else:
            full_z = current_z['out_params'].mean.cpu().detach().numpy()
            full_var = current_z['out_params'].variance.cpu().detach().numpy()
            full_var = np.mean(full_var, tuple(range(1,full_var.ndim)))

        # check if given latent vectors are sequences
        sequence = len(full_z.shape) > 2

        # transform in case
        transformation = transformation or PCA
        assert target_dim in [2, 3], "target_dim for latent plotting must be 2 or 3"
        if full_z.shape[-1] > target_dim and not sequence:
            if issubclass(type(transformation), type):
                full_z = transformation(n_components=target_dim).fit_transform(full_z)
            else:
                full_z = transformation.transform(full_z)

        if full_z.shape[-1] == 2:
            fig, ax = plot_2d(full_z, var=full_var, class_ids=class_ids, *args, **kwargs)
        elif full_z.shape[-1] >= 3:
            fig, ax = plot_3d(full_z, var=full_var, class_ids=class_ids, *args, **kwargs)

    elif isinstance(current_z['out_params'], dist.Multinomial):
        fig, ax = plot_multinomial(current_z, class_ids=class_ids, **kwargs)

    return fig, ax


def plot_multinomial(current_z, meta=None, classes=None, class_ids=None, class_names=None, cmap="plasma", legend=True, **kwargs):
    fig = plt.figure()
    ax = fig.gca()
    n_classes = len(list(class_ids.keys()))
    width = 1 / (n_classes+1)
    i = 0

    if meta is None:
        meta = np.zeros((current_z.shape[0]))
        cmap = get_cmap(0, color_map=cmap)
        cmap_hash = {0:0}
    else:
        cmap = get_cmap(len(classes), color_map=cmap)
        cmap_hash = {classes[i]:i for i in range(len(classes))}

    for k, v in class_ids.items():
        current_probs = current_z['out_params'][v]
        dim_latent = current_probs.event_shape[0]
        class_prob = torch.eye(dim_latent)
        log_probs = np.stack([current_probs.log_prob(c).exp().numpy() for c in class_prob], 1)
        x = np.arange(dim_latent) + i / (n_classes+1)
        plt.bar(x, log_probs.mean(0), width * np.ones_like(x), align="edge", color=cmap(i / n_classes))
        i += 1

    if legend and not meta is None and not classes is None:
        handles = []
        for cl in classes:
            patch = mpatches.Patch(color=cmap(cmap_hash[cl]), label=str(class_names[cl]))
            handles.append(patch)
        chartBox = ax.get_position()
        ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.8, chartBox.height])
        ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(1.2, 0.8), shadow=True, ncol=1)
    return fig, ax


def plot_2d(current_z, meta=None, var=None, classes=None, class_ids=None, class_names=None, cmap='plasma',
            sequence=None, shadow_z=None, centroids=None, legend=True):
    fig = plt.figure(figsize=(8,6))
    ax = fig.gca()

    if meta is None:
        meta = np.zeros((current_z.shape[0]))
        cmap = get_cmap(0, color_map=cmap)
        cmap_hash = {0:0}
    else:
        cmap = get_cmap(len(classes), color_map=cmap)
        cmap_hash = {classes[i]:i for i in range(len(classes))}

    current_alpha = 0.06 if (centroids and not meta is None) else 1.0
    current_var = var if not var is None else np.ones(current_z.shape[0])
    current_var = (current_var - current_var.mean() / np.abs(current_var).max())+1
    meta = meta.astype(np.int)

    # plot
    if sequence:
        if shadow_z is not None:
            for i in range(shadow_z.shape[0]):
                ax.plot(shadow_z[i, :, 0], shadow_z[i, :,1], c=np.array([0.8, 0.8, 0.8, 0.4]))
        for i in range(current_z.shape[0]):
            ax.plot(current_z[i, :, 0], current_z[i, :,1], c=cmap(cmap_hash[meta[i]]), alpha = current_alpha)
            ax.scatter(current_z[i,0,0], current_z[i,0,1], c=cmap(cmap_hash[meta[0]]), alpha = current_alpha, marker='o')
            ax.scatter(current_z[i,-1,0], current_z[i,-1,1], c=cmap(cmap_hash[meta[0]]), alpha = current_alpha, marker='+')
    else:
        cs = np.array([cmap_hash[m] for m in meta])
        ax.scatter(current_z[:, 0], current_z[:,1], c=cs, alpha = current_alpha, s=current_var)
    # make centroids
    if centroids and not meta is None:
        for i, cid in class_ids.items():
            centroid = np.mean(current_z[cid], axis=0)
            ax.scatter(centroid[0], centroid[1], centroid[2], s = 30, c=cmap(classes[i]))
            ax.text(centroid[0], centroid[1], centroid[2], class_names[i], color=cmap(classes[i]), fontsize=10)
    # make legends
    if legend and not meta is None and not classes is None:
        handles = []
        for cl in classes:
            patch = mpatches.Patch(color=cmap(cmap_hash[cl]), label=str(class_names[cl]))
            handles.append(patch)
        ax.legend(handles=handles, loc='upper left', borderaxespad=0.)

    return fig, ax


def plot_dims(current_z, meta=None, var=None, classes=None, class_ids=None, class_names=None, cmap='plasma', sequence=False, centroids=None, legend=True, scramble=True):
    fig = plt.figure(figsize=(8,6))
    ax = fig.gca(projection='3d')

    if meta is None:
        meta = np.zeros((current_z.shape[0])).astype(np.int)
        cmap = get_cmap(1, color_map=cmap)
        cmap_hash = {0:0}
    else:
        cmap = get_cmap(len(checklist(classes)), color_map=cmap)
        cmap_hash = {classes[i]:i for i in range(len(checklist(classes)))}

    # flatten if sequence
    if len(current_z.shape) == 3:
        current_z = current_z.reshape((-1, current_z.shape[-1]))
        meta = np.array(meta, dtype=np.int).reshape((-1))
        if var is not None:
            var = var.reshape((-1, current_z.shape[-1]))

    n_rows, n_columns = get_divs(current_z.shape[-1])
    fig, ax = plt.subplots(n_rows, n_columns)
    if n_rows == 1:
        ax = ax[np.newaxis]
    if n_columns == 1:
        ax = ax[:, np.newaxis]

    cs = np.array([cmap_hash[i.item()] for i in meta])
    current_var = var if var is not None else np.zeros_like(current_z)

    if scramble:
        index_ids = np.random.permutation(current_z.shape[0])
    else:
        index_ids = np.arange(current_z.shape[0])

    for i in range(n_rows):
        for j in range(n_columns):
            current_dim =  i*n_columns+j
            ax[i,j].scatter(current_z[index_ids, current_dim], current_var[index_ids, current_dim],
                          c=cmap(cs[index_ids]), s=0.8, marker=".")
            #ax[i, j].set_ylim([0, 1])
            ax[i,j].set_xticklabels(ax[i,j].get_xticks(), {'size':3})
            ax[i,j].xaxis.set_major_formatter(FormatStrFormatter('%2.g'))
            ax[i,j].set_yticklabels(ax[i,j].get_yticks(), {'size':3})
            ax[i,j].yaxis.set_major_formatter(FormatStrFormatter('%2.g'))
            ax[i,j].set_ylim([0, min(1.0, 1.2*current_var.max())])


    # make legends
    if legend and not meta is None and not classes is None:
        handles = []
        for cl in classes:
            patch = mpatches.Patch(color=cmap(cmap_hash[cl]), label=str(class_names[cl]))
            handles.append(patch)
        fig.legend(handles=handles, loc='upper left', borderaxespad=0.)

    return fig, ax


def plot_pairwise_trajs(z_pos, var=None, **kwargs):

    # # flatten if sequence
    # if len(current_z.shape) == 3:
    #     meta_rs = np.ones(current_z.shape[:-1]+(1,))
    #     for i in range(current_z.shape[0]):
    #         meta_rs[i] *= meta[i]
    #     current_z = current_z.reshape(np.cumprod(current_z.shape[:-1])[-1], current_z.shape[-1])
    #     meta_rs = meta_rs.reshape(np.cumprod(meta_rs.shape[:-1])[-1], meta_rs.shape[-1])
    #     if var is not None:
    #         var = var.reshape(np.cumprod(var.shape[:-1])[-1], var.shape[-1])
    #     meta = meta_rs[:, 0]
    n_examples = z_pos[0].shape[0]
    figs = []; axs = []
    for ex in range(n_examples):
        current_z = [z_pos[0][ex], z_pos[1][ex]]
        n_rows, n_columns = get_divs(current_z[0].shape[-1])
        fig, ax = plt.subplots(n_rows, n_columns, figsize=(8,6))
        if n_rows == 1:
            ax = ax[np.newaxis]
        if n_columns == 1:
            ax = ax[:,np.newaxis]
        current_var = None if var is None else [var[0][ex], var[1][ex]]
        for i in range(n_rows):
            for j in range(n_columns):
                current_dim =  i*n_columns+j
                ax[i,j].plot(current_z[0][:, current_dim], linewidth=0.8, c='b')
                ax[i,j].plot(current_z[1][:, current_dim], linewidth=0.8, c='r')
                if current_var is not None:
                    ax[i,j].plot(current_z[0][:, current_dim] + current_var[0][:, current_dim], c="b", linewidth=0.5, linestyle="-")
                    ax[i,j].plot(current_z[0][:, current_dim] - current_var[0][:, current_dim], c="b", linewidth=0.5, linestyle="-")
                    ax[i,j].plot(current_z[1][:, current_dim] + current_var[1][:, current_dim], c="r", linewidth=0.5, linestyle="-")
                    ax[i,j].plot(current_z[1][:, current_dim] - current_var[1][:, current_dim], c="r", linewidth=0.5, linestyle="-")
        figs.append(fig); axs.append(ax)

    return figs, axs



def plot_3d(current_z, meta=None, var=None, classes=None, class_ids=None, class_names=None, cmap='plasma', sequence=False, centroids=None, legend=True, shadow_z=None, scramble=True):
    fig = plt.figure(figsize=(8,6))
    ax = fig.gca(projection='3d')
    
    if meta is None:
        meta = np.arange((current_z.shape[0]))
        cmap = get_cmap(meta.shape[0])
        cmap_hash = {x:x for x in meta}
        legend = False
    else:
        cmap = get_cmap(0) if classes is None else get_cmap(len(classes))
        cmap_hash = {None:None} if classes is None else {classes[i]:i for i in range(len(classes))}

    current_alpha = 0.06 if (centroids and not meta is None) else 1.0
    current_var = var if not var is None else np.ones(current_z.shape[0])
    current_var = (current_var - current_var.mean() / np.abs(current_var).max())+1

    if meta is not None:
        meta = np.array(meta).astype(np.int)

    n_examples = current_z.shape[0]
    if scramble:
        index_ids = np.random.permutation(n_examples)
    else:
        index_ids = np.arange(n_examples)

    # plot
    if sequence:
        if shadow_z is not None:
            for i in range(shadow_z.shape[0]):
                ax.plot(shadow_z[i, :, 0], shadow_z[i, index_ids ,1], shadow_z[i, index_ids ,2], c=np.array([0.8, 0.8, 0.8, 0.4]))
        for i in index_ids:
            if meta.ndim == 1:
                meta = meta[:, np.newaxis]
            if isinstance(meta, np.ndarray):
                c = cmap(cmap_hash[set(meta[i]).pop()])
            else:
                c = cmap(cmap_hash[meta[i]])
            ax.plot(current_z[i, :, 0], current_z[i, :,1],current_z[i, :,2], c=c, alpha = current_alpha)
            c = np.array(c)[np.newaxis]
            ax.scatter(current_z[i,0,0], current_z[i,0,1],current_z[i,0,2], c=c, alpha = current_alpha, marker='o')
            ax.scatter(current_z[i,-1,0], current_z[i,-1,1],current_z[i,-1,2], c=c, alpha = current_alpha, marker='+')
    else:
        cs = cmap(np.stack([cmap_hash[m] for m in meta]))
        if current_z.shape[1]==2:
            ax.scatter(current_z[index_ids, 0], current_z[:,1], np.zeros_like(current_z[index_ids,0]), c=cs[index_ids], alpha = current_alpha, s=current_var)
        else:
            ax.scatter(current_z[index_ids, 0], current_z[index_ids,1], current_z[index_ids, 2], c=cs[index_ids], alpha = current_alpha, s=current_var)
    # make centroids
    if centroids and not meta is None:
        for i, cid in class_ids.items():
            centroid = np.mean(current_z[cid], axis=0)
            color=np.array(cmap(cmap_hash[i]))[np.newaxis]
            ax.scatter(centroid[0], centroid[1], centroid[2], s = 30, c= color)
            ax.text(centroid[0], centroid[1], centroid[2], class_names[i], color=cmap(cmap_hash[i]), fontsize=10)
    # make legends   
    if legend and not meta is None and not classes is None:
        handles = []
        for cl in classes:
            patch = mpatches.Patch(color=cmap(cmap_hash[cl]), label=str(class_names[cl]))
            handles.append(patch)
        ax.legend(handles=handles, loc='upper left', borderaxespad=0.)
        
    return fig, ax




##############################
######   confusion matirices


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    source : https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py
    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    plt.figure(figsize=figsize)
    sns.set(font_scale=0.5)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)

    return plt.gcf(), plt.gca()

