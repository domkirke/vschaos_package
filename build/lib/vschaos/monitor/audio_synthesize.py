#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 15:11:17 2018

@author: chemla
"""
import os
import numpy as np, torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import ndimage
from librosa.core import load
from ..data import Dataset, ComposeTransform, ComposeAudioTransform, STFT
from ..utils.trajectory import  get_interpolation
from ..utils import checklist, choices, apply_method
from vschaos.data import dyn_collate
from ..utils import trajectories as traj
import torchaudio as ta

# CORE FUNCTIONS

def path2audio(model, current_z, transforms, meta=None,n_interp=1, order_interp=1, from_layer=-1, out=None, graphs=True, sample=False, projection=None, sr=44100, **kwargs):
    # make interpolation
    if order_interp == 0:
        z_interp = np.zeros((current_z.shape[0]*n_interp, current_z.shape[1]))
        for i in range(current_z.shape[0]):
            z_interp[i*n_interp:(i+1)*n_interp] = current_z[i]
    else:
        coord_interp = np.linspace(0, current_z.shape[0], (current_z.shape[0])*n_interp)
        z_interp = np.zeros((len(coord_interp), current_z.shape[1]))
        for i,y in enumerate(coord_interp):
            z_interp[i] = ndimage.map_coordinates(current_z, [y * np.ones(current_z.shape[1]), np.arange(current_z.shape[1])], order=2)
        z_interp = torch.from_numpy(z_interp)

    # get corresponding sound distribution
    model.eval()
    with torch.no_grad():
        if model.take_sequences:
            z_interp = z_interp.unsqueeze(0)
        vae_out = model.decode( model.format_input_data(z_interp), from_layer=from_layer, sample=False, y=meta)
    if sample:
        signal_out = vae_out[0]['out_params'].sample().squeeze()
    else:
        signal_out = vae_out[0]['out_params'].mean.squeeze()
    if len(signal_out.shape) > 2:
        signal_out = signal_out.reshape(signal_out.shape[0]*signal_out.shape[1], signal_out.shape[2])

    if model.take_sequences:
        z_interp = z_interp.squeeze(0)

    if graphs:
        if projection:
            fig = plt.figure()
            ax_latent = fig.add_subplot(121, projection="3d")
            z_projected = projection.transform(z_interp)
            if len(z_projected.shape) > 2:
                z_projected = z_projected[0]
            ax_latent.plot(z_projected[:,0], z_projected[:, 1], z_projected[:, 2])
            fig.add_subplot(122).imshow(signal_out.cpu().detach().numpy(), aspect='auto')
            fig.savefig(out+'.pdf', format="pdf")
        else:
            fig = plt.figure()
            plt.imshow(signal_out.cpu().detach().numpy(), aspect='auto')
            fig.savefig(out+'.pdf', format="pdf")

    # output signal
    signal_out[signal_out <= 5e-4] = 0.
    signal_out = transforms.invert(signal_out)
    signal_out = signal_out / signal_out.abs().max()
    ta.save(out+'.wav', signal_out, sr)
    return z_interp



def window_signal(sig, window, overlap=None, axis=0, window_type=None, pad=True):
    overlap = overlap or window
    if sig.shape[0] < window:
        return sig[np.newaxis]
    if pad:
        n_windows = sig.shape[axis] // overlap
        target_size = n_windows * overlap + window
        if sig.shape[axis] < target_size:
            pads = [(0,0)]*len(sig.shape)
            pads[axis]=(0,target_size-sig.shape[axis])
            sig = np.pad(sig, pads, mode='constant')
    else:
        n_windows = (sig.shape[axis]-window)//overlap
    sig = np.stack([sig[i*overlap:i*overlap+window] for i in range(n_windows)], axis=axis)
    if window_type:
        sig = sig * window_type(sig.shape[-1])
    return torch.from_numpy(sig)


def overlap_add(sig, axis=0, overlap=None, window_type=None, fusion="stack_right"):
    overlap = overlap or sig.shape[axis]
    new_size = sig.shape[axis]*overlap + sig.shape[axis+1]
    new_shape = (*sig.shape[:axis], new_size)
    if len(sig.shape) > axis:
        new_shape += sig.shape[axis+2:]
    sig_t = torch.zeros(new_shape, dtype=sig.dtype)
    if window_type:
        sig *= window_type(sig.shape[-1])
    for i in range(sig.shape[axis]):
        idx = [slice(None)]*len(sig_t.shape); idx[axis] = slice(i*overlap, i*overlap+sig.shape[axis+1])
        idx_2 = [slice(None)]*len(sig.shape); idx_2[axis] = i
        if fusion == "stack_right":
            sig_t.__setitem__(tuple(idx), sig.__getitem__(tuple(idx_2)))
        elif fusion == "overlap_add":
            sig_t.__getitem__(tuple(idx)).__iadd__(sig.__getitem__(tuple(idx_2)))
    return sig_t
        # sig_t.__getitem__((*take_all, slice(i*overlap,i*overlap+window))).__iadd__(sig.__getitem__()))


def get_transform_from_files(files, transform, transformOptions, window=None, take_sequences=False, merge_mode="min"):
    transform = transform or transformOptions.get('transformType')
    transform_datas = []
    for i, current_file in enumerate(files):
        # get transform from file
        if transform is not None:
            if issubclass(type(transform), (list, tuple)):
                transform = transform[0]
            current_transform = computeTransform([current_file], transform, transformOptions)[0]
            current_transform = np.array(current_transform)
        else:
            current_transform, _= load(current_file, sr=transformOptions.get('resampleTo', 22050))
        originalPhase = np.angle(current_transform)

        # window dat
        if window:
            overlap = overlap or window
            current_transform = window_signal(current_transform, window, overlap, axis=0)
        if take_sequences:
            current_transform = current_transform[np.newaxis]
        transform_datas.append(current_transform)
    if merge_mode == "min":
        if take_sequences:
            min_size = min([m.shape[1] for m in transform_datas])
            transform_datas = [td[:, :min_size] for td in transform_datas]
        else:
            min_size = min([m.shape[0] for m in transform_datas])
            transform_datas = [td[:min_size] for td in transform_datas]

    return transform_datas, originalPhase


# HIGH-LEVEL SYNTHESIS METHODS

def resynthesize_files(dataset, model, transforms=None, metadata=None, out='./', take_sequences=True,
                       sample=False, norm=False, n_files=10, retain_phase=False, files=None, epoch=None, **kwargs):

    if files is None:
        if issubclass(type(dataset), Dataset):
            files = choices(dataset.files, k=n_files)
        else:
            files = choices(dataset, k=n_files)

    originals = {}; resynths = {}
    if len(files)==0:
        raise ValueError("Files are empty")

    stft_transform = None
    if retain_phase:
        if isinstance(transforms, (ComposeTransform, ComposeAudioTransform)):
            stft_index = None
            for i in range(len(transforms)):
                if isinstance(transforms.transforms[i], STFT):
                    stft_index = i
            
            if stft_index is None:
                raise ValueError('STFT must be present in transforms %s when retain_phase=True'%transforms)
            stft_transform = transforms.transforms[stft_index].retain_phase(True)
        else:
            if not isinstance(transforms, STFT):
                raise ValueError('STFT must be present in transforms %s when retain_phase=True'%transforms)

    for i, current_file in enumerate(files):
        # get transform from file
        raw_input, fs = ta.load(current_file)
        if dataset is not None:
            if dataset.sample_rate:
                raw_input = ta.transforms.Resample(fs, dataset.sample_rate)(raw_input)
        current_transform = transforms(raw_input)
        original_out = transforms.invert(current_transform)
        is_original_clipping = original_out.abs().max() > 1.0
        if norm or is_original_clipping:
            original_out = original_out / original_out.abs().max()
            if is_original_clipping:
                print('{Warning} resynthesis of original file is clipping, normalizing')
        if take_sequences:
            current_transform = apply_method(current_transform, "unsqueeze", 0)

        path_out = out+'/audio_reconstruction/'+os.path.splitext(os.path.basename(current_file))[0]+('' if epoch is None else '_%d'%epoch)+'.wav'
        original_path = out+'/audio_reconstruction/'+os.path.splitext(os.path.basename(current_file))[0]+('' if epoch is None else '_%d'%epoch)+'_original.wav'
        path_resynth_out = out + '/audio_reconstruction/' + os.path.splitext(os.path.basename(current_file))[0] + (
            '' if epoch is None else '_%d' % epoch) + '_original_resynth.wav'
        if not os.path.isdir(os.path.dirname(path_out)):
            os.makedirs(os.path.dirname(path_out))
        os.system('cp "%s" "%s"'%(current_file, original_path))


        # forward
        y = {}
        if metadata is not None:
            if isinstance(current_transform, list):
                current_transform_tmp = current_transform[0]
            else:
                current_transform_tmp = current_transform
            for k, v in metadata.items():
                if current_transform_tmp.ndim == 2:
                    y[k] = v[i].unsqueeze(0).repeat(current_transform_tmp.shape[0])
                elif current_transform_tmp.ndim > 2:
                    y[k] = v[i][np.newaxis, np.newaxis].repeat(*current_transform_tmp.shape[:2])
        with torch.no_grad():
            # if isinstance(current_transform, list):
            #     current_transform =  [c_i.unsqueeze(0) for c_i in current_transform]
            vae_out = model.forward(current_transform, y=y, **kwargs)
        if issubclass(type(vae_out['x_params']), list):
            transform_out = [None, None]
            transform_out[0] = vae_out['x_params'][0].sample().cpu() if sample else vae_out['x_params'][0].mean.cpu()
            transform_out[1] = vae_out['x_params'][1].sample().cpu() if sample else vae_out['x_params'][1].mean.cpu()
            current_transform = [current_transform[0], current_transform[1]]
        else:
            transform_out = vae_out['x_params'].sample().cpu() if sample else vae_out['x_params'].mean.cpu()
            current_transform = current_transform

        if take_sequences:
            current_transform = apply_method(current_transform, "squeeze", 0)
            transform_out = apply_method(transform_out, "squeeze", 0)

        signal_out = transforms.invert(transform_out)
        is_signal_clipping = signal_out.abs().max() > 1.0

        if dataset is not None:
            final_fs =dataset.sample_rate or fs
        else:
            final_fs = fs
        if norm or is_signal_clipping:
            if is_signal_clipping:
                print('{Warning} synthesis of model output is clipping, normalizing')
            signal_out = signal_out / signal_out.abs().max()
        ta.save(path_out, signal_out, sample_rate=final_fs)
        ta.save(path_resynth_out, original_out, sample_rate=final_fs)

        originals[original_path]= original_out; resynths[original_path] = signal_out

    if retain_phase:
        stft_transform = transforms.transforms[stft_index].retain_phase(False)
    return {'original':originals, 'resynth':resynths}, final_fs



def trajectory2audio(model, traj_types, transforms, n_trajectories=1, n_steps=64, target_duration=None,
                     out=None, layers=None, projection=None, meta=None, sr=44100, sample=False, **kwargs):
    # load model
    if not os.path.isdir(out+'/trajectories'):
        os.makedirs(out+'/trajectories')
    paths = []
    layers = layers or range(len(model.platent))
    if projection is not None:
        assert len(projection) == len(layers), "%d projections given for %d latent layers"%(len(projection), len(layers))
    for layer in layers:
        for i, traj_type in enumerate(traj_types):
            trajectories = []
            # generate trajectories
            latent_dim = model.platent[layer]['dim']
            current_meta = None
            for n_traj in range(n_trajectories):
                if meta is not None:
                    current_meta = {k:v[n_traj].repeat(n_steps) for k, v in meta.items()}
                if target_duration is not None:
                    t_options = {'n_steps': target_duration, 'fs':sr}
                else:
                    t_options = {'n_steps':n_steps}
                if traj_type=='line':
                    trajectory = traj.Line(**t_options,
                                           dim=latent_dim,
                                           origin=traj.Uniform(dim=latent_dim, range=[-6,6]),
                                           end=traj.Uniform(dim=latent_dim, range=[-6,6]))
                elif traj_type == 'ellipse':
                    geom_args = {'radius':[traj.Uniform(range=[0.1, 10])()]*2, 'origin':traj.Uniform(range=[-2, 2]), 'plane':2}
                    trajectory = traj.Ellipse(**t_options, dim=model.platent[layer]['dim'], **geom_args)
                elif traj_type in ["square", "triangle"]:
                    # geom_args = {'freq':np.random.uniform(1e-3, 1e-2, (latent_dim,)),
                    geom_args = {'freq': 0.01,
                                 'phase':np.random.uniform(-3, 3, (latent_dim,)),
                                 'amp':np.random.uniform(0.01, 10, (latent_dim, 2)),
                                 'pulse':np.random.uniform(0.2, 0.8, (latent_dim,))}
                    if traj_type == "square":
                        trajectory = traj.Square(**t_options, dim=model.platent[layer]['dim'], **geom_args)
                    elif traj_type == "triangle":
                        trajectory = traj.Triangle(**t_options, dim=model.platent[layer]['dim'], **geom_args)
                elif traj_type in ["sin","sawtooth"]:
                    # geom_args = {'freq':np.random.uniform(1e-3, 1e-2, (latent_dim,)),
                    geom_args = {'freq': 0.1,
                                 'phase': np.random.uniform(-3, 3, (latent_dim,)),
                                 'amp': np.random.uniform(0.01, 3, (latent_dim, 2))}
                    if traj_type == "sin":
                        trajectory = traj.Sine(**t_options, dim=model.platent[layer]['dim'], **geom_args)
                    elif traj_type == "sawtooth":
                        trajectory = traj.Sawtooth(**t_options, dim=model.platent[layer]['dim'], **geom_args)
                trajectories.append(trajectory())

                # forward trajectory
                current_proj = projection
                if issubclass(type(projection), list):
                    current_proj = projection[layer]
            for i,t in enumerate(trajectories):
                z = path2audio(model, t, transforms, n_interp=1, meta=current_meta,
                               out=out+'/trajectories/%s_%s_%s'%(traj_type, layer,  i),
                               from_layer=layer, sr=sr, projection=current_proj, sample=sample, **kwargs)
            paths.append(z)
    return paths




def interpolate_files(dataset, vae, n_files=None, files=None, n_interp=10, out=None, transforms=None, preprocess=False,
                      projections=None, meta=None, **kwargs):

    n_files = n_files or len(files)
    for f in range(n_files):
        #sequence_length = loaded_data['script_args'].sequence
        #files_to_morph = random.choices(range(len(dataset.data)), k=2)
        if files is None:
            ids = torch.randperm(len(dataset.files))[:2]
            files_to_morph = [dataset.files[i] for i in ids]
            if meta is None:
                meta = {k: v[ids] for k, v in dataset.metadata.items()}
        else:
            if meta is None:
                meta = {}
                for k, v in dataset.metadata.items():
                    n_classes = dataset.classes[k].get('_length', v.max())
                    meta[k] = torch.randint(n_classes, (len(files_to_resynthesize),))
            files_to_morph = files[f]

        data_outs = []
        with torch.no_grad():

            loaded = [ta.load(f) for f in files_to_morph]
            vae_input = dyn_collate([transforms(l[0]) for l in loaded])
            sr = set([l[1] for l in loaded]).pop()
            if checklist(vae_input)[0].ndim > 2:
                meta  = {k:v.unsqueeze(1).repeat(1, checklist(vae_input)[0].shape[1]) for k, v in meta.items()}
            vae_input = vae.format_input_data(vae_input)
            z_encoded = vae.encode(vae_input, y=meta)

            trajs = []
            for l in range(len(vae.platent)):
                # get trajectory between target points

                #pdb.set_trace()
                traj, meta_traj = get_interpolation(z_encoded[l]['out_params'].mean.cpu().detach(), n_steps=n_interp+2, meta=meta)
                device = next(vae.parameters()).device
                traj = traj.to(device).float()
                #pdb.set_trace()
                vae_out = vae.decode(traj, from_layer=l, y=meta_traj)[0]['out_params'].mean
                data_outs.append(vae_out)
                trajs.append(traj)

        # plot things
        for l, data_out in enumerate(data_outs):
            if projections:
                grid = gridspec.GridSpec(2,6)
                steps = np.linspace(0, n_interp-1, 6, dtype=np.int)
                fig = plt.figure(figsize=(15, 6))
                ax = fig.add_subplot(grid[:2, :2], projection='3d')
                proj = projections[l].transform(trajs[l].cpu().detach().numpy())
                cmap = plt.cm.get_cmap('plasma', n_interp)
                proj = proj.squeeze()
                if len(proj.shape) == 3:
                    for i in range(1, n_interp-1):
                        ax.plot(proj[i,:, 0], proj[i,:, 1], proj[i,:, 2], color=cmap(i), alpha=0.3)
                    ax.plot(proj[0, :, 0], proj[0, :, 1], proj[0, :, 2], color=cmap(0))
                    ax.scatter(proj[0, 0, 0], proj[0, 0, 1], proj[0, 0, 2], color=cmap(0), marker='o')
                    ax.scatter(proj[0, -1, 0], proj[0, -1, 1], proj[0, -1, 2], color=cmap(0), marker='+')
                    ax.plot(proj[-1,:, 0], proj[-1,:, 1], proj[-1,:, 2], color=cmap(n_interp))
                    ax.scatter(proj[-1,0, 0], proj[-1,0, 1], proj[-1,0, 2], color=cmap(n_interp), marker='o')
                    ax.scatter(proj[-1,-1, 0], proj[-1,-1, 1], proj[-1,-1, 2], color=cmap(n_interp), marker='+')
                for i in range(2):
                    for j in range(3):
                        ax = fig.add_subplot(grid[i,2+j])
                        spec = data_outs[l][steps[i*2+j]].squeeze().cpu().detach().numpy()
                        ax.imshow(spec, aspect="auto")
                        ax.set_xticks([]); ax.set_yticks([])
                plt.title(f'{files_to_morph[0]}\n{files_to_morph[1]}')

            else:
                raise NotImplementedError

            if not os.path.exists(f'{out}/interpolations'):
                os.makedirs(f'{out}/interpolations')
            for i in range(len(data_out)):
                signal_out = transforms.invert(data_out[i].cpu()).clamp(-1, 1)
                ta.save('%s/interpolations/morph_%d_%d_%d.wav'%(out,l,f,i), signal_out, sr)
            fig.savefig('%s/interpolations/morph_%d_%d.pdf'%(out,l,f), format="pdf")
            plt.close('all')

