#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 14:03:10 2018

@author: chemla
"""
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from numpy import fromstring
import torch, os
import numpy as np
import matplotlib.pyplot as plt
from . import visualize_plotting as lplt
from .audio_synthesize import resynthesize_files, interpolate_files
from .audio_descriptor import plot2Ddescriptors,plot3Ddescriptors
from ..utils import apply_method, checklist
from torch.utils.tensorboard import SummaryWriter
import collections



plot_hash = {'reconstructions': lplt.plot_reconstructions,
             'latent_space': lplt.plot_latent,
             'latent_trajs': lplt.plot_latent_trajs,
             'latent_dims': lplt.plot_latent_dim,
             'sample': lplt.plot_samples,
             'confusion':lplt.plot_confusion,
             'latent_consistency': lplt.plot_latent_consistency,
             'statistics':lplt.plot_latent_stats,
             'images':lplt.image_export,
             'conv_weights': lplt.plot_conv_weights,
             'losses': lplt.plot_losses,
             'class_losses':lplt.plot_class_losses,
             'grid_latent':lplt.grid_latent,
             'descriptors_2d':plot2Ddescriptors,
             'descriptors_3d':plot3Ddescriptors,
             'prediction':lplt.plot_prediction,
             'audio_reconstructions': resynthesize_files,
             'audio_interpolate': interpolate_files}


def dict_merge(dct, merge_dct):

    for k, v in merge_dct.iteritems():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]

class TensorboardHandler(object):
    def __init__(self, path="runs"):
        os.system(f'rm -rf {path}')
        self.writer = SummaryWriter(path)
        self._counter = 0
        self.models = []
        self.losses = []

    def update(self, models, losses, epoch=None):
        if epoch is None:
            epoch = self._counter
        apply_method(self, "add_model", models, epoch=epoch)
        apply_method(self, "add_learning_rate", models, epoch=epoch)
        apply_method(self, "add_loss", losses, epoch=epoch)
        self._counter += 1

    def add_model(self, model, epoch):
        # self.writer.add_graph(model, verbose=True)
        for n, p in model.named_parameters():
            self.writer.add_histogram('model/params/'+n,p.detach().cpu().numpy(),epoch)

    def add_model_grads(self, model, epoch):
        for n, p in model.named_parameters():
            if p.grad is not None:
                self.writer.add_histogram('model/grads/'+n,p.grad.detach().cpu().numpy(),epoch)
            else:
                print('{Warning} %s has no gradient'%n)

    def add_loss(self, losses, epoch):
        for loss in losses:
            loss_names = list(loss.loss_history[list(loss.loss_history.keys())[0]].keys())
            for l in loss_names:
                # write last losses
                # pdb.set_trace()
                try:
                    last_loss = {k: np.array(v[l]['values'][-1]) if len(v) > 0 else 0. for k, v in loss.loss_history.items()}
                    self.writer.add_scalars('loss_%s/' % l, last_loss, epoch)
                except Exception as e:
                    print('hello')
                    raise e

    def add_learning_rate(self, models, epoch):
        if not hasattr(models, "optimizers"):
            return
        for k, v in models.optimizers.items():
            learning_rates = {'k'+str(i):o['lr'] for i, o in enumerate(v.param_groups)}
        self.writer.add_scalars('learning_rates', learning_rates)


    def add_image(self, name, pic, epoch):
        self.writer.add_image(name, pic, epoch)

    def add_audio(self, name, audio, sr, epoch):
        if len(audio.shape) > 1:
            audio = audio.mean(0)
        self.writer.add_audio(name, audio, epoch, sr)



class Monitor(object):
    def __init__(self, model, dataset, loss, labels, plots={}, synth={}, partitions=['train', 'test'],
                 output_folder=None, tasks=None, use_tensorboard=None, **kwargs):
        self.model = model
        self.dataset = dataset
        self.loss = loss
        self.labels = labels
        self.tasks = tasks if tasks else None
        self.use_tensorboard = use_tensorboard
        self.output_folder = output_folder if output_folder else None
        self.plots = plots
        self.synth = synth
        self.partitions = partitions
        self.writer = None
        if use_tensorboard:
            self.writer = TensorboardHandler(use_tensorboard)
            
    def update(self, epoch=None):
        if self.use_tensorboard:
            self.writer.update(self.model, self.loss, epoch=epoch)

    def close(self):
        pass

    def record_image(self, image_list, partition=None, epoch=None):
        if not self.use_tensorboard:
            return
        for name, fig in image_list.items():
            cv = FigureCanvas(fig)
            cv.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            image = fromstring(cv.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            image = np.transpose(image, (2,0,1)) / 255
            if partition:
                name = str(name)+'_%s'%partition
            if self.writer:
                self.writer.add_image(name, image, epoch)

    def plot_grads(self, model, epoch=None):
        if self.writer:
            self.writer.add_model_grads(model, epoch=epoch)

    def plot(self, out=None, epoch=None, loader=None, trainer=None, **kwargs):
        # plot reconstructions
        if out is None:
            out = self.output_folder
        plt.close('all')

        for plot, plot_args in self.plots.items():
            plot_args = checklist(plot_args)
            plot_args = [dict(p) for p in plot_args]

            if plot_hash[plot].partition_dependent:
                for partition in self.partitions:
                    output_name = None if out is None else plot
                    for pa in plot_args:
                        print('--monitor : %s, %s, partition : %s' % (plot, pa, partition))
                        pa = dict(pa)
                        pa['name'] = output_name+'_'+pa.get('name', '')

                        fig, axes = plot_hash[plot](self.dataset, self.model, loader=loader,# preprocessing=preprocessing,
                                                    trainer=trainer, partition=partition, out=out, epoch=epoch, **pa)
                        self.record_image(fig, epoch=epoch, partition=partition)
                        plt.close('all')
            else:
                output_name = None if out is None else plot
                for pa in plot_args:
                    print('--monitor : %s, %s' % (plot, pa))
                    pa = dict(pa)
                    pa['name'] = output_name + '_' + pa.get('name', '')

                    fig, axes = plot_hash[plot](self.dataset, self.model, loader=loader,  # preprocessing=preprocessing,
                                                trainer=trainer, partition=None, out=out, epoch=epoch, **pa)
                    self.record_image(fig, epoch=epoch, partition=None)
                    plt.close('all')

    def synthesize(self, out=None, epoch=None, preprocessing=None, loader=None, trainer=None, **kwargs):
        # plot reconstructions
        if out is None:
            out = self.output_folder
        plt.close('all')

        for plot, plot_args in self.synth.items():
            plot_args = checklist(plot_args)
            plot_args = [dict(p) for p in plot_args]
            print('--monitor : %s, %s'%(plot, plot_args))

            for partition in self.partitions:
                if issubclass(type(self.model), (list, tuple)):
                    for i in range(len(self.model)):
                        output_name = None if out is None else '/%s_%s_%s_%s'%(plot, partition, epoch, i)
                        dataset = self.dataset if not issubclass(type(self.dataset), (list, tuple)) else self.dataset[i]
                        if issubclass(type(losses), (list, tuple)):
                            plot_args['loss'] = losses[i]
                        if issubclass(type(reinforcers), (list, tuple)):
                            plot_args['reinforcers'] = reinforcers[i]
                        audio, sr = plot_hash[plot](dataset, self.model[i], loader=loader,
                                                    trainer=trainer, partition=partition, out=out, name=output_name, **plot_args)
                        if self.writer:
                            self.writer.add_audio(output_name, audio, sr)

                else:
                    output_name = None if out is None else plot
                    for pa in plot_args:
                        losses = pa.get('loss')
                        reinforcers = pa.get('reinforcers')
                        current_name = output_name+'_'+pa.get('name', '')
                        if pa.get('name'):
                            current_name = current_name+'_'+pa['name']
                            del pa['name']
                        audio, sr = plot_hash[plot](self.dataset, self.model, loader=loader,# preprocessing=preprocessing,
                                                    trainer=trainer, partition=partition, name=current_name, out=out,epoch=epoch, **pa)
                        if self.writer:
                            original = torch.cat(list(audio['original'].values()), -1)
                            resynth = torch.cat(list(audio['resynth'].values()), -1)
                            self.writer.add_audio('original_%s'%partition, original, sr, epoch)
                            self.writer.add_audio('resynth_%s'%partition, resynth, sr, epoch)

        # if image_export:
        #     output_name = None if out is None else out+'/grid%s'%epoch_suffix
        #     fig, axes = lplt.image_export(self.dataset, self.model, self.labels, out=output_name, ids=reconstruction_ids)
        #     self.record_image(fig, 'images', epoch)

