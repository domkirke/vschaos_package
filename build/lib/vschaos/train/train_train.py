import torch, numpy as np, os, gc, copy, pdb, time
from time import time, process_time
import matplotlib.pyplot as plt
from ..monitor.visualize_monitor import Monitor
from ..utils.dataloader import DataLoader
from ..utils.misc import GPULogger, sample_normalize, NaNError, print_module_grad, print_module_stats,checklist,  apply, apply_method
from .train_run import run


def rec_attr(obj, attr, *args, **kwargs):
    assert type(attr) == str
    if issubclass(type(obj), list):
        return [rec_attr(o, attr) for o in obj]
    if issubclass(type(obj), tuple):
        return tuple([rec_attr(o, attr) for o in obj])
    else:
        return getattr(obj, attr)



class Trainer(object):
    def __init__(self):
        super(Trainer, self).__init__()

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def test(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def plot(self, *args, **kwargs):
        raise NotImplementedError


def log():
    torch.cuda.synchronize()
    print(torch.cuda.memory_allocated(), torch.cuda.memory_cached())

def log_dist(msg, dist):
    with torch.no_grad():
        msg = "%s mean mean : %s \n --  mean var %s \n -- var mean %s \n -- var vars %s \n"%(msg, dist.mean.mean(0).squeeze(), dist.mean.std(0).squeeze(), dist.stddev.mean(0).squeeze(), dist.stddev.std(0).squeeze())
        msg += "\n%s mean min : %s \n --  mean max %s \n -- var min %s \n -- var max %s \n"%(msg, dist.mean.min(0)[0].squeeze(), dist.mean.max(0)[0].squeeze(), dist.stddev.min(0)[0].squeeze(), dist.stddev.max(0)[0].squeeze())
    return msg

def simple_forward(model, reinforcer=None, x=None, y=None, **kwargs):
    assert x is not None, "at least x is needed to train the model"
    x = model.format_input_data(x)
    out = model.forward(x, y=y, **kwargs)
    if reinforcer:
        out = reinforcer.forward(out, target=x, optimize=False)
    return x, out

class SimpleTrainer(Trainer):
    dataloader_class = DataLoader
    def __init__(self, models=None, datasets=None, losses=None, tasks=None, **kwargs):
        assert models is not None, "SimpleTrainer needs models"
        assert datasets is not None, "SimpleTrainer needs datasets"
        assert losses is not None, "SimpleTrainer needs losses"

        super(SimpleTrainer, self).__init__()
        self.name = kwargs.get('name', "model")
        self.models = models
        self.datasets = datasets
        self.losses = losses
        self.reinforcers = kwargs.get('reinforcers')
        self.conditioning = tasks
        self.semi_supervision = kwargs.get("semi_supervision", [])
        self.semi_sup_dropout = kwargs.get("semi_supervision_dropout", 0.0)
        self.dataloader_class = kwargs.get('dataloader', self.dataloader_class)
        self.optim_balance = kwargs.get('optim_balance')
        self.scheduled_params = kwargs.get('scheduled_params', {})
        # additional args
        self.trace_mode = kwargs.get('trace_mode', 'epoch')
        self.device = kwargs.get('device')
        self.loader_args = kwargs.get('loader_args', dict())
        self.split = kwargs.get('split', False)
        self.run = run
        self.forward_hook = simple_forward
        # plot args
        plot_tasks = kwargs.get('plot_tasks', tasks)
        self.plots = kwargs.get('plots', {})
        self.synth = kwargs.get('synth', {})
        self.plot_grads = True
        # init monitors
        use_tensorboard = kwargs.get('use_tensorboard')
        total_losses = checklist(losses) + [self.reinforcers] if self.reinforcers is not None else checklist(losses)
        self.init_monitors(models, datasets, total_losses, self.tasks, plots=self.plots, synth=self.synth, plot_tasks=plot_tasks, use_tensorboard=use_tensorboard)
        self.best_model = None
        self.best_test_loss = np.inf
        self.logger = GPULogger(kwargs.get('export_profile',None), kwargs.get('verbose', False))

    @property
    def tasks(self):
        return list(set((self.conditioning or []) + (self.semi_supervision or [])))

    def init_monitors(self, models, datasets, losses, tasks, plots=None, synth=None, plot_tasks=None, use_tensorboard=None):
        self.monitor = Monitor(models, datasets, losses, tasks, plots=plots, synth=synth, tasks = plot_tasks, use_tensorboard=use_tensorboard)

    def init_time(self):
        self.start_time = process_time()

    def get_time(self):
        return process_time() - self.start_time

    def get_dataloader(self, datasets, batch_size=64, partition=None, tasks=None, batch_catch_size=1, **kwargs):
        return self.dataloader_class(datasets, batch_size=batch_size, tasks=self.tasks, partition=partition, **self.loader_args)

    def monitor_grads(self, models, loss):
        self.monitor.plot_grads(models, loss)

    def train(self, partition=None, write=False, batch_size=64, tasks=None, batch_cache_size=1, **kwargs):
        # set models and partition to train
        apply_method(self.models, 'train')
        if partition is None:
            partition = 'train'
        # get dataloader
        loader = self.get_dataloader(self.datasets, batch_size=batch_size, partition=partition, tasks=tasks, batch_cache_size=batch_cache_size)
        # run
        full_loss, full_losses = self.run(self, loader, period="train", plot=False, **kwargs)
        # write losses in loss objects
        if self.trace_mode == "epoch" and write:
            if issubclass(type(self.losses), (list, tuple)):
                [apply_method(self.losses[i], 'write', 'train', full_losses['main_losses'][i], time=self.get_time()) for i in range(len(self.losses))]
            else:
                apply_method(self.losses, 'write', 'train', full_losses['main_losses'], time=self.get_time())
            if self.reinforcers:
                self.reinforcers.write('train', full_losses['reinforcement_losses'], time=self.get_time())
            self.monitor.update(kwargs.get('epoch'))
        return full_loss, full_losses

    def test(self, partition=None, write=False, tasks=None, batch_size=None, batch_cache_size=1, **kwargs):
        # set models and partition to test
        apply_method(self.models, 'eval')
        if partition is None:
            partition = 'test'
        # get dataloader and run without gradients
        with torch.no_grad():
            apply_method(self.losses, "unsupervised")
            loader = self.get_dataloader(self.datasets, batch_size=batch_size, tasks=tasks, batch_cache_size=batch_cache_size, partition=partition)
            full_loss, full_losses = self.run(self, loader, optimize=False, schedule=True, period="test", plot=True, track_loss=True, **kwargs)
        # write losses in loss objects
        if write:
            if issubclass(type(self.losses), (list, tuple)):
                [apply_method(self.losses[i], 'write', 'test', full_losses['main_losses'][i], time=self.get_time()) for i in range(len(self.losses))]
            else:
                apply_method(self.losses, 'write', 'test', full_losses['main_losses'], time=self.get_time())
            self.monitor.update(kwargs.get('epoch'))
            if self.reinforcers:
                self.reinforcers.write('test', full_losses['reinforcement_losses'], time=self.get_time())
            self.monitor.update(kwargs.get('epoch'))
        # check if current model is the best model
        if full_loss < self.best_test_loss:
            self.best_model = self.get_best_model()
        return full_loss, full_losses

    def optimize(self, models, loss, epoch=None, batch=None):
        #pdb.set_trace()
        update_model = True if self.optim_balance is None else batch % self.optim_balance[0] == 0
        update_loss = True if self.optim_balance is None else batch % self.optim_balance[1] == 0
        if self.plot_grads:
            self.monitor_grads(models, loss)
        if update_model:
            apply_method(self.models, 'step', loss)
        if update_loss:
            apply_method(self.losses, 'step', loss)
        # print_grad_stats(self.models)

    def update_scheduled(self, epoch, period):
        for k, v in self.scheduled_params.items():
            v.update(epoch=epoch, period=period)
            print('-- updated %s : %f'%(k, float(v)))

    def get_best_model(self, models=None, **kwargs):
        # get objects to save
        models = checklist(models or self.models)

        best_model = []
        for i,model in enumerate(models):
            current_device = next(model.parameters()).device
            # move best model to cpu to free GPU memory
            apply_method(model, 'cpu')
            if self.reinforcers:
                kwargs['reinforcers'] = self.reinforcers
            best_model.append(apply(apply_method(model, 'get_dict', **kwargs), copy.deepcopy))

            if current_device != torch.device('cpu'):
                with torch.cuda.device(current_device):
                    apply_method(model, 'cuda')
        return best_model

    def save(self, results_folder, models=None, save_best=True, **kwargs):
        # saving current model
        if models is None:
            models = self.models
        name = str(self.name)
        epoch = kwargs.get('epoch')
        print('-- saving model at %s'%'results/%s/%s.pth'%(results_folder, name))
        if not issubclass(type(models), list):
            models = [models]
        datasets = self.datasets
        if not issubclass(type(datasets), list):
            datasets = [datasets]
        partitions = rec_attr(datasets, "partitions")
        for i in range(len(models)):
            current_name = name if len(models) == 1 else '/vae_%d/%s'%(i, name)
            if not os.path.isdir(results_folder+'/vae_%d'%i):
                os.makedirs(results_folder+'/vae_%d'%i)
            if kwargs.get('epoch') is not None:
                current_name = current_name + '_' + str(kwargs['epoch'])
            additional_args = {'loss':self.losses, 'partitions':partitions, **kwargs}
            if self.reinforcers:
                additional_args['reinforcers'] = self.reinforcers
            models[i].save('%s/%s.pth'%(results_folder, current_name), **additional_args)

        # saving best model
        best_model = self.best_model
        if not issubclass(type(best_model), list):
            best_model = [best_model]
        if not self.best_model is None and save_best:
            print('-- saving best model at %s'%'results/%s/%s_best.pth'%(results_folder, name))
            for i in range(len(best_model)):
                current_name = name+'_best' if len(models) == 1 else '/vae_%d/%s_best'%(i, name)
                torch.save({'loss':self.losses, 'partitions':partitions, **kwargs, **best_model[i]}, '%s/%s.pth'%(results_folder, current_name))

    def plot(self, figures_folder, epoch=None, **kwargs):
        if epoch is not None:
            apply_method(self.monitor, 'update', epoch)
        plt.close('all')
        apply_method(self.monitor, 'plot', out=figures_folder, epoch=epoch, loader=self.dataloader_class, trainer=self)
        apply_method(self.monitor, 'synthesize', out=figures_folder, epoch=epoch, loader=self.dataloader_class, trainer=self)

def train_model(trainer,  options={}, save_with={}, **kwargs):
    epochs = options.get('epochs', 3000) # number of epochs
    save_epochs = options.get('save_epochs', 100) # frequency of model saving
    plot_epochs = options.get('plot_epochs', 100) # frequency of plotting
    batch_size = options.get('batch_size', 64) # batch size
    batch_split = options.get('batch_split')
    remote = options.get('remote', None) # automatic scp transfer

    # Setting results & plotting directories
    results_folder = options.get('results_folder', 'saves/' + trainer.name)
    figures_folder = options.get('figures_folder', results_folder+'/figures')
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)
    if not os.path.isdir(figures_folder):
        os.makedirs(figures_folder)

    # Init training
    epoch = options.get('first_epoch', 0)
    trainer.init_time()
    while epoch < epochs:
        train_loss, train_losses = trainer.train(epoch=epoch, batch_size=batch_size, batch_split=batch_split, write=True, **kwargs)
        with torch.no_grad():
            test_loss, test_losses = trainer.test(epoch=epoch, batch_size=batch_size, batch_split=batch_split, write=True, **kwargs)

        if epoch%save_epochs==0 or epoch == (epochs-1):
            trainer.save(results_folder, epoch=epoch, **save_with)

        # plot
        if epoch % plot_epochs == 0:
            trainer.plot(figures_folder, epoch=epoch, **options)
            if remote is not None:
                remote_folder= remote+'/figures_'+trainer.name
                print('scp -r %s %s:'%(figures_folder, remote_folder))
                os.system('scp -r %s %s:'%(figures_folder, remote_folder))

        # clean
        gc.collect(); gc.collect()

        # do it again
        epoch += 1


