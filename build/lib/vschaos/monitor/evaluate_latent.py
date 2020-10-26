import random, pdb, numpy as np, matplotlib.pyplot as plt, time
import torch, torch.nn as nn
from ..distributions.distribution_priors import get_default_distribution
from ..criterions.criterion_divergence import KLD, ReverseKLD, RD, JSD, MMD
from ..criterions.criterion_tc import TotalCovariance
from .evaluate_evalute import Evaluation, EvaluationContainer
from ..utils.misc import checklist, check_dir



class LatentEvaluation(Evaluation):
    divergences = [KLD(), ReverseKLD(), RD(alpha=2), JSD(alpha=0.5), MMD(max_kernel=400), TotalCovariance()]


    def parse_params(self, eval_params={}, stats=True, **kwargs):
        self.stats=stats
        self.divergences = eval_params.get('divergences') or self.divergences

    def get_divergences(self, out, latent_params, **kwargs):
        divergences = {k:[] for k in self.divergences}
        n_layers = len(checklist(latent_params))
        q = out['z_params_enc']; p =  out['z_params_dec'] or []
        div_results = {}
        inputs = []
        for l in range(n_layers):
            q_l = q[l]
            if l < len(p):
                if p[l] is not None:
                    p_l = p[l]
                else:
                    p_l = latent_params[l].get('prior') or get_default_distribution(latent_params[l]['dist'], q_l.batch_shape)
            else:
                p_l = latent_params[l].get('prior') or get_default_distribution(latent_params[l]['dist'], q_l.batch_shape)
            inputs.append({'params1':q_l, 'params2':p_l})

            
        for div in self.divergences:
            print('   - computing %s...'%div)
            div_out = [div(**inputs[i], compute_logdets=True, **kwargs)[1] for i in range(len(inputs))]
            div_results = {**div.get_named_losses(div_out), **div_results}

        return div_results

    def evaluate(self, outputs, model=None, **kwargs):
        print("-- start latent evaluation")
        divergences =  self.get_divergences(outputs, model.platent, **kwargs)
        global_stats = {'enc': [], 'dec': []}
        out = {**divergences}
        if self.stats:
            for l, latent_out in enumerate(outputs['z_params_enc']):
                dim_shape = tuple(range(latent_out.mean.ndimension()))
                latent_mean_mean = latent_out.mean.mean(dim_shape[:-1])
                latent_mean_std = latent_out.mean.std(dim_shape[:-1])
                latent_std_mean = latent_out.stddev.mean(dim_shape[:-1])
                latent_std_std = latent_out.stddev.std(dim_shape[:-1])
                global_stats['enc'].append([latent_mean_mean, latent_mean_std, latent_std_mean, latent_std_std])

            if outputs.get('z_params_dec'):
                for l, latent_out in enumerate(outputs['z_params_dec']):
                    if latent_out is None:
                        continue
                    dim_shape = tuple(range(latent_out.mean.ndimension()))
                    latent_mean_mean = latent_out.mean.mean(dim_shape[:-1])
                    latent_mean_std = latent_out.mean.std(dim_shape[:-1])
                    latent_std_mean = latent_out.stddev.mean(dim_shape[:-1])
                    latent_std_std = latent_out.stddev.std(dim_shape[:-1])
                    global_stats['dec'].append([latent_mean_mean, latent_mean_std, latent_std_mean, latent_std_std])
            out = {**out, 'stats':global_stats}
        return out







class DisentanglementEvaluation(LatentEvaluation):

    def parse_params(self, eval_params={}, dis_params={}, latent_params={}, **kwargs):
        self.tasks = dis_params.get('tasks')
        self.cuda = dis_params.get('cuda', -1)
        if issubclass(type(self.cuda), list):
            if len(self.cuda) != 0:
                self.cuda = self.cuda[0]
            else:
                self.cuda = -1
        self.label_params = dis_params.get('label_params')
        self.optim_params = dis_params.get('optim_params', {})
        self.discriminators = self.init_discriminators(latent_params, dis_params, cuda=self.cuda)
        self.init_optimizers(self.optim_params)

    def init_discriminators(self, latent_params, dis_params, cuda=-1):
        tasks = dis_params['tasks']
        label_params = dis_params['label_params']
        discriminators = []
        for t in range(len(tasks)):
            discriminators.append(self.make_discriminator(latent_params[-1]['dim'], label_params[t]['dim']))
            if cuda >= 0:
                discriminators[-1] = discriminators[-1].cuda(cuda)
        return discriminators

    def init_optimizers(self, optim_params):
        # init optimizers
        self.optimizers = []
        for i,t in enumerate(self.tasks):
            self.optimizers.append(getattr(torch.optim, optim_params.get('optim_type','Adam'))(self.discriminators[i].parameters(), lr=1e-3))

    def make_discriminator(self, latent_input, label_input):
        return nn.Sequential(nn.Linear(latent_input, label_input))

    def evaluate(self, outputs, target=None, y=None, model=None, out=None, **kwargs):
        # init optimizers
        n_tasks = len(self.tasks)
        n_epochs = self.optim_params.get('epochs', 1000)
        output_path = out or './'

        metric_scores = {k: [None]*self.label_params[i]['dim'] for i, k in enumerate(self.tasks)}
        check_dir(output_path+'/dis_discriminators')
        present_idxs = {t: torch.unique(y[t]) for t in self.tasks}
        batch_size = 64

        history = {t:[] for t in self.tasks}
        with torch.cuda.device(self.cuda):
            for epoch in range(n_epochs):
                # pick random tasks
                current_task = random.randrange(n_tasks)
                z_diffs = []; current_labels = []
                time_in = time.process_time()
                print('hello', time_in)
                present_labels = present_idxs[self.tasks[current_task]]
                label_idx = torch.randint(len(present_labels),torch.Size([1]))
                current_label = present_labels[label_idx]
                idxs = torch.nonzero(y[self.tasks[current_task]] == current_label)
                if idxs.shape[0] == 0:
                    epoch -= 1
                    continue
                idxs = idxs[torch.randperm(idxs.shape[0])][:(idxs.shape[0]//2)*2]
                idxs = [idxs[:idxs.shape[0]//2], idxs[idxs.shape[0]//2:]]
                current_out = outputs['z_params_enc'][-1].mean
                z_diff = torch.abs(current_out[idxs[0]] - current_out[idxs[1]]).mean(0)

                #z_diffs.zero_()
                #for i in range(z_diffs.shape[0]):
                #    z_diffs[i, current_labels[i]] = 1
                if self.cuda >= 0:
                    z_diff = z_diff.cuda(); current_label = current_label.cuda()
                pred_out = self.discriminators[current_task](z_diff)
                #print(pred_out.shape[-1], current_labels.max(), self.tasks[current_task])
                loss = nn.functional.cross_entropy(pred_out, current_label.long())
                loss.backward()
                self.optimizers[current_task].step()
                #print('epoch %d / %d ; task %s ; label %s ; classification result : %s'%(epoch, n_epochs, self.tasks[current_task], current_label, loss))
                if epoch % 200 == 0:
                    print('epoch %d / %d : %f'%(epoch, n_epochs, loss))
                if epoch % 1000 == 0:
                    for t in range(len(self.tasks)):
                        torch.save(self.discriminators[current_task], '%s/dis_discriminators/discriminator_%s.t7'%(output_path, self.tasks[current_task]))
                if epoch % 10:
                    plt.close('all')


                    
                self.optimizers[current_task].zero_grad()

            # final evaluation
            data_in = outputs['z_params_enc'][-1].mean
            final_tasks = self.tasks + [t+'_random' for t in self.tasks]
            losses = {t:None for t in self.tasks}
            for t, task in enumerate(self.tasks):
                with torch.no_grad():
                    loss = 0; random_loss = 0
                    self.discriminators[t].eval()
                    if self.cuda > -1:
                        data_in = data_in.cuda()
                    disc_out = self.discriminators[t](data_in)
                    if self.cuda >= 0:
                        disc_out = disc_out.cpu()
                    loss =  loss + float(nn.functional.cross_entropy(disc_out, y[task].long()))
                    random_labels = torch.randint_like(y[task].long(), self.label_params[t]['dim'])
                    random_loss = float(nn.functional.cross_entropy(disc_out, random_labels))
                losses[task] = loss
                losses[task+'_random'] = random_loss

        # save measures
        torch.save(metric_scores, '%s/dis_discriminators/metric_scores.t7'%output_path)
        for i, disc in enumerate(self.discriminators):
            torch.save(disc, '%s/dis_discriminators/discriminator_%s.t7'%(output_path, self.tasks[i]))

        return losses






