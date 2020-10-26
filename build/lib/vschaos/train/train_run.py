import torch, gc, itertools
from tqdm import tqdm
from ..utils.onehot import fromOneHot
from ..utils.misc import sample_normalize, NaNError, apply_method, pdb, merge_dicts, accumulate_losses, normalize_losses, checklist


def recursive_add(obj, factor=1.):
    if issubclass(type(obj[0]), (list, tuple)):
        return [recursive_add([obj[j][i] for j in range(len(obj))], factor=factor) for i in range(len(obj[0]))]
    else:
        return sum(obj)*factor


def get_metadata(y, dataset, conditioning, semi_supervised, semi_sup_dropout=0.0, batch_size=None):
    y_out = {}
    if y == {}:
        return {}, False
    batch_size = batch_size or list(y.values())[0].shape[0]
    sup = True
    if conditioning is not None:
        y_out = {m: y.get(m) for m in conditioning}
    if semi_supervised is not None:
        meta_dropout = torch.bernoulli(torch.tensor(float(semi_sup_dropout))).item()
        # print(meta_dropout)
        if meta_dropout == 0:
            y_out = {**{m: y.get(m) for m in semi_supervised}, **y_out}
        else:
            classes = {t:dataset.classes.get(t) for t in semi_supervised}
            params = tuple([list(range(classes[t].get('_length') or len(classes[t].values()))) for t in classes.keys()])
            full_y = []
            for z in itertools.product(*params):
                semi_sup_y = {semi_supervised[i]:torch.Tensor([z[i]]*batch_size) for i in range(len(semi_supervised))}
                full_y.append({**y_out, **semi_sup_y})
            y_out = full_y
            sup = False
    return y_out, sup


def run(self, loader, epoch=None, optimize=True, schedule=False, period=None, plot=None, **kwargs):
    # train phase
    batch = 0; current_loss = 0;
    self.logger('start epoch')
    train_losses = {'main_losses':[]}
    if self.reinforcers:
        train_losses['reinforcement_losses'] = []
    self.plot_grads = True
    self.update_scheduled(epoch=epoch, period=period)

    num_batches = len(loader)//loader.batch_size+1
    for x,y in loader:
        # forward
        self.logger('data loaded')
        if kwargs.get('sample_norm'):
            x = sample_normalize(x)

        y, sup = get_metadata(y, self.datasets, self.conditioning, self.semi_supervision, self.semi_sup_dropout)
        y = checklist(y)
        if sup and period != "test":
            apply_method(self.losses, "supervised")
        else:
            apply_method(self.losses, "unsupervised")
        batch_loss = 0.
        losses = []
        for y_tmp in y:
            try:
                #x = self.models.format_input_data(x)
                x, out = self.forward_hook(model=self.models, reinforcer=self.reinforcers,
                                           x=x, y=y_tmp, epoch=epoch, batch=batch, period=period, **kwargs)
                # compute loss
                self.logger('data forwarded')
                batch_loss_tmp, losses_tmp = self.losses.loss(model=self.models, out=out, target=x, y=y_tmp, epoch=epoch, plot=plot and not batch, period=period)
                batch_loss = batch_loss + batch_loss_tmp
                losses.append(losses_tmp)
            except NaNError:
                pdb.set_trace()
        losses = accumulate_losses(losses)
        train_losses['main_losses'].append(losses)
        # print_stats(out['z_enc'][0])
        #print_stats(out['z_preflow_enc'][0])

        # learn
        self.logger('loss computed')
        if optimize:
            batch_loss.backward(retain_graph=False)
            self.optimize(self.models, batch_loss, epoch=epoch, batch=batch)
            self.plot_grads = False

        # learn reinforcers in case
        if self.reinforcers:
            _, reinforcement_losses = self.reinforcers(out, target=x, epoch=epoch, optimize=optimize)
            train_losses['reinforcement_losses'].append(reinforcement_losses)

            self.logger('optimization done')

        # trace
        if self.trace_mode == "batch":
            if period is None:
                period = "train" if optimize else "test"
            apply_method(self.losses, "write", period, losses)
            apply_method(self.monitor, "update")
            self.logger("monitor updated")

        # update loop
        named_losses = self.losses.get_named_losses(losses)
        desc = str(named_losses)
        if self.reinforcers:
            named_losses = {**named_losses, **self.reinforcers.get_named_losses(reinforcement_losses)}
        # print("epoch %d / batch %d / full loss: %s / losses : %s "%(epoch, batch, batch_loss, named_losses), end="\r", flush=True)

        print_end = "\n" if batch == num_batches-1 else '\r'
        print(f"epoch {epoch} ({period}) {batch+1}/{num_batches} {named_losses} total : {batch_loss.item()}", end=print_end, flush=True)

        if kwargs.get('track_loss'):
            current_loss = current_loss + float(batch_loss)
        else:
            current_loss += float(batch_loss)
        batch += 1
        del out; del x

    current_loss /= batch
    try:
        for k in train_losses.keys():
            train_losses[k] = accumulate_losses(train_losses[k], weight = 1/len(train_losses[k]))
    except IndexError as e:
        pdb.set_trace()
    # scheduling the training
    if schedule:
        apply_method(self.models, "schedule", current_loss)
    # cleaning cuda stuff
    gc.collect(); gc.collect()
    self.logger("cuda cleaning done")

    return current_loss, train_losses





def run_scan(self, loader, preprocessing=None, epoch=None, optimize=True, schedule=False, period=None, plot=None, **kwargs):
    # train phase
    batch = 0; current_loss = 0;
    train_losses = None;
    if preprocessing is None:
        preprocessing = self.preprocessing
    self.logger('start epoch')
    full_losses = None
    for x,y in loader:
        # prepare audio data
        self.logger('data loaded')
        if preprocessing[0] is not None:
            x = preprocessing[0](x)
        if kwargs.get('sample_norm'):
            x = sample_normalize(x)
        self.logger('data preprocessed')
        # prepare symbol data
        x_sym = [y.get(s) for s in self.symbols]
        if preprocessing[1] is not None:
            x_sym = preprocessing[1](x_sym)
        try:
            out_audio = self.models[0].forward(x, y=y, epoch=epoch)
            if self.reinforcers[0]:
                out_audio = self.reinforcers[0].forward(out_audio)

            out_symbol = self.models[1].forward(x_sym, y=y, epoch=epoch)
                            #self.logger(log_dist("latent", out['z_params_enc'][-1]))
            #self.logger(log_dist("data", out['x_params']))
            # compute loss
            self.logger('data forwarded')
            audio_batch_loss, audio_losses = self.losses[0].loss(model=self.models[0], out=out_audio, target=x, epoch=epoch, plot=plot and not batch, period=period)
            symbol_batch_loss, symbol_losses = self.losses[1].loss(model=self.models[1], out=out_symbol, target=[fromOneHot(x) for x in x_sym], epoch=epoch, plot=plot and not batch, period=period)
            # compute transfer loss
            transfer_loss, transfer_losses = self.get_transfer_loss(out_audio, out_symbol)

            batch_loss = audio_batch_loss + symbol_batch_loss + transfer_loss
            losses = [audio_losses, symbol_losses, transfer_losses]
            train_losses = losses if train_losses is None else train_losses + losses
        except NaNError:
            pdb.set_trace()

        # trace
        if self.trace_mode == "batch":
            if period is None:
                period = "train" if optimize else "test"
            apply_method(self.losses, "write", period, losses)
            apply_method(self.monitor, "update")
            self.logger("monitor updated")

        # learn
        self.logger('loss computed')
        if optimize:
            batch_loss.backward()
            self.optimize(self.models, batch_loss)
        if self.reinforcers[0]:
            _, reinforcement_losses = self.reinforcers[0](out_audio, target=x, epoch=epoch, optimize=optimize)

        self.logger('optimization done')
        # update loop
        named_losses_audio = self.losses[0].get_named_losses(losses[0])
        named_losses_symbol = self.losses[1].get_named_losses(losses[1])
        named_losses_transfer = self.transfer_loss.get_named_losses(losses[2])

        print("epoch %d / batch %d \nfull loss: %s"%(epoch, batch, batch_loss))
        print('audio : %s, %s'%(audio_batch_loss, named_losses_audio))
        print('symbol : %s, %s'%(symbol_batch_loss, named_losses_symbol))
        print('transfer : %s, %s'%(transfer_loss, named_losses_transfer))

        if kwargs.get('track_loss'):
            current_loss = current_loss + batch_loss
        else:
            current_loss += float(batch_loss)
        batch += 1
        if full_losses is None:
            full_losses = losses
        else:
            full_losses = accumulate_losses([full_losses, losses])
        del out_audio; del out_symbol; del x

    current_loss /= batch
    normalize_losses(full_losses, batch)
    # scheduling the training
    if schedule:
        apply_method(self.models, "schedule", current_loss)
    # cleaning cuda stuff
    torch.cuda.empty_cache()
    gc.collect(); gc.collect()
    self.logger("cuda cleaning done")

    return current_loss, full_losses
