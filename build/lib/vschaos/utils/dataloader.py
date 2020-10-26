# -*- coding: utf-8 -*-

#import pdb
#pdb.set_trace()
import torch
import random
from numpy.random import permutation
import numpy as np
import pdb
from .misc import GPULogger
from ..data.data_asynchronous import OfflineDataList


def length(array):
    if issubclass(type(array), np.ndarray) or issubclass(type(array), OfflineDataList):
        return array.shape[0]
    elif issubclass(type(array), list):
        return len(array)

# def pack_sequence(sample, is_sequence):
#     if not is_sequence:
#         return sample
#     lens = sum([(sample[i].shape[0],) for i in range(len(sample))], tuple())
#     sample_ordered = sample[np.argsort(lens)]
#     sample_ordered = [t for t in reversed(sample_ordered)]
#     batch_sequences = torch.nn.utils.rnn.pad_sequence(sample_ordered, batch_first=True)
#     return batch_sequences

class DataLoader(object):
    is_sequence = False
    num_workers = 0
    def __init__(self, dataset, batch_size, conditioning=None, semi_sup=None, partition=None, ids=None, is_sequence=None,
                 pin_memory=False, meta_dropout=0.0, num_workers=0, shuffle=True, *args, **kwargs):
        self.dataset = dataset
        self.conditinoning = conditioning
        if partition is not None:
            dataset = dataset.retrieve(partition)

        if ids is not None:
            dataset = dataset.retrieve(ids)

        if batch_size is None:
            batch_size = len(dataset.data)
        self.batch_size = batch_size

        if is_sequence:
            self.is_sequence = is_sequence
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=False,
                                                      shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    def __len__(self):
        return len(self.dataset)

    def pick_data(self, batch_id):
        isseq = self.is_sequence
        if issubclass(type(self.dataset.data), OfflineDataList):
            current_data = [self.dataset.data[i] for i in self.random_ids[batch_id]]
        elif issubclass(type(self.dataset.data), list):
            current_data = [[self.dataset.data[i][d] for d in range(len(self.dataset.data[i]))] for i in self.random_ids[batch_id]]
        else:
            current_data = [self.dataset.data[i] for i in self.random_ids[batch_id]]
        return current_data

    def __iter__(self):
        for x, y in self.dataloader:
            yield x, y


        # for batch_id in range(len(self.random_ids)):
        #     # load current cache
        #     current_data = self.pick_data(batch_id)
        #     for b, ids in enumerate(self.random_ids[batch_id]):
        #         self.current_ids = ids
        #         x = current_data[b]
        #         if not self.tasks is None:
        #             y = {t:self.dataset.metadata[t][self.current_ids] for t in self.tasks}
        #             yield x, y
        #         else:
        #             yield x, None

#                yield self.transform(self.dataset.data[self.random_ids[i]]), None


# Can load randomly from several datasets
def MultiDataLoader(datasets):
    class MultiDataLoader(DataLoader):
        def __init__(self, dataset, batch_size, tasks=None, partition=None, ids=None, is_sequence=None,
                     pin_memory=False, num_workers=0, preprocessing=None, shuffle=True, *args, **kwargs):
            self.dataset = datasets
            self.preprocessing = preprocessing
            if partition is not None:
                self.dataset = [d.retrieve(partition) for d in self.dataset]
            if ids is not None:
                self.dataset = [d.retrieve(ids) for d in self.dataset]
            if batch_size is None:
                batch_size = len(dataset.data)
            self.batch_size = batch_size

            if is_sequence:
                self.is_sequence = is_sequence

            num_workers = kwargs.get('num_workers', 0) or self.num_workers
            loader_args = {'pin_memory': pin_memory, 'num_workers': num_workers or 0}

            self.dataloader = [self.get_dataloader(self.dataset[i], batch_size=batch_size, shuffle=shuffle, drop_last=False, **loader_args) for i in range(len(self.dataset))]

            # metadata to retrieve
            self.tasks = tasks

        def __len__(self):
            return sum([len(d) for d in self.dataset])


        def get_dataloader(self, dataset, **kwargs):
            return iter(torch.utils.data.DataLoader(dataset, **kwargs))


        def __iter__(self):
            while True:
                try:
                    idx = random.randrange(len(self.dataloader))
                    x, y = next(self.dataloader[idx])
                    yield x, y
                except StopIteration:
                    del self.dataloader[idx]
                    if len(self.dataloader) == 0:
                        break


    return MultiDataLoader

class UniformPicking(object):
    def __init__(self, window_size, n_overlap=None, n_seq=None, absolute=False, squeeze_last=False):
        self.window_size = window_size
        self.n_overlap = n_overlap or window_size/2
        self.n_seq = n_seq
        self.absolute = absolute
        self.squeeze_last = squeeze_last

    def __call__(self, series, y=None, window_size=None, n_overlap=None, n_seq=None, absolute=False, time=None):
        window_size = window_size or self.window_size
        overlap = n_overlap or self.n_overlap
        n_seq = n_seq or self.n_seq
        indices = np.array(range(0, series.shape[1]-int(window_size), int(overlap)))
        if n_seq:
            indices = indices[:n_seq]
        data = torch.stack([series[:, idx:idx+window_size] for idx in indices], dim=1)
        if self.squeeze_last:
            data = data.squeeze(-1)
        new_y = dict(y)
        if y:
            for k, t in y.items():
                if t.shape[1] == 1:
                    pass
                elif t.shape[1] == series.shape[1]:
                    new_y[k] = np.array(t)[:, indices, np.newaxis]

        return data, new_y


def SequenceLoader(sequence_length=None, random_pos=False, num_workers=None, picker=None):
    class SequenceLoader(DataLoader):
        def __init__(self, *args, **kwargs):
            self.num_workers = num_workers 
            super(SequenceLoader, self).__init__(*args, num_workers=num_workers, **kwargs)
            self.is_sequence = True
            self.sequence_length = sequence_length
            self.random_pos = random_pos
            self.picker = picker or UniformPicking(self.sequence_length)

        def __iter__(self):
            for x, y in self.dataloader:
                x, y = self.picker(x, y=y)
                yield x, y
           
    return SequenceLoader


logger=GPULogger(verbose=False)
def RawDataLoader(grain_length=None, grain_hop=None, sequence_length=0, resample=None, add_channel=False, window=False, **func_args):

    class RawDataLoader(DataLoader):
        def __repr__(self):
            return "RawDataLoader (grain_length : %d)"%self.grain_length

        def __init__(self, *args, grain_length=grain_length, grain_hop=grain_hop, sequence_length=sequence_length, resampleFactor=resample, window=False, **func_args):
            super(RawDataLoader, self).__init__(*args, **func_args)
            self.grain_length= grain_length
            self.grain_hop = grain_hop or int(grain_length / 2)
            self.resampleFactor = resampleFactor
            self.sequence_length = sequence_length or 0
            self.window = window

        def __iter__(self):

            logger('start loading')
            for x, y in self.dataloader:
                logger('loaded')

                if self.resampleFactor:
                    x = np.take(x, range(0, x.shape[-1], self.resampleFactor), axis=-1)
                if self.grain_length is not None:
                    if self.sequence_length == 0:
                        assert self.grain_length < x.shape[-1], "grain length is greater than length of chunks"
                        #TODO maybe more subtle slicing?
                        random_pos = random.randrange(x.shape[-1] - self.grain_length)
                        x = np.take(x, range(random_pos, random_pos+self.grain_length), axis=-1)
                    else:

                        full_length =  (self.sequence_length-1)*self.grain_hop  + self.grain_length
                        assert full_length < x.shape[1], \
                            "sequence length is greater than length of chunks (asked %d, is %d)"%(full_length, x.shape[1])

                        logger('start slicing')

                        max_idx = np.max(np.where((x==0).sum(0) == 0)[0])
                        random_pos = np.random.randint(0, max_idx - full_length, x.shape[0])
                        random_pos = np.stack([np.arange(r, r + full_length) for r in random_pos])
                        mask = np.zeros_like(x)
                        for i in range(x.shape[0]):
                            mask[i, random_pos[i]] = 1

                        #TODO take care of this fucking zero padding
                        # take sub-chunk
                        logger('masked selecet')
                        full_shape = x.shape
                        x = torch.masked_select(x, torch.ByteTensor(mask, device=x.device))#.reshape(x.shape[0], full_length)


                        # split in grains
                        x = torch.cat([np.take(x, range(i*self.grain_hop,(i*self.grain_hop+self.grain_length)), -1).unsqueeze(1) for i in range(self.sequence_length)], dim=1)
                        logger('splitted')

                if add_channel:
                    if self.sequence_length <= 0:
                        x = x.unsqueeze(1)
                    else:
                        x = x.unsqueeze(2)

                yield x.float(), y

    return RawDataLoader





class MixtureLoader(DataLoader):
    def __init__(self, datasets, batch_size, tasks=None, partition=None, ids=None, batch_cache_size=1, random_mode='uniform', *args, **kwargs):
        self.batch_size = batch_size
        self.tasks = None
        self.partition = None
        self.batch_cache_size = batch_cache_size
        self.loaders = []
        self.random_mode = random_mode
        for d in datasets:
            self.loaders.append(DataLoader(d, batch_size, tasks, partition, ids, batch_cache_size, *args, **kwargs))

    def get_random_weights(self, n_batches, n_weights, mode):
        if mode == 'uniform':
            weights = np.random.rand(n_batches, n_weights)
        elif mode == 'normal':
            weights = np.random.randn(n_batches, n_weights)
        elif mode == 'constant':
            weights = np.ones((n_batches, n_weights))
        elif mode == 'bernoulli':
            weights = torch.distributions.Bernoulli(torch.full((n_batches, n_weights),0.5)).sample().numpy()
        return weights

    def __iter__(self, *args, **kwargs):
        # load iterators
        iterators = [loader.__iter__() for loader in self.loaders]
        finished = False
        random_mode = kwargs.get('random_mode', self.random_mode)
        try:
            # launch the loop
            while not finished:
                x = []; y = []; self.current_ids = []
                # iterate through loaders
                for i, iterator in enumerate(iterators):
                    x_tmp, y_tmp = next(iterator)
                    x.append(x_tmp); y.append(y_tmp); self.current_ids.append(self.loaders[i].current_ids)

                min_size = min([x_tmp.shape[0] for x_tmp in x])
                x = [x_tmp[:min_size] for x_tmp in x]
                self.current_ids = [cid[:min_size] for cid in self.current_ids]
                # make a mixture of data
                self.random_weights = self.get_random_weights(x[0].shape[0], len(x), random_mode)
                final_mixture = np.zeros_like(x[0])
                for i in range(len(x)):
                    final_mixture += (np.expand_dims(self.random_weights[:, i], 1) * x[i])
                yield final_mixture, x, y
        except StopIteration:
            # stop iteration
            finished = True

class CPCLoader(DataLoader):
    is_sequence = True

    def get_cpc_examples(self, true_x, preprocessing, prob=0.5):
        # get negative ids
        #TODO with torch dataloader, we cannot retrieve the used IDs... find a solution!
        #   (there is a few chance that negative examples are positive, but...
        #   (idea : split dataset in two when loading?)
        ids = np.array(list(set(range(len(self.dataset.data)))))
        ids = ids[np.random.permutation(ids.shape[0])[:self.batch_size]]
        b = np.ones_like(ids)
        # randomly pick positive ids
        fake_data = true_x.clone()
        selectors = np.where((np.random.random(self.batch_size) >= prob) == 0)[0]
        if len(selectors) > 0:
            if preprocessing:
                fake_data[selectors] = torch.from_numpy(preprocessing(self.dataset.data[ids[selectors]]))
            else:
                fake_data[selectors] = torch.from_numpy(self.dataset.data[ids[selectors]])
            b[selectors] = 0
        return fake_data, b


"""
class SemiSupervisedDataLoader(object):
    #TODO default supervised ids
    def __init__(self, dataset, batch_size, task=None, sup_ids=None, ratio = 0.2, partition=None, *args, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        if partition is None:
            random_indices = permutation(len(dataset.data)) 
        else:
            partition_ids = dataset.partitions[partition]
            random_indices = partition_ids[permutation(len(partition_ids))]
            
        if sup_ids is None:
            n_sup = int(ratio*len(dataset.data))
            sup_ids = np.random.permutation(len(dataset.data))[:n_sup]
            
        filtered_ids = np.array([x for x in filter(lambda x: not x in sup_ids, random_indices)])
        self.random_ids = np.split(filtered_ids[:len(filtered_ids)//batch_size*batch_size], len(filtered_ids)//batch_size)
        self.sup_ids = np.split(sup_ids, len(sup_ids)//batch_size)
        self.task = task
            
    def __iter__(self):
        for i in range(len(self.sup_ids)):
            yield self.dataset.data[self.sup_ids[i]], self.dataset.metadata[self.task][self.sup_ids[i]]
        for i in range(len(self.random_ids)):
            yield self.dataset.data[self.random_ids[i]], None
"""
