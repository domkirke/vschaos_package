"""

 Import toolbox       : Import utilities

 This file contains lots of useful utilities for dataset import.

 Author               : Philippe Esling
                        <esling@ircam.fr>

"""

import torch
import os

def checklist(item, n=1, rev=False, fill=False, copy=False):
    if not issubclass(type(item), list):
        if copy:
            item = [copy.deepcopy(item) for i in range(n)]
        else:
            item = [item]*n
    if rev:
        item = list(reversed(item))
    if fill and len(item) < n:
        item = item + [None]*(n - len(item))
    return item

def checktuple(item, n=1, rev=False):
    if not issubclass(type(item), tuple):
        item = (item,)*n
    if rev:
        item = list(reversed(item))
    return item

def dyn_expand(tensor, target_size):
    for i, t in enumerate(target_size):
        if tensor.size(i) > t:
            if i > 0:
                index = tuple([slice(None, None, None)]*(i) + [slice(0, t)])
            else:
                index = slice(0, t)
            tensor = tensor.__getitem__(index)
        elif tensor.size(i) < t:
            zeros_shape = list(target_size[:(i+1)]) + [int(t) for t in tensor.shape[(i+1):]]
            zeros_shape[i] = int(target_size[i] - tensor.size(i))
            zeros = torch.zeros(*tuple(zeros_shape))
            tensor = torch.cat([tensor, zeros], dim=i)
    return tensor

def dyn_collate(tensor_list, mode="max"):
    if isinstance(tensor_list[0], list):
        out = [None]*len(tensor_list[0])
        for i in range(len(out)):
            out[i] = dyn_collate([t[i] for t in tensor_list])
        return out
    sizes = torch.Tensor([t.shape for t in tensor_list]).int()
    if mode == "min":
        target_size = [torch.min(sizes[:, i]) for i in range(sizes.size(1))]
    elif mode == "max":
        target_size = [torch.max(sizes[:, i]) for i in range(sizes.size(1))]
    else:
        raise ValueError("mode %s not recognized"%mode)
    target_size = [int(t) for t in target_size]
    for i, t in enumerate(tensor_list):
        tensor_list[i] = dyn_expand(t, target_size)
    return torch.stack(tensor_list, dim=0)

def window_data(chunk, window=None, window_overlap=None, dim=0):
    n_windows = (chunk.shape[dim] - window) // window_overlap
    chunk_list = []
    if n_windows >= 0:
        for i in range(n_windows + 1):
            chunk_list.append(torch.index_select(chunk, dim, torch.arange(i * window_overlap, i * window_overlap + window)).unsqueeze(dim))
    else:
        chunk_list = [dyn_expand(chunk), (*chunk.shape[:dim], 1, window)]
    if len(chunk_list) > 2:
        return torch.cat(chunk_list, dim)
    else:
        return chunk_list[0]


def overlap_add(chunk, window, window_overlap, dim=0, mode="overlap_add", window_fn = None):
    target_size = chunk.shape[:dim] + (chunk.size(dim)*window_overlap + window,)
    tensor = torch.zeros(target_size, device=chunk.device, requires_grad=chunk.requires_grad)
    for j in range(chunk.size(dim)):
        chunk_to_add = chunk[..., j, :]
        if window_fn is not None:
            chunk_to_add = window_fn(window)*chunk_to_add
        if mode=="overlap_add":
            tensor[..., slice(j*window_overlap, j*window_overlap+window)] += chunk_to_add
        else:
            tensor[..., slice(j * window_overlap, (j + 1) * window_overlap)] += chunk_to_add
    return tensor





def retrieve_tasks_from_path(metadata_path):
    files = list(filter(lambda x: os.path.isdir(metadata_path + '/' + x), os.listdir(metadata_path)))
    for i in reversed(range(len(files))):
        if not os.path.isfile(metadata_path + '/' + files[i] + '/metadata.txt'):
            del files[i]
    return files

def mkdir(path):
    """ Create a directory """
    assert (os.popen('mkdir -p %s' % path), 'could not create directory')
