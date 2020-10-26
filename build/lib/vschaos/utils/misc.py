# -*- coding: utf-8 -*-

import torch, numpy as np, os, pdb, copy as copy_module, random
from time import time
from .onehot import oneHot
from .. import distributions as dist
import functools



class NaNError(Exception):
    pass

class Logger(object):
    def __init__(self, log_file=None, verbose=True, synchronize=False):
       self.reset()
       self.log_file = log_file
       if self.log_file:
           os.system('echo "" > %s &'%self.log_file)
       self.verbose = verbose

    def __call__(self, *args, **kwargs):
        if self.verbose:
            print(*args)
        if self.log_file:
            os.system("echo '%s' >> %s &"%(args , self.log_file))
        self.update(*args, **kwargs)

    def reset(self):
        pass

    def update(self, *args, **kwargs):
        pass


class GPULogger(object):
    def __init__(self, log_file=None, verbose=True, synchronize=False):
       self.reset()
       self.log_file = log_file
       if self.log_file:
           os.system('echo "" > %s &'%self.log_file)
       self.verbose = verbose
       self.synchronize = synchronize

    def update(self, msg):
        if not torch.cuda.is_available():
            return
        msg = "%s : %s, %s"%(msg , str(torch.cuda.memory_allocated())," "+str(torch.cuda.memory_cached()))
        """
        if torch.cuda.is_available() and self.synchronize:
            torch.cuda.synchronize()
        """
        time_elapsed =  time() - self.time
        if self.verbose:
            print(msg, time_elapsed)

        self.reset()
        if self.log_file:
            os.system("echo '%s %s' >> %s &"%(time_elapsed, msg , self.log_file))

    def __call__(self, msg):
        self.update(msg)

    def reset(self):
        self.time = time()

def print_stats(v, k=""):
    if v is not None:
        print('%s : \t mean : %s \t std : %s \n --- \t min : %s \t max : %s'%(k, v.mean(), v.std(), v.min(), v.max()))
    else:
        print('%s : None'%k)

def print_module_stats(module):
    for k, v in module.named_parameters():
        print_stats(v, k)

def print_module_grad(module):
    for k, v in module.named_parameters():
        print_stats(v.grad, k)



def lget(list, i):
    if list is None:
        return None
    if not hasattr(list, '__len__'):
        return None
    if i >= len(list):
        return None
    return list[i]


def recgetattr(obj, attr):
    if hasattr(obj, attr):
        return getattr(obj, attr)
    elif hasattr(obj, "__iter__"):
        rec_attr = [recgetattr(o, attr) for o in obj]
        for i,a in enumerate(rec_attr):
            if a is None:
                del rec_attr[a]
        if len(rec_attr) == 0:
            return None
        else:
            return rec_attr[0]
    else:
        return None

def recgetitem(obj, idx, dim=0):
    if issubclass(type(obj), list):
        return [recgetitem(o, idx, dim=dim) for o in obj]
    elif issubclass(type(obj), tuple):
        return tuple([recgetitem(o, idx, dim=dim) for o in obj])
    elif issubclass(type(obj), dict):
        return {k: recgetitem(v, idx, dim=dim) for k,v in obj.items()}
    elif issubclass(type(obj), (torch.Tensor)):
        return torch.index_select(obj, dim, torch.LongTensor(idx))
    elif issubclass(type(obj), (dist.Distribution)):
        return obj.index_select(dim, torch.LongTensor(idx))
    elif issubclass(type(obj), (np.ndarray)):
        return np.take(obj, np.array(idx), axis=dim)
    else:
        return

def decudify(obj, scalar=False):
    if issubclass(type(obj), dict):
        return {k:decudify(v, scalar=scalar) for k,v in obj.items()}
    elif issubclass(type(obj), list):
        return [decudify(i, scalar=scalar) for i in obj]
    elif issubclass(type(obj), tuple):
        return tuple([decudify(i, scalar=scalar) for i in list(obj)])
    elif issubclass(type(obj), dist.Distribution):
        return obj.to('cpu')
    elif torch.is_tensor(obj):
        obj = obj.cpu()
        if scalar:
            obj = obj.detach().numpy()
        return obj

    else:
        return obj

def concat_distrib(distrib_list, unsqueeze=False, dim=1):

    def concat_normal(distrib_list):
        means = [d.mean for d in distrib_list]
        stds = [d.stddev for d in distrib_list]
        if unsqueeze:
            means = [m.unsqueeze(dim) for m in means]
            stds = [s.unsqueeze(dim) for s in stds]
        means = torch.cat(means, dim)
        stds = torch.cat(stds, dim)
        return type(distrib_list[0])(means, stds)
    
    def concat_probs(distrib_list):
        probs = [d.probs for d in distrib_list]
        if unsqueeze:
            probs = [p.unsqueeze(dim) for p in probs]
        probs = torch.cat(probs, dim)
        return type(distrib_list[0])(probs=probs)

    def concat_flow(distrib_list):
        flow_ids = len(set([id(d.flow) for d in distrib_list])) == 1, "flow concatenantion requires identical flow modules"
        #TODO multi flow concatenation?
        return dist.FlowDistribution(concat_distrib([d.base_dist for d in distrib_list],unsqueeze=unsqueeze, dim=dim),
                distrib_list[0].flow, unwrap_blocks=distrib_list[0].unwrap_blocks, in_select=distrib_list[0].in_selector,
                div_mode = distrib_list[0].div_mode)

    assert(len(set([type(d) for d in distrib_list])) == 1)
    if type(distrib_list[0]) in (dist.Normal, dist.RandomWalk):
        return concat_normal(distrib_list)
    elif type(distrib_list[0]) in (dist.Bernoulli, dist.Categorical, dist.Multinomial):
        return concat_probs(distrib_list)
    elif type(distrib_list[0]) in (dist.FlowDistribution,):
        return concat_flow(distrib_list)
    else:
        raise Exception('cannot concatenate distribution of type %s'%type(distrib_list[0]))


def crossed_select(mask, mat1, mat2):
    assert mat1.shape == mat2.shape
    if len(mat1.shape) > 1:
        for a in mat1.shape[1:]:
            mask = mask.unsqueeze(-1)
        print(mat1.shape, mat2.shape, mask.shape)
        mask.repeat(1, *mat1.shape[1:])
    return torch.where(mask == 1 , mat2, mat1)


def dist_crossed_select(mask, dist1, dist2=None):
    if dist2 is None:
        raise NotImplementedError
    if type(dist1) == torch.distributions.Normal or type(dist1) == dist.Normal:
        return type(dist1)(crossed_select(mask, dist1.mean, dist2.mean), crossed_select(mask, dist1.stddev, dist2.stddev))


def merge_dicts(obj, dim=0, unsqueeze=None):
    def np_cat(obj):
        if unsqueeze is not None:
            return np.concatenate([np.expand_dims(o, axis=dim) for o in obj], axis=dim)
        else:
            return np.concatenate(obj, axis=dim)
    def torch_cat(obj):
        if unsqueeze is not None:
            return torch.cat([o.unsqueeze(unsqueeze) for o in obj], dim=dim)
        else:
            return torch.cat(obj, dim=dim)
    if not (issubclass(type(obj),list) or issubclass(type(obj),tuple)) or len(obj) == 0 :
        return
    list_type = type(obj)
    if issubclass(type(obj[0]), dict):
        return {k:merge_dicts([ d[k] for d in obj ], dim=dim, unsqueeze=unsqueeze) for k, v in obj[0].items()}
    elif issubclass(type(obj[0]), torch.Tensor):
        return torch_cat(obj)
    elif issubclass(type(obj[0]), np.ndarray):
        return list_type(np_cat(obj))
    elif issubclass(type(obj[0]), list) or issubclass(type(obj[0]), tuple) :
        return list_type([ merge_dicts( [l[i] for l in obj], dim=dim, unsqueeze=unsqueeze) for i in range(len(obj[0]))])
    elif issubclass(type(obj[0]), dist.Distribution):
        return concat_distrib(obj, unsqueeze=unsqueeze, dim=dim)
    else:
        return obj
    

def checklist(item, n=1, rev=False, fill=False, copy=False):
    if not issubclass(type(item), list):
        if copy:
            item = [copy_module.deepcopy(item) for i in range(n)]
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


def parse_additional_args(kwargs, criterion=None, parser=None):
    kwargs_parsed = dict()
    if criterion is None:
        criterion = lambda x: True
    if parser is None:
        parser = lambda x: x
    keys_idx = list(filter(lambda x: kwargs[x][:2] == "--", list(range(len(kwargs))))) # gather
    for idx in range(len(keys_idx)):
        if idx < len(keys_idx)-1:
            kwargs_parsed[kwargs[keys_idx[idx]][2:]] = kwargs[keys_idx[idx]+1:keys_idx[idx+1]]
        else:
            kwargs_parsed[kwargs[keys_idx[idx]][2:]] = kwargs[keys_idx[idx]+1:]

    for key, values in kwargs_parsed.items():
        for i, v in enumerate(values):
            if v == '-':
                kwargs_parsed[key][i] = None
            else:
                kwargs_parsed[key][i] = parser(v)
        if not criterion(values):
            del kwargs_parsed[key]

    return kwargs_parsed

def rec_attr(obj, attr, *args, **kwargs):
    assert type(attr) == str
    if issubclass(type(obj), list):
        return [rec_attr(o, attr) for o in obj]
    if issubclass(type(obj), tuple):
        return tuple([rec_attr(o, attr) for o in obj])
    else:
        return getattr(obj, attr)

def apply(obj, fun, *args, rec=False, **kwargs):
    if issubclass(type(obj), list):
        if rec:
            return [apply(o, *args, **kwargs) for o in obj]
        else:
            return [fun(o, *args, **kwargs) for o in obj]
    elif issubclass(type(obj), tuple):
        if rec:
            return tuple([apply(o, *args, **kwargs) for o in obj])
        else:
            return tuple([fun(o, *args, **kwargs) for o in obj])
    else:
        return fun(obj)

def apply_method(obj, fun, *args, **kwargs):
    assert type(fun) == str
    if issubclass(type(obj), list):
        return [getattr(o, fun)(*args, **kwargs) for o in obj]
    if issubclass(type(obj), tuple):
        return tuple([getattr(o, fun)(*args, **kwargs) for o in obj])
    else:
        # pdb.set_trace()
        return getattr(obj, fun)(*args, **kwargs)
        #return getattr(obj, fun)(obj, *args, **kwargs)

def get_latent_out(out, layer=None):
    out = dict(out)
    layer = int(layer)
    for k in list(out.keys()):
        if not issubclass(type(out[k]), list):
            continue
        if layer >= len(out[k]):
            del out[k]
        else:
            out[k] = out[k][layer]
    return out

def parse_folder(folder, exts=None, hidden=False):
    valid_files = []
    for root,directory,files in os.walk(folder):
        if exts is not None:
            files = list(filter(lambda x: os.path.splitext(x)[1] in exts, files))
        if not hidden:
            files = list(filter(lambda x: x[0] != '.', files))
        valid_files.extend([root+'/'+f for f in files])
    return valid_files

def split_select(tensor, idxs, dim=-1):
    slices = [0, *tuple(np.cumsum(idxs).tolist())]
    slices_idxs = [torch.arange(slices[i], slices[i+1]).long().to(tensor.device) for i in range(len(idxs))]
    out = [tensor.index_select(dim, idxs) for idxs in slices_idxs]
    return out

# def get_latent_out(out, layer=None):
#     latent = {}
#     if out.get('z_enc'):
#         latent['z_enc'] = out['z_enc']
#     if out.get('z_params_enc'):
#         latent['z_params_enc'] = out['z_params_enc']
#     if out.get('z_dec'):
#         latent['z_dec'] = out['z_dec']
#     if out.get('z_params_dec'):
#         latent['z_params_dec'] = out['z_params_dec']
#
#     if layer is not None:
#         layer = int(layer)
#         for k in list(latent.keys()):
#             if not issubclass(type(latent[k]), list):
#                 continue
#             if layer >= len(latent[k]):
#                 del latent[k]
#             else:
#                 latent[k] = latent[k][layer]
#     return latent

def denest_dict(nest_dict):
    keys = set()
    new_dict = {}
    for item in nest_dict:
        keys = keys.union(set(item.keys()))
    for k in keys:
        new_dict[k] = [x.get(k) for x in nest_dict]
    return new_dict

def concat_tensors(tensor_list):
    return torch.cat([d.unsqueeze(1) for d in tensor_list], dim=1)

def accumulate_losses(losses, weight=1.):
    if issubclass(type(losses[0]), (np.ndarray, torch.Tensor, int, float)):
        return sum(losses)*weight
    elif issubclass(type(losses[0]), (list, tuple)):
        compiled = []
        for i in range(len(losses[0])):
            compiled.append(accumulate_losses(tuple([losses[j][i] for j in range(len(losses))])))
        return tuple(compiled)

def normalize_losses(losses, N):
    if issubclass(type(losses), (np.ndarray, torch.Tensor, int, float)):
        return losses / N
    elif issubclass(type(losses), (list, tuple)):
        compiled = []
        compiled.append([normalize_losses(losses[j], N) for j in range(len(losses))])
        return compiled


def pack_output(output, dim=1):
    def dict_walk(pattern, out_list):
        if torch.is_tensor(pattern):
            return torch.cat([i.unsqueeze(dim) for i in out_list], dim)
        elif issubclass(type(pattern), dist.Distribution):
            return concat_distrib(out_list, dim)
        elif issubclass(type(pattern), (tuple, list)):
            return type(pattern)([dict_walk(pattern[i], [o[i] for o in out_list]) for i in range(len(pattern))])
        elif issubclass(type(pattern), dict):
            return {k:dict_walk(pattern[k], [o[k] for o in out_list]) for k in pattern.keys()}
    ref_struct = output[0]
    packed_output = dict_walk(ref_struct, output)
    return packed_output

def apply_distribution(obj, func, *args, **kwargs):
    if issubclass(type(obj), torch.distributions.Bernoulli):
        return torch.distributions.Bernoulli(func(obj.mean, *args, **kwargs))
    if issubclass(type(obj), torch.distributions.Normal):
        return torch.distributions.Normal(func(obj.mean, *args, **kwargs),
                                          func(obj.stddev, *args, **kwargs))
    if issubclass(type(obj), torch.distributions.Categorical):
        return torch.distributions.Normal(func(logits=obj.logits, *args, **kwargs))


def recursive_view(obj, *args):
    if issubclass(type(obj), dict):
        return {k:recursive_view(v, *args) for k,v in obj.items()}
    elif issubclass(type(obj), list):
        return [recursive_view(i, *args) for i in obj]
    elif issubclass(type(obj), tuple):
        return tuple([recursive_view(i, *args) for i in list(obj)])
    elif torch.is_tensor(obj):
        obj_dim = tuple(obj.shape[1:])
        return obj.view(*args, *obj_dim)
    elif issubclass(type(obj), torch.distributions.Distribution):
        #TODO to change when I will finally set MultivariateNormal
        batch_shape = tuple(obj._batch_shape[1:])
        return apply_distribution(obj, torch.Tensor.view, *args, *batch_shape)
    else:
        return obj

def flatten_seq(func):
    @functools.wraps(func)
    def unwrap(x, *args, **kwargs):
        n_batch = x.shape[0]; n_seq = x.shape[1]; n_dims = tuple(x.shape[2:])
        x = x.view(n_batch * n_seq, *n_dims)
        return recursive_view(func(x, *args, **kwargs), n_batch, n_seq)
    return unwrap


def flatten_seq_method(func):
    @functools.wraps(func)
    def unwrap(self, x, *args, **kwargs):
        if hasattr(self, "input_ndim"):
            input_nshape = self.input_ndim
        else:
            input_nshape = 1

        if hasattr(self, "in_channels"):
            input_nshape += self.in_channels
        #pdb.set_trace()


        if isinstance(x, list):
            n_batch = set([x[i].size(0) for i in range(len(x))])
            assert len(n_batch) == 1, "batch sizes must be the same (got %s)"%n_batch
            n_batch = n_batch.pop()
            n_seq = 0
            for i, x_tmp in enumerate(x):
                if len(x_tmp.shape) <= 1 + input_nshape:
                    continue
                n_seq_tmp = x_tmp.shape[1]; n_dims = tuple(x_tmp.shape[2:])
                x = [x_tmp.reshape(n_batch * n_seq_tmp, *n_dims) for x_tmp in x]
                n_seq = max(n_seq, n_seq_tmp)
            if len(x_tmp.shape) <= 1 + input_nshape:
                original_shape = (n_batch,)
            else:
                original_shape = (n_batch, n_seq)
        else:
            n_batch = x.shape[0]; n_seq = x.shape[1]; n_dims = tuple(x.shape[2:])
            if len(x.shape) > 1 + input_nshape:
                x = x.reshape(n_batch * n_seq, *n_dims)
                original_shape = (n_batch, n_seq)
            else:
                original_shape = (n_batch,)
        if kwargs.get('y') is not None:
            y = dict(kwargs['y'])
            if len(original_shape) > 1:
                for k, v in y.items():
                #if y[k].ndim > 1:
                    y[k] = v.reshape(n_batch * n_seq, *tuple(v.shape[2:]))
            kwargs['y'] = y
        return recursive_view(func(self, x, *args, **kwargs), *original_shape)
        # if isinstance(x, list):
        #     out = [recursive_view(func(self, x, *args, **kwargs), *original_shape[i]) for i in range(len(x))]
        # else:
        #     return recursive_view(func(self, x, *args, **kwargs), *original_shape)

    return unwrap


def sample_normalize(x):
    shape = x.shape
    x_mean = x.detach().cpu().numpy().mean(axis=tuple(range(1, len(shape))))
    x_std = x.detach().cpu().numpy().std(axis=tuple(range(1, len(shape))))
    x = x - torch.tensor(x_mean, requires_grad=False, device=x.device).view(x.shape[0], *tuple([1]*(len(shape)-1)))
    x = x / torch.tensor(x_std, requires_grad=False, device=x.device).view(x.shape[0], *tuple([1]*(len(shape)-1)))
    return x




def reshape_distribution(distrib, shape):
    if issubclass(type(distrib), dist.Bernoulli):
        return dist.Bernoulli(distrib.mean.reshape(shape))
    if type(distrib) == dist.Normal:
        return dist.Normal(distrib.mean.reshape(shape), distrib.stddev.reshape(shape))
    if type(distrib) == dist.WeinerProcess:
        return dist.WeinerProcess(distrib.mean.reshape(shape), distrib.stddev.reshape(shape))
    else:
        raise NotImplementedError

def view_distribution(distrib, shape):
    if issubclass(type(distrib), dist.Bernoulli):
        return dist.Bernoulli(distrib.mean.contiguous().view(shape))
    if type(distrib) == dist.Normal:
        return dist.Normal(distrib.mean.contiguous().view(shape), distrib.stddev.contiguous().view(shape))
    if type(distrib) == dist.WeinerProcess:
        return dist.WeinerProcess(distrib.mean.view(shape), distrib.stddev.view(shape))
    else:
        raise NotImplementedError



def cat_distribution(distribs, dim):
    assert len(set([type(d) for d in distribs])) == 1, "distributions must be of same type to be concatenated"
    if issubclass(type(distribs[0]), dist.Bernoulli):
        mean = torch.cat([d.mean for d in distribs], dim=dim)
        return dist.Bernoulli(mean)
    if issubclass(type(distribs[0]), dist.Normal):
        mean = torch.cat([d.mean for d in distribs], dim=dim)
        stddev = torch.cat([d.stddev for d in distribs], dim=dim)
        return dist.Normal(mean, stddev)
    else:
        raise NotImplementedError


class CollapsedIds(object):
    def __init__(self):
        self.ids = dict()
        self.full_set = set()

    def add(self, task, ids):
        self.ids[task] = ids
        self.full_set |= set(ids)

    def get_full_ids(self):
        return list(self.full_set)

    def get_ids(self, task):
        current_ids = self.ids[task]
        full_set = list(self.full_set)
        idxs = [full_set.index(i) for i in current_ids]
        return idxs

    def transform(self, ids):
        full_set = list(self.full_set)
        idxs = [full_set.index(i) for i in ids]
        return idxs


def check_dir(fold):
    if not os.path.isdir(fold):
        os.makedirs(fold)



    param_buffer = dict()

def get_flatten_meshgrid(n_dim, scales, n_grid):
    if not issubclass(type(scales[0]), list):
        scales = [scales for _ in range(n_dim)]
    ranges = tuple([np.linspace(scales[n][0], scales[n][1], n_grid) for n in range(n_dim)])
    meshes = np.meshgrid(*ranges)

    iterator = np.nditer(meshes[0], ['multi_index'])
    full_z = np.zeros((meshes[0].size, n_dim))
    current_id = 0; idxs = []

    for _ in iterator:
        z = np.array([meshes[a][iterator.multi_index] for a in range(n_dim)])
        full_z[current_id] = z
        idxs.append(iterator.multi_index)
        current_id += 1
    return full_z, meshes, idxs


def choices(l, k=1):
    if hasattr(random, 'choices'):
        return random.choices(l, k=k)
    else:
        if issubclass(type(l), list):
            return [l[i] for i in np.random.permutation(len(l))[:k]]
        elif issubclass(type(l), np.ndarray):
            return l[np.random.permutation(len(l))[:k]]
        else:
            raise TypeError('list of type %s is not recognized for function choices'%type(l))


# CONDITIONING UTILITARY METHODSZ

def get_conditioning_params(model):
    if model is None:
        return "None"
    hidden_params = []
    for ph in checklist(model.phidden):
        if ph.get('encoder') or ph.get('decoder'):
            if ph.get('encoder'):
                hidden_params.append(ph['encoder'])
            if ph.get('decoder'):
                hidden_params.append(ph['decoder'])
        else:
            hidden_params.extend(ph)

    labels = {}
    for ph in hidden_params:
        ph = checklist(ph);
        for phh in ph:
            if phh.get('label_params') is not None:
                labels = {**labels, **phh['label_params']}
    return labels


def generate_conditioning_mesh(label_params):
    ys = {}
    for k, v in label_params:
        if issubclass(v['dist'], dist.Categorical):
            ys[k] = oneHot(range(v['dim']), v['dim'])

    return ys


def parse_filtered_classes(dataset, string_args):
    if string_args[0][0] != "#":
        raise SyntaxError('parse_filtered_classes : string must begin with a task')
    tasks_parsed = {}
    current_task = None
    for str in string_args:
        if str[0] == "#":
            assert str[1:] in dataset.tasks, "task %s not found in dataset %s"%(str, dataset)
            current_task = str[1:]
            tasks_parsed[current_task] = []
        else:
            # tasks_parsed[current_task].extend(dataset.get_ids_from_class(dataset.classes[current_task][str], current_task))
            tasks_parsed[current_task].append(str)
    # ids = np.unique(np.array(sum(tasks_parsed.values(), [])))

    return tasks_parsed







