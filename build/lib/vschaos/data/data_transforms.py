import abc, random, pdb
from numbers import Number
import torch
from ..utils.misc import checklist
from .data_utils import window_data, overlap_add
import torch_dct as dct
import numpy as np
from numpy import pi
torch_pi = torch.tensor(pi, requires_grad=False)
eps = 1e-3

class NotInvertibleError(Exception):
    def __init__(self, transform):
        super(NotInvertibleError, self).__init__()
        self.transform = transform

    def __repr__(self):
        return "NotInvertibleError(%s)"%self.transform

class NoTimingError(Exception):
    pass


class Transform(object):
    def __init__(self):
        pass

    def forward(self, x, *args, **kwargs):
        return x

    def invert(self, x, *args, **kwargs):
        return x

    def scale(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_timing(self, x, time=None, **kwargs):
        return time


class ComposeTransform(Transform):
    def __init__(self, transforms=[]):
        self.transforms = checklist(transforms)

    def __getitem__(self, item):
        return self.transforms[item]

    def __len__(self):
        return len(self.transforms)

    def __eq__(self, other):
        if isinstance(other, ComposeTransform):
            other = other.transforms
        if len(other) != len(self.transforms):
            return False
        return not False in [other[i] == self.transforms[i] for i in range(len(self))]

    def append(self, item):
        self.transforms.append(item)

    def extend(self, item):
        if issubclass(type(item), ComposeTransform):
            self.transforms.extend(item.transforms)
        else:
            self.transforms.extend(item)

    def forward(self, x, *args, **kwargs):
        for t in self.transforms:
            x = t(x)
        return x

    def invert(self, x):
        for t in reversed(self.transforms):
            x = t.invert(x)
        return x

    def scale(self, x):
        for t in self.transforms:
            if hasattr(t, "scale"):
                t.scale(x)
            x = t(x)

    def __add__(self, other):
        if not isinstance(other, (ComposeTransform, list)):
            raise TypeError('__add__ for ComposeTransorm is only with other ComposeTransform')
        if isinstance(other, list):
            return ComposeTransform(transforms=self.transforms + other)
        else:
            return ComposeTransform(transforms = self.transforms + other.transforms)

    def __radd__(self, other):
        if not isinstance(other, (ComposeTransform, list)):
            raise TypeError('__add__ for ComposeTransorm is only with other ComposeTransform')
        if isinstance(other, list):
            return ComposeTransform(transforms=other+self.transforms)
        else:
            return ComposeTransform(transforms = other.transforms+self.transforms)




## Generic transforms

class Repeat(Transform):
    def __init__(self, repeat, dim=1, unsqueeze=True):
        self.repeat = repeat
        self.dim = dim
        self.unsqueeze = unsqueeze

    def forward(self, x, *args, **kwargs):
        if isinstance(x, list):
            return [self(x_i) for x_i in x]
        if self.unsqueeze:
            x = x.unsqueeze(self.dim)
        sizes = [1]*len(x.size())
        sizes[self.dim] = self.repeat
        return x.repeat(*sizes)

    def invert(self, x, *args, **kwargs):
        if isinstance(x, list):
            return [self.invert(x_i) for x_i in x]
        current_size = x.size(self.dim)
        x = torch.index_select(x, self.dim, torch.range(0, current_size//self.repeat))
        if self.unsqueeze:
            x = x.squeeze(self.dim)
        return x



class Squeeze(Transform):
    def __init__(self, dim=None):
        self.dim = dim

    def forward(self, x, *args, **kwargs):
        if isinstance(x, list):
            return [self(x_i) for x_i in x]
        return x.squeeze(dim=self.dim)

    def invert(self, x, *args, **kwargs):
        if isinstance(x, list):
            return [self.invert(x_i, *args, **kwargs) for x_i in x]
        if self.dim is not None:
            return x.unsqueeze(dim=self.dim)
        else:
            return x

    def get_timing(self, x, time=None, **kwargs):
        return self(time)


class Permute(Transform):
    def __init__(self, dims=None):
        self.dims = dims

    def forward(self, x, *args, **kwargs):
        if isinstance(x, list):
            return [self(x_i) for x_i in x]
        return x.permute(self.dims)

    def invert(self, x, *args, **kwargs):
        if isinstance(x, list):
            return [self.invert(x_i, *args, **kwargs) for x_i in x]
        return x.permute(self.dims)

    def get_timing(self, x, time=None, **kwargs):
        return self(time)


class Unsqueeze(Transform):
    def __init__(self, dim=None):
        self.dim = dim

    def forward(self, x, *args, **kwargs):
        if isinstance(x, list):
            return [self(x_i) for x_i in x]
        return x.unsqueeze(dim=self.dim)

    def invert(self, x, *args, **kwargs):
        if isinstance(x, list):
            return [self.invert(x_i, *args, **kwargs) for x_i in x]
        if self.dim is not None:
            return x.squeeze(dim=self.dim)
        else:
            return x

    def get_timing(self, x, time=None, **kwargs):
         return self(time)


class Flatten(Transform):

    def __init__(self, dim):
        assert dim < 0
        self.dim =dim
        self.original_shape = None

    def scale(self, data):
        if torch.is_tensor(data):
            self.original_shape = data.shape[self.dim:]
        elif isinstance(data, list):
            shapes = [t.shape for t in data]
            if len(set(shapes)) > 1:
                print('{Warning} Flatten transform scaled on heterogeneous data. inversion not possible')
            else:
                shape = data[0].shape[self.dim:]

    def forward(self, x, *args, **kwargs):
        if isinstance(x, list):
            return [self(x_i) for x_i in x]
        size = x.shape[:self.dim]
        return x.view(*size, -1)

    def invert(self, x, *args, **kwargs):
        if self.original_shape is None:
            return x
        size = x.shape[:-1]
        return x.view(*size, *self.original_shape)


class Sequence(Transform):

    def __init__(self, seq, dim=1, random_start=False):
        self.seq = seq
        self.dim = dim
        self.random_start = random_start

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if isinstance(x, list):
            return [self(x_i) for x_i in x]
        if x.size(self.dim) < self.seq:
            size = list(x.size()); size[self.dim] = self.seq - x.shape[self.dim]
            x = torch.cat([x, torch.zeros(size, dtype=x.dtype, device=x.device)], axis=self.dim)
        else:
            sizes = [slice(None)]*x.ndimension()
            start = 0
            if self.random_start:
                start = random.randrange(0, x.size(self.dim) - self.seq)
            sizes[self.dim] = slice(start, start+self.seq)
            x = x[sizes]
        return x

    def invert(self, x, *args, **kwargs):
        return x

    def get_timing(self, x, time=None, **kwargs):
        return self(time)


class Normalize(Transform):
    def __init__(self, mode="minmax", scale="bipolar"):
        super(Normalize, self).__init__()
        self.mode = mode or "minmax"
        self.polarity = scale or "bipolar"

    @staticmethod
    def get_stats(data, mode="gaussian", polarity="bipolar"):
        if mode == "minmax":
            if polarity == "bipolar":
                mean = (torch.max(data) - torch.sign(torch.min(data))*torch.min(data)) / 2
                max = torch.max(torch.abs(data - mean))
            elif polarity == "unipolar":
                mean = torch.min(data); max = torch.max(torch.abs(data))
        elif mode == "gaussian":
            if polarity=='bipolar':
                mean = torch.mean(data); max = torch.std(data)
            if polarity=='unipolar':
                mean = torch.min(data); max = torch.std(data)
        return mean, max

    def scale(self, data):
        if issubclass(type(data), list):
            stats = torch.Tensor([self.get_stats(d, self.mode, self.polarity) for d in data])
            if self.mode == "minmax":
                self.mean = torch.mean(stats[:,0]); self.max = torch.max(stats[:,1])
            else:
                self.mean = torch.min(stats[:, 0]);
                # recompose overall variance from element-wise ones
                n_elt = torch.Tensor([torch.prod(torch.Tensor(list(x.size()))) for x in data])
                std_unscaled = ((stats[:,1]**2) / (stats.shape[0]))
                self.max = torch.sqrt(torch.sum(std_unscaled))
        else:
            self.mean, self.max = self.get_stats(data, self.mode, self.polarity)

    def forward(self, x, batch_first=True):
        if issubclass(type(x), list):
            return [self(x[i]) for i in range(len(x))]
        out = torch.true_divide(x - self.mean, self.max)

        if self.polarity == "unipolar":
            out = torch.clamp(out, eps, None)
        return out

    def invert(self, x, batch_first=True):
        return x * self.max + self.mean

class Binary(Normalize):

    def __init__(self, dataset=None):
        super(Binary, self).__init__(mode="minmax", scale="unipolar")

    def scale(self, data):
        super(Binary, self).scale(data)

    def forward(self, *args, **kwargs):
        normalized_data = super(Binary, self).forward(*args, **kwargs)
        return (normalized_data >= 0.5).int()

    def invert(self, x, batch_first=True):
        return x



def oneHot(labels, dim, is_sequence=False):
    if isinstance(labels, Number):
        t = np.zeros((1, dim))
        t[0, int(labels)] = 1
    else:
        if len(labels.shape) <= 2:
            if issubclass(type(labels), np.ndarray):
                n = labels.shape[0]
                t = np.zeros((n, dim))
                for i in range(n):
                    if labels[i] == -1:
                        continue
                    t[i, int(labels[i])] = 1
            elif torch.is_tensor(labels):
                n = labels.size(0)
                t = torch.zeros((n, dim), device=labels.device)
                for i in range(n):
                    if labels[i] == -1:
                        continue
                    t[i, int(labels[i])] = 1
                t.requires_grad_(labels.requires_grad)
            else:
                raise Exception('type %s is not recognized by oneHot function'%type(labels))
        elif len(labels.shape) == 3:
            orig_shape = labels.shape[:2]
            labels = labels.reshape(labels.shape[0]*labels.shape[1], *labels.shape[2:])
            t = np.concatenate([oneHot(labels[i], dim)[np.newaxis] for i in range(labels.shape[0])], axis=0)
            t = t.reshape((orig_shape[0], orig_shape[1], *t.shape[1:]))
    return t


def fromOneHot(vector, is_sequence=False):
    axis = 2 if is_sequence else 1
    if issubclass(type(vector), np.ndarray):
        ids = np.argmax(vector, axis=axis)
        return ids
    elif issubclass(type(vector), torch.Tensor):
        return torch.argmax(vector, dim=axis)
    else:
        raise TypeError('vector must be whether a np.ndarray or a tensor.')


class OneHot(Transform):
    def __repr__(self):
        return "OneHot(classes:%s, is_sequence: %s, make_channels:%s)"%(self.classes, self.is_sequence, self.make_channels)

    def __init__(self, classes=None, is_sequence=False, make_channels=False, flatten=False):
        self.classes = classes
        self.hashes = None
        if classes is not None:
            self.hashes = {k:k for k in range(classes)}
        self.is_sequence = is_sequence
        self.flatten = flatten
        self.make_channels = make_channels

    def get_hash(self, data):
        min_label = torch.min(data.int())
        max_label = torch.max(data.int())
        hash = {}; curHash = 0
        for i in range(min_label, max_label+1):
            is_present = torch.sum(data == i) > 0
            if is_present:
                hash[int(curHash)] = i
                curHash += 1
        return curHash, hash

    def scale(self, data):
        if issubclass(type(data), (list, tuple)):
            self.classes = []; self.hashes = []
            for d in data:
                c, h = self.get_hash(d)
                self.classes.append(c)
                self.hashes.append(h)
        else:
            c, h = self.get_hash(data)
            self.classes = c
            self.hashes = h

    def __call__(self, x, classes=None, hashes=None, *args, **kwargs):
        classes = classes or self.classes
        hashes = hashes or self.hashes
        if issubclass(type(x), (list, tuple)):
            t = []
            for i in range(len(x)):
                t.append(self(x[i], classes=classes[i], hashes=hashes[i]))
            return t

        for h, k in hashes.items():
            x = torch.where(x==k, torch.full_like(x, h), x)

        if x.shape[-1] == 1:
            x = x.squeeze(-1)
        if len(x.shape) > 1:
            return torch.stack([self(x[i], classes, hashes, *args, **kwargs) for i in range(x.shape[0])])

        """
        false_ids = filter(lambda i: not i in x, torch.arange(x.min(), x.max()).to(x.device))
        for f in false_ids:
            x = torch.where(x==f, x, torch.full_like(x, -1))
        """
        t =  oneHot(x, dim=classes)
        if self.make_channels:
            t = t.transpose(-1, -2)
        if self.flatten:
            t = t.reshape(*t.shape[:-2], t.shape[-2]*t.shape[-1])
        return t

    def invert(self, x, hashes=None, original_size=None):
        hashes = hashes or self.hashes
        if issubclass(type(x), (list, tuple)):
            t = []
            for i in range(len(x)):
                t.append(self.invert(x[i], hashes=hashes[i]))
            return t

        if self.flatten:
            n_seq = x.shape[-1] // self.classes
            x = x.reshape(*x.shape[:-1], self.classes, n_seq)
        if self.make_channels:
            x = x.transpose(-2, -1)

        out = fromOneHot(x)
        for h, k in hashes.items():
            if issubclass(type(out), np.ndarray):
                np.place(out, out==h, [k])
            else:
                out = torch.where(out==h, torch.full_like(out, k), out)
        return out


class Itemize(Transform):
    def __init__(self, index_dims, normalize_indexes=True, target_dim=None, return_indices = True, post_transform=None):
        super().__init__()
        index_dims = torch.Tensor(index_dims)
        assert torch.unique(index_dims[1:] - index_dims[:-1]) == 1, \
            "index dimensions should be contiguous ; please reshape your tensors before"
        self.index_dims = index_dims
        self.normalize_indexes = normalize_indexes
        self.target_dim = target_dim
        self._target_dim_tmp = None
        self.return_indices = return_indices
        self.post_transform = post_transform
        if post_transform:
            self.post_transform = checklist(post_transform, n=2)

    def scale(self, data):
        if self.post_transform is not None:
            if self.post_transform[0] is not None:
                self.post_transform[0].scale(data[0])
            if self.post_transform[1] is not None:
                self.post_transform[1].scale(data[1])

    def __call__(self, x, *args, batch_first=True, **kwargs):
        index_dims = [int(i) for i in self.index_dims]
        if batch_first:
            index_dims = [i+1 for i in index_dims]
        norm = torch.Tensor([x.shape[i] for i in index_dims])
        if self.target_dim is None:
            self._target_dim_tmp = norm
        n = list(torch.meshgrid(*tuple([torch.arange(norm[i]) for i in range(len(norm))])))
        n[0], n[1] = n[1], n[0]
        n = torch.stack(n, axis=-1).reshape(int(torch.prod(norm)), 2)
        if self.normalize_indexes:
            n = n / (norm-1)
        if batch_first:
            n = n.unsqueeze(0).repeat(x.shape[0], 1, 1)
        reshape_args = (*tuple(x.shape[:index_dims[0]]),
                        np.prod(x.shape[slice(index_dims[0], index_dims[-1]+1)]),
                        *tuple(x.shape[index_dims[1]+1:]))
        x = x.reshape(*reshape_args)
        n = n.to(x.device)
        if self.post_transform is not None:
            if self.post_transform[1] is not None:
                x = self.post_transform[1](x)
            if self.post_transform[0] is not None:
                n = self.post_transform[0](n)

        if self.return_indices:
            return [n, x]
        else:
            return x

    def invert(self, x, target_dim=None, ordered=True, batch_first=True):
        n = None
        index_dims = [int(i) for i in self.index_dims]
        if batch_first:
            index_dims = [i + 1 for i in index_dims]
        if issubclass(type(x), (list, tuple)):
            n = x[0]; x = x[1]
        target_dim = target_dim or self.target_dim or self._target_dim_tmp
        if target_dim is None:
            print('If no target dim is provided, must be following a call()')
        reshape_args = (*x.shape[:index_dims[0]], *target_dim, *x.shape[index_dims[0] + 1:])
        if ordered:
            x_inv = x.reshape(reshape_args)
        else:
            assert n is not None, "not ordered inversion requires corresponding indices"
            n = (n * np.array(target_dim)).astype('int')
            x_inv = torch.zeros(reshape_args)
            for i in range(n.shape[self.index_dims[0]]):
                current_idx = n[..., i, :][self.index_dims[0]]
                target_idx = [slice(None)]*len(x_inv.shape)
                target_idx[self.index_dims[0]:self.index_dims[1]+1] = current_idx
                x_inv.__setitem__(target_idx, x[..., i, :])

        if self.post_transform:
            if self.post_transform[1] is not None:
                x_inv = self.post_transform[1].invert(x_inv)
        return x_inv



## Audio Transforms
import torchaudio

class AudioTransform(Transform):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def get_timing(self, x, time=None, **kwargs):
        return time


class ComposeAudioTransform(ComposeTransform):

    def forward(self, x, *args, **kwargs):
        time = kwargs.get('time')
        for t in self.transforms:
            x = t(x)
            if time is not None and hasattr(t, "get_timing"):
                time = t.get_timing(x, **kwargs)
                kwargs['time'] = time
        if time is None:
            return x
        else:
            return x, time


class Mono(AudioTransform):
    def __init__(self, mixmode="left"):
        self.mixmode = mixmode

    def forward(self, x):
        if isinstance(x, list):
            return [self(x_i) for x_i in x]
        if x.size(0) == 1 or self.mixmode == "left":
            return x[0].unsqueeze(0)
        if self.mixmode == "mix":
            return torch.mean(x, dim=0).unsqueeze(0)
        elif self.mixmode == "right":
            return x[1].unsqueeze(0)

    def invert(self, x, *args, **kwargs):
        repeat_idxs = [1]*len(x.size()); repeat_idxs[0] = 2
        return x.repeat(*tuple(repeat_idxs))

    def get_timing(self, x, time=None, **kwargs):
        return self(time)


class Stereo(AudioTransform):

    def forward(self, x):
        if isinstance(x, list):
            return [self(x_i) for x_i in x]
        if x.size(0) == 1:
            x = torch.cat([x, x], 0)
        if x.size(0) == 2:
            return x
        if x.size(0) > 2:
            raise NotImplementedError

    def invert(self, x, *args, **kwargs):
        return self(x)

    def get_timing(self, x, time=None, **kwargs):
        return self(time)


def fdiff(x, order=2):
    if order == 1:
        inst_f = torch.cat([x[0].unsqueeze(0), (x[1:] - x[:-1])/2], axis=0)
    elif order == 2:
        inst_f = torch.cat([x[0].unsqueeze(0), (x[2:] - x[:-2])/4, x[-1].unsqueeze(0)], axis=0)
    return inst_f


def fint(x, order=1):
    if order == 1:
        out = x
        out[1:] = out[1:] * 2
        if torch.is_tensor(x):
            out = torch.cumsum(out, axis=0)
        else:
            out = torch.cumsum(out, axis=0)
    elif order == 2:
        out = torch.zeros_like(x)
        out[0] = x[0]; out[-1] = x[-1]

        for i in range(2, x.shape[0], 2):
            out[i] = out[i-2] + 4 * x[i-1]
        for i in reversed(range(1, x.shape[0], 2)):
            out[i-2] = out[i] - 4 * x[i-1]
    return out


def get_window(N):
    n = torch.arange(N)
    return (1.5*N)/(N**2 - 1) * (1 - ((n - (N/2 - 1))/(N/2))**2)


def unwrap(tensor: torch.Tensor):
    """
    unwrap phase for tensors
    :param tensor: phase to unwrap (seq x spec_bin)
    :return: unwrapped phase
    """
    if isinstance(tensor, list):
        return [unwrap(t) for t in tensor]
    if tensor.ndimension() == 2:
        unwrapped = tensor.clone()
        diff = tensor[1:] - tensor[:-1]
        ddmod = (diff + torch_pi)%(2 * torch_pi) - torch_pi
        mask = (ddmod == -torch_pi).bitwise_and(diff > 0)
        ddmod[mask] = torch_pi
        ph_correct = ddmod - diff
        ph_correct[diff.abs() < torch_pi] = 0
        unwrapped[1:] = tensor[1:] + torch.cumsum(ph_correct, 1)
        return unwrapped
    else:
        return torch.stack([unwrap(tensor[i]) for i in range(tensor.size(0))], dim=0)


def get_timing_from_windowing(x, time, sr, length):
    dims = x.shape[:-1]
    time = ((torch.cumsum(torch.ones(*dims), dim=-1) - 1) * length) / sr
    return time.unsqueeze(-1)

class STFT(AudioTransform):
    def __init__(self, n_fft, win_length=None, hop_length=None, normalized=False, win_fn=None, retain_phase=False):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.hop_length = hop_length or self.win_length // 2
        self.normalized = normalized or False
        self.win_fn = win_fn or torch.hann_window
        self.griffin_lim = torchaudio.transforms.GriffinLim(n_fft=self.n_fft, win_length=self.win_length, power=1, rand_init=True,
                                                            training=True, center=True, window_fn=self.win_fn, hop_length=self.hop_length, n_iter=64)
        self._retain_phase = retain_phase
        self.current_phase = None

    def forward(self, x, window=None):
        if isinstance(x, list):
            return [self(x[i]) for i in range(len(x))]
        window = window if window is not None else self.win_fn(self.win_length)
        transform = torch.view_as_complex(torch.stft(x, n_fft=self.n_fft, win_length=self.win_length, center=True,
                               hop_length=self.hop_length, normalized=self.normalized, window=window))
        if hasattr(self, "_retain_phase"):
            if self._retain_phase:
                self.current_phase = transform.angle()
        else:
            self._retain_phase = False
            self.current_phase = None
        if transform.ndimension() == 2:
            return transform.transpose(0, 1)
        else:
            return transform.transpose(1, 2)

    def invert(self, x, window=None):
        if x.ndimension() == 2:
            x = x.transpose(0, 1)
        else:
            x = x.transpose(1, 2)
        window = window if window is not None else self.win_fn(self.win_length)
        if torch.is_complex(x):
            x = torch.view_as_real(x)
            return torch.istft(x, n_fft=self.n_fft, window=window,
                                               win_length=self.win_length, hop_length=self.hop_length,
                                               normalized=self.normalized, center=True)
        else:
            if not hasattr(self, "_retain_phase"):
                self._retain_phase = False
            if  x.size(-1) == 2:
                return torch.istft(x, n_fft=self.n_fft, window=window,
                                                win_length=self.win_length, hop_length=self.hop_length,
                                                normalized=self.normalized, center=True)
            else:
                if self._retain_phase:
                    assert self.current_phase is not None, "when retain_phase=True a previous forward is needed to correctly apply the retained phase"
                    assert self.current_phase.shape == x.shape, "when retain_phase is on inverted input must correspond to the previous forward pass"
                    x = x * torch.exp(1j*self.current_phase)
                    return torch.istft(torch.view_as_real(x), n_fft=self.n_fft, window=window,
                                               win_length=self.win_length, hop_length=self.hop_length,
                                               normalized=self.normalized, center=True)
                else:
                    # phase = 2*torch_pi*torch.rand(x.shape[0]).unsqueeze(1).repeat(1, x.shape[1])
                    # phase = 2*torch_pi*torch.rand_like(x)
                    # x = x * torch.exp(1j * phase)
                    # if x.size(1) == 1:
                    #     return torch.view_as_real(x[:, 0]).ifft(1)[:, 0].unsqueeze(0)
                    # else:
                    #     return torch.istft(torch.view_as_real(x), n_fft=self.n_fft, window=window,
                    #                        win_length=self.win_length, hop_length=self.hop_length,
                    #                        normalized=self.normalized, center=True)
                    return self.griffin_lim(x)

    def retain_phase(self, retain_phase):
        if not hasattr(self, "_retain_phase"):
            self._retain_phase = False
        retain_phase = bool(retain_phase)
        self._retain_phase = retain_phase

    def get_timing(self, x, sr=None, time=None, **kwargs):
        if (sr is None) or (time is None):
            raise NoTimingError
        return get_timing_from_windowing(x, time, sr, self.hop_length)

class DCT(AudioTransform):
    def __init__(self, n_fft, win_length=None, hop_length=None, normalized=False, win_fn=None):
        super(DCT, self).__init__()
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.hop_length = hop_length or self.win_length // 2
        self.normalized = normalized or False
        self.win_fn = win_fn or torch.hann_window
        self.griffin_lim = torchaudio.transforms.GriffinLim(n_fft=self.n_fft, win_length=self.win_length, power=1,
                                                            window_fn=self.win_fn, hop_length=self.hop_length, n_iter=32)

    def forward(self, x):
        if isinstance(x, list):
            return [self(x[i]) for i in range(len(x))]

        slice_dim = 0 if x.ndim == 1 else 1
        x = window_data(x, self.win_length, self.hop_length, slice_dim)
        transform = dct.dct(x)
        return transform

    def invert(self, x):
        transform_inv = dct.idct(x)
        return overlap_add(transform_inv, self.win_length, self.hop_length, dim=-2, window_fn=self.win_fn)

    def get_timing(self, x, sr=None, time=None, **kwargs):
        if (sr is None) or (time is None):
            raise NoTimingError
        return get_timing_from_windowing(x, time, sr, self.hop_length)

class Mel(AudioTransform):
    def __init__(self, fs, n_fft, win_length=None, hop_length=None, normalized=False, win_fn=None,
                 n_mels=128, f_min=0., f_max= None):
        super(Mel, self).__init__()
        self.win_fn = win_fn or torch.hann_window
        self.fs = fs
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.hop_length = hop_length or self.win_length // 2
        self.transform = torchaudio.transforms.MelSpectrogram(fs, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                                          normalized=normalized, window_fn=self.win_fn, n_mels=n_mels,
                                                          f_min=f_min, f_max=f_max)

    def forward(self, x):
        if isinstance(x, list):
            return [self(x[i]) for i in range(len(x))]
        transform = self.transform(x)
        if transform.ndimension() == 2:
            return transform.transpose(0, 1)
        else:
            return transform.transpose(1, 2)

    def invert(self, x):
        if x.ndimension() == 2:
            x = x.transpose(0, 1)
        else:
            x = x.transpose(1, 2)
        if torch.is_complex(x):
            x = torch.view_as_real(x)
            return torch.istft(x, n_fft=self.n_fft, window=self.win_fn(self.win_length),
                               win_length=self.win_length, hop_length=self.hop_length,
                               normalized=self.normalized, center=True)
        elif x.size(-1) == 2:
            return torch.istft(x, n_fft=self.n_fft, window=self.win_fn(self.win_length),
                               win_length=self.win_length, hop_length=self.hop_length,
                               normalized=self.normalized, center=True)
        else:
            return self.griffin_lim(x)

    def get_timing(self, x, sr=None, time=None, **kwargs):
        if (sr is None) or (time is None):
            raise NoTimingError
        return get_timing_from_windowing(x, time, sr, self.hop_length)



class Magnitude(AudioTransform):
    def __init__(self, normalize=None, contrast=None, shrink=1, **kwargs):
        super(Magnitude, self).__init__()
        self.normalize = None
        self.constrast = contrast
        if normalize is not None:
            self.normalize = Normalize(**normalize)
        self.shrink = shrink

    def preprocess(self, x):
        if self.constrast is None:
            return x
        elif self.constrast == "log":
            return torch.log(x/self.shrink)
        elif self.constrast == "log1p":
            return torch.log1p(x/self.shrink)
        else:
            raise ValueError('constrast %s not valid for Magnitude transform'%self.constrast)

    def scale(self, x):
        if isinstance(x, list):
            x = [self.preprocess(x[i].abs()) for i in range(len(x))]
        else:
            x = self.preprocess(x.abs())
        if self.normalize is not None:
            self.normalize.scale(x)

    def forward(self, x):
        if isinstance(x, list):
            return [self(x_i) for x_i in x]

        out = torch.abs(x)
        out = self.preprocess(out)
        if self.normalize is not None:
            out = self.normalize(out)
        return out

    def invert(self, x):
        if isinstance(x, list):
            return [self.invert(x_i) for x_i in x]
        if self.normalize is not None:
            x = self.normalize.invert(x)
        if self.constrast == "log":
            x = torch.exp(x)*self.shrink
        elif self.constrast == "log1p":
            x = torch.expm1(x)*self.shrink
        return x

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(2, 1)

class Phase(AudioTransform):
    def __init__(self, unwrap=True, normalize=None,**kwargs):
        super(Phase, self).__init__()
        self.unwrap = unwrap
        self.normalize = None
        if normalize is not None:
            self.normalize = Normalize(**normalize)

    def scale(self, x):
        if isinstance(x, list):
            x = [x[i].angle() for i in range(len(x))]
        else:
            x = x.angle()
        if self.unwrap:
            x = unwrap(x)
        if self.normalize is not None:
            self.normalize.scale(x)

    def forward(self, x):
        if isinstance(x, list):
            return [self(x_i) for x_i in x]
        phase = x.angle()

        if self.unwrap:
            # phase = torch.from_numpy(np.unwrap(phase.numpy()))
            phase = unwrap(phase)

        # ax[0].imshow(phase[0], aspect="auto")
        if self.normalize is not None:
            phase = self.normalize(phase)
        return phase

    def invert(self, x, *args, **kwargs):
        if self.normalize is not None:
            x = self.normalize.invert(x)

        # ax[1].imshow(x[0], aspect="auto")
        if self.unwrap:
            x = torch.fmod(x, 2*torch_pi)
        return x


class InstantaneousFrequency(AudioTransform):
    methods = ['backward', 'forward', 'central', 'fm-disc']

    def __repr__(self):
        return "<preprocessing InstantaneousFrequency with method: %s, normalize: %s, mode: %s>"%(self.method, self.normalize, self.mode)

    def __init__(self, method="backward", wrap=False, weighted = False, normalize=None, mode=None):
        assert method in self.methods
        self.method = method
        self.wrap = wrap
        self.weighted = weighted
        self.normalize = normalize
        self.mode = mode
        if self.normalize is not None:
            self.normalize = Normalize(**normalize)

    def scale(self, data):
        self.normalize.scale(self.get_if(data))

    def get_if(self, data):
        if issubclass(type(data), list):
            return [self.get_if(i) for i in data]

        if self.method in ['forward', 'backward', 'central']:
            phase = unwrap(torch.angle(data))
            # mag = np.abs(data)
            if self.method == "backward":
                inst_f = fdiff(phase, order=1)
                inst_f[1:] /= torch_pi
            elif self.method == "forward":
                inst_f = torch.flip(fdiff(torch.flip(phase, axis=0), order=1), axis=0)
                inst_f[:-1] /= -torch_pi
            if self.method == "central":
                inst_f = fdiff(phase, order=2)
                inst_f[1:-1] /= torch_pi
            if self.weighted:
                window = get_window(inst_f.shape[0]).unsqueeze(1)
                inst_f = window * inst_f

        if self.method == "fm-disc":
            real = data.real; dreal = fdiff(real, order=1)
            imag = data.imag; dimag = fdiff(imag, order=1)
            inst_f = (real * dimag - dreal * imag) / (real**2 + imag**2)
            inst_f = inst_f / (2*torch_pi)

        if self.wrap:
            inst_f = torch.fmod(inst_f, 2*torch_pi)

        return inst_f

    def __call__(self, data, batch_first=False):
        if isinstance(data, list):
            return [self(x_i) for x_i in data]
        if batch_first:
            return torch.stack([self(data[i], batch_first=False) for i in range(data.shape[0])])
        inst_f = self.get_if(data)
        if self.normalize is not None:
            inst_f = self.normalize(inst_f)
        return inst_f

    def invert(self, data, batch_first=False):
        if issubclass(type(data), list):
            return [self.invert(x, batch_first=batch_first) for x in data]
        if batch_first:
            if torch.is_tensor(data):
                return torch.stack([self.invert(data[i], batch_first=False) for i in range(data.shape[0])])
            else:
                return torch.stack([self.invert(data[i], batch_first=False) for i in range(data.shape[0])])
        if self.normalize:
            data = self.normalize.invert(data)
        if self.wrap:
            data = unwrap(data)
        if self.method == "backward":
            data[1:] *= torch_pi
            phase = fint(data, order=1)
        if self.method == "forward":
            data[:-1] *= -torch_pi
            if torch.is_tensor(data):
                phase = torch.flip(fint(torch.flip(data, axis=0), order=1), axis=0)
            else:
                phase = torch.flip(fint(torch.flip(data, axis=0), order=1), axis=0)
        elif self.method == "central":
            data[1:-1] *= torch_pi
            phase = fint(data, order=2)

        return phase


class Polar(AudioTransform):
    magnitude_transform = Magnitude
    phase_transform = Phase

    def __init__(self, *args, mag_options={}, phase_options={}, **kwargs):
        super(Polar, self).__init__()
        self.transforms = [self.magnitude_transform(**mag_options),
                           self.phase_transform(**phase_options)]

    def __getitem__(self, item):
        return self.transforms[item]

    def forward(self, x):
        # fft = super(Polar, self).forward(x)
        return [self.transforms[0](x), self.transforms[1](x)]

    def invert(self, x):
        mag, phase = self.transforms[0].invert(x[0]), self.transforms[1].invert(x[1])
        fft = mag*torch.exp(1j*phase)
        return fft

    def scale(self, x):
        self.transforms[0].scale(x)
        self.transforms[1].scale(x)


class PolarInst(Polar):
    phase_transform = InstantaneousFrequency

