import numpy as np, torch

class NoShapeError(Exception):
    pass

class ShapeError(Exception):
    def __init__(self, true_shape, false_shape):
        self.true_shape = true_shape
        self.false_shape = false_shape
    def __repr__(self):
        print('ShapeError(%s != %s)'%(self.true_shape, self.false_shape))

def is_same(x):
    return x.count(x[0]) == len(x)



## Memmap reading routines

def get_final_dim(sl, shape):
    final_dim = [None]*len(sl)
    for i, s in enumerate(sl):
        if isinstance(s, int):
            final_dim[i] = 1
        elif isinstance(s, slice):
            start = s.start or 0
            stop = s.stop or shape[i]
            step = s.step or 1
            final_dim[i] = np.round((stop - start)/step).astype(np.int)
    return np.array(final_dim)

def load_memmap(fp, idx, shape, strides, dtype, squeeze=True):
    sort = np.argsort(strides)
    if not np.unique(np.diff(sort)).all(-1):
        raise NotImplementedError
    offsets = np.array([0])
    shapes = np.array([shape])
    end_idx = len(idx)
    for i in reversed(idx):
        if i == slice(None):
            end_idx -= 1
        else:
            break
    final_shape = get_final_dim(idx, shape)
    for dim in range(end_idx):
        if isinstance(idx[dim], int):
            offsets = offsets + strides[dim] * idx[dim]
            shapes[:, dim] = 1
        elif isinstance(idx[dim], slice):
            if idx[dim].step is None:
                start = idx[dim].start if idx[dim].start is not None else 0
                offsets = offsets + strides[dim] * start
                shapes[:, dim] = final_shape[dim]
            else:
                dim_idx = list(range(x.shape[dim]))[idx[dim]]
                # if rest is slice(None, None), can be accelerated (check before?)
                offsets = (offsets[np.newaxis] + (np.array(dim_idx) * strides[dim])[:, np.newaxis]).flatten()
                offsets.sort()
                shapes = shapes.repeat(len(dim_idx), 0)
                shapes[:, dim] = 1
    arr = np.zeros(np.prod(final_shape), dtype=dtype)
    current_idx = 0
    for i in range(len(offsets)):
        current_slice = slice(current_idx, current_idx + np.prod(shapes[i]))
        arr[current_slice] = np.array(
            np.memmap(fp, dtype=dtype, mode="r", shape=tuple(shapes[i]), offset=int(offsets[i]))).flatten()
        current_idx += np.prod(shapes[i])
    arr = arr.reshape(final_shape)
    if squeeze:
        arr = arr.squeeze()
    return arr

## Selector objects

class Selector(object):
    """takes everything"""
    def __init__(self, **kwargs):
        pass

    def __repr__(self):
        return "Selector()"

    def __getitem__(self, idx):
        return IndexPick(idx)

    def get_shape(self, shape):
        return shape

    def __call__(self, file, shape=None, dtype=np.float, time=None, drop_time=None, **kwargs):
        if drop_time:
            assert time is not None, "if drop_time is true the keyword time is needed"
        if drop_time:
            return np.array(np.memmap(file, dtype=dtype, mode='r', offset=0, shape=shape)), time
        else:
            return np.array(np.memmap(file, dtype=dtype, mode='r', offset=0, shape=shape))

class IndexPick(Selector):
    def __init__(self, idx=None, axis=1, **kwargs):
        assert idx is not None
        self.idx =  idx; self.axis = axis

    def  __repr__(self):
        return "IndexPick(%d)"%self.idx

    def __getitem__(self):
        raise NotImplementedError

    def get_shape(self, shape):
        new_shape = list(shape); del new_shape[self.axis]
        return new_shape

    def __call__(self, file, shape, strides, dtype=np.float,  time=None, drop_time=None, **kwargs):
        if drop_time:
            assert time is not None, "if drop_time is true the keyword time is needed"
        idx = [slice(None)]*len(shape); idx[self.axis] = self.idx
        if drop_time:
            return load_memmap(file, idx, shape, strides, dtype, squeeze=False).squeeze(self.axis), time[idx]
        else:
            return load_memmap(file, idx, shape, strides, dtype, squeeze=False).squeeze(self.axis)
        # if drop_time:
        #     return np.array(np.memmap(file, dtype=dtype, mode='r', offset=offset, shape=self.get_shape(shape))), time[idx]
        # else:
        #     return np.array(np.memmap(file, dtype=dtype, mode='r', offset=offset, shape=self.get_shape(shape)))


class SequencePick(Selector):
    def __init__(self, sequence_length=None, random_idx=False, offset=0, relative_timing=False, axis=0, **kwargs):
        assert sequence_length
        self.sequence_length =  sequence_length
        self.random_idx = random_idx
        self.offset = offset
        self.relative_timing = relative_timing
        self.axis = axis

    def  __repr__(self):
        return "SequencePick(%d, random_idx:%s)"%(self.sequence_length, self.random_idx)

    def __getitem__(self):
        raise NotImplementedError

    def get_shape(self, shape):
        new_shape = list(shape)
        new_shape[self.axis] = self.sequence_length
        return tuple(new_shape)

    def __call__(self, file, shape=None,  dtype=np.float, idx=None, strides=None,
                 time=None, drop_time=None, sr=None, **kwargs):
        # get start
        start_idx = 0
        if not (shape[self.axis] - self.sequence_length <= 0):
            if self.random_idx:
                start_idx = np.random.randint(shape[self.axis] - self.sequence_length)
        else:
            start_idx = self.offset

        idx = [slice(None)]*len(shape)
        if not (start_idx + self.sequence_length) > shape[self.axis]:
            idx[self.axis] = slice(start_idx, start_idx+self.sequence_length)

        data = load_memmap(file, idx, shape=shape, dtype=dtype, strides=strides, squeeze=False)
        if data.shape[self.axis] < self.sequence_length:
            pads = [(0,0)]*len(shape)
            pads[self.axis] = (0, self.sequence_length - data.shape[self.axis])
            data = np.pad(np.array(data), pad_width=pads, mode="constant", constant_values=0)
        if drop_time:
            if len(time.shape) != data.shape[0]:
                time = np.arange(start_idx, start_idx + self.sequence_length)/sr + time[0]
            else:
                time = time[idx:idx+self.sequence_length][:, np.newaxis]
            if self.relative_timing:
                time = time - time.min()
            return data, time
        else:
            return data

class UnwrapBatchPick(SequencePick):
    def get_shape(self, shape):
        return shape[1:]


class SequenceBatchPick(Selector):
    def __init__(self, sequence_length=None, batches=32, overlap=None, random_idx=True, **kwargs):
        assert sequence_length
        self.sequence_length =  sequence_length
        self.random_idx = random_idx
        self.batches = batches
        self.overlap = overlap or sequence_length//2

    def  __repr__(self):
        if self.random_idx:
            return "SequenceBatchPick(%d, random_idx:%s, batches:%s)"%(self.sequence_length, self.random_idx,
                                                                               self.batches)
        else:
            return "SequenceBatchPick(%d, random_idx:%s, overlap:%s)"%(self.sequence_length, self.random_idx,
                                                                               self.overlap)
    def __getitem__(self, i):
        if self.random_idx:
            return SequencePick(self.sequence_length, random_idx=True)
        else:
            return SequencePick(self.sequence_length, random_idx=False, offset=(i*self.overlap))

    def get_shape(self, shape):
        return (self.batches, self.sequence_length, *shape[1:])

    def __call__(self, file, shape=None, axis=0, dtype=np.float, idx=None, strides=None, **kwargs):
        assert axis==0, "memmap indexing on axis > 0 is not implemented yet"
        if self.random_idx:
            indices = np.array([np.random.randint(shape[0] - self.sequence_length) for _ in range(self.batches)])
        else:
            indices = np.array([i*self.overlap for i in range(self.batches)])

        items = np.zeros(self.get_shape(shape), dtype=dtype)
        for i, idx in enumerate(indices):
            offset = (idx * shape[1])*strides[1]
            load_shape = shape if shape[0] < self.sequence_length else self.get_shape(shape)[1:]
            data = np.memmap(file, dtype=dtype, mode='r', offset=offset, shape=load_shape)
            if data.shape[0] < self.sequence_length:
                pads = [(0,0)]*len(shape)
                pads[0] = (0, self.sequence_length - data.shape[0])
                data = np.pad(np.array(data), pad_width=pads, mode="constant", constant_values=0)
            items[i] = data

        return items


class RandomPick(Selector):
    def __init__(self, range=None, axis=0, **kwargs):
        self.range = range
        if self.range:
            self.range = tuple(self.range)
        self.axis = axis

    def  __repr__(self):
        return "RandomPick(range=%s)"%self.range

    def get_shape(self, shape):
        return shape[self.axis+1:]

    def __call__(self, file, shape=None, axis=0, dtype=np.float,  strides=None, **kwargs):
        assert axis==0, "memmap indexing on axis > 0 is not implemented yet"
        random_range = self.range or (0, shape[0])
        idx = np.random.randint(*random_range)
        offset = (idx * shape[1])*strides[1]
        return np.memmap(file, dtype=dtype, mode='r', offset=offset, shape=self.get_shape(shape))


class RandomRangePick(Selector):
    def __init__(self, range=None, length=64, axis=0, **kwargs):
        self.range = range; self.length = length
        if self.range:
            self.range = tuple(self.range)
        self.axis = axis

    def __getitem__(self, idx):
        return RandomPick(self.range)

    def  __repr__(self):
        return "RandomRangePick(length=%s, range=%s)"%(self.length, self.range)

    def get_shape(self, shape):
        return (self.length, *shape[self.axis+1:])

    def __call__(self, file, shape=None, axis=0, dtype=np.float,  strides=None, **kwargs):
        assert axis==0, "memmap indexing on axis > 0 is not implemented yet"
        random_range = self.range or (0, shape[0])
        chunks = [None]*self.length
        idx_buffer = np.array(range(*random_range))
        np.random.shuffle(idx_buffer)
        for i in range(self.length):
            offset = (idx_buffer[i] * shape[1])*strides[1]
            chunks[i] = np.memmap(file, dtype=dtype, mode='r', offset=offset, shape=self.get_shape(shape))
        return np.stack(chunks, axis=0)


class OfflineEntry(object):
    """
    Simple object that contains a file pointer, and a selector callback. Don't have the *shape* attribute if not loaded
    once first
    """

    def __repr__(self):
        return "OfflineEntry(lambda %s of file %s)" % (self.selector, self.file)

    def __init__(self, file, selector=Selector(), dtype=None, shape=None, strides=None, target_length=None,
                 time=None, drop_time=False, sr=None):
        """
        :param file: numpy file to load (.npy / .npz)
        :type file: str
        :param func: callback loading the file
        :type func: function
        :param dtype: optional cast of loaded data
        :type dtype: numpy.dtype
        :param target_length: specifies a target_length for the imported data, pad/cut in case
        :type target_length: int
        """

        self._pre_shape = shape
        self.target_length = target_length
        if issubclass(type(file), OfflineEntry):
            self.file = file.file
            self.selector = selector or file.selector
            self.strides = strides or file.strides
            self._pre_shape = file.shape
            self._post_shape = self.selector.get_shape(file._pre_shape)
            if not target_length is None:
                self._post_shape = tuple(target_length)
            self._dtype = file.dtype
            self.drop_time = file.drop_time
        else:
            self.file = file
            self.selector = selector
            self.strides = strides
            self._dtype = dtype

        if shape is not None:
            if self.selector is not None:
                self._post_shape = self.selector.get_shape(self._pre_shape)
            else:
                self._post_shape = self._pre_shape

        self.time = time
        self.drop_time = drop_time
        self.sr = sr

    def __call__(self, file=None):
        """
        loads the file and extracts data
        :param file: optional file path
        :type file: str
        :return: array of data
        """
        file = file or self.file
        data = self.selector(file, shape=self._pre_shape, dtype=self.dtype, strides=self.strides,
                             time=self.time, drop_time=self.drop_time, sr=self.sr)
        time = None
        if self.drop_time:
            data, time = data

        if not self.target_length is None:
            pads = []
            for i, tl in enumerate(self.target_length):
                pads.append((0, tl - data.shape[i]))
            data = np.pad(data, pads, mode="constant", constant_values=0)

        if data.shape != self.shape:
            try:
                data = np.reshape(data, self.shape)
            except:
                raise ShapeError(data.shape, self.shape)

        if self._post_shape is None:
            self._post_shape = data.shape
        if self._dtype is None:
            data = data._dtype
        if self.drop_time:
            return torch.from_numpy(data), time
        else:
            return torch.from_numpy(data)

    @property
    def shape(self):
        shape = self._post_shape or self._pre_shape
        if shape is None:
            self()
            # raise NoShapeError('OfflineEntry has to be called once to retain shape')
        return self._post_shape

    @property
    def dtype(self):
        if self._dtype is None:
            raise TypeError('type of entry %s is not known yet' % self)
        return self._dtype

    def split(self, axis=0):
        if axis > 0:
            raise NotImplementedError
        if self.shape is None:
            raise NoShapeError()
        if axis > len(self.shape):
            raise ValueError('%s with shape %s cannot be split among axis %d' % (type(self), self.shape, axis))

        if len(self.shape) == 1:
            entries = [self]
        else:
            entries = [None] * self.shape[axis]
            for i in range(self.shape[axis]):
                # entries[i] = type(self)(self.file, dtype=self.dtype, shape=self.shape, strides=self.strides, selector=self.selector[i])
                entries[i] = type(self)(self.file, dtype=self.dtype, shape=self._pre_shape, strides=self.strides,
                                        selector=self.selector[i], time=self.time, drop_time=self.drop_time)
        return entries


class OfflineDataList(object):
    """
    OfflineDataList is suupposed to be a ordered list of :class:`OfflineEntry`, that are callback objects that read files
    and pick data inside only once they are called. :class:`OfflineDataList` dynamically loads requested data when function
    :function:`__getitem__`is loaded, such that it can replace a casual np.array. Some functions of np.array are also
    overloaded such that squeeze / unsqueeze, based a sequence of data transforms that are applied once the data is
    loaded.
    """
    EntryClass = OfflineEntry

    def __repr__(self):
        string = '[\n'
        for e in self.entries:
            string += f"\t{str(e)},\n"
        string += ']'
        return string

    def __iter__(self):
        for entry in self.entries:
            yield entry()

    def __len__(self):
        return len(self.entries)

    @property
    def shape(self):
        if self._shape is None or self._update_shape:
            try:
                shapes = [e.shape for e in self.entries]
            except NoShapeError:
                self.check_entries(self.entries)
                shapes = [e.shape for e in self.entries]
            if is_same(shapes):
                self._shape = (len(self.entries), *shapes[0])
            self._update_shape = False
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        if self._dtype is None:
            self._dtype = self[:].dtype
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = np.dtype(value)

    def cast(self, data):
        if self._dtype is None:
            return data
        else:
            return data.astype(self._dtype)

    def __radd__(self, l):
        return l + self.entries

    def __setitem__(self, idx, elt):
        if issubclass(type(elt), OfflineEntry):
            if not issubclass(type(idx), tuple):
                self.entries[idx] = elt
                self._update_shape = True
            else:
                raise IndexError('cannot set elements in offline files')
        else:
            raise IndexError('class of OfflineDataList elements must be OfflineEntry')

    def __getitem__(self, ids):
        if hasattr(ids, '__iter__') or isinstance(ids, slice):
            if isinstance(ids, slice):
                current_entries = self.entries[ids]
            elif isinstance(ids, (np.ndarray)) or torch.is_tensor(ids):
                current_entries = np.array(self.entries)[ids]
            outs = [];
            time = None
            for c in current_entries:
                if c.drop_time:
                    if time is None:
                        time = []
                    out, t = c()
                    outs.append(out);
                    time.append(t)
                else:
                    outs.append(c())
            if time is not None:
                assert len(time) == len(outs)
                time = np.array(time)
                if len(time.shape) == 1:
                    time = time[:, np.newaxis]
            # outs = [c() for c in current_entries]
            for t, kw in self._transforms:
                outs = t(outs, **kw)
            if time is not None:
                return outs, time
            else:
                return outs

        elif type(ids) == int:
            out = self.entries[ids]()
            if self.entries[ids].drop_time:
                out, time = out
            # out = self.cast(self.entries[ids]())
            for t, kw in self._transforms:
                kw = dict(kw)
                if kw.get('axis') is not None:
                    kw['axis'] = kw['axis'] - 1 if kw['axis'] >= 0 else kw['axis']
                out = t(out, **kw)
            if self._dtype:
                out = out.astype(self._dtype)
            if self.entries[ids].drop_time:
                return out, time
            else:
                return out
        else:
            return self[ids[0]].__getitem__(ids[1:]).astype(self._dtype)

    def take(self, ids):
        """
        return the entries object at given ids
        :param ids: entries ids to be picked
        :type ids: iterable
        :return: list(OfflineEntry)
        """
        target_entries = [self.entries[i] for i in ids]
        entry_list = type(self)(target_entries, transforms=self._transforms)
        return entry_list

    def pad_entries(self, dim=None):
        """
        pad all the entries up to the one with the highest dimension
        :param dim: axis to pad
        :type dim: int
        """
        shapes = np.array([s.shape for s in self.entries]).T
        dim = dim or tuple(range(len(shapes[0].shape)))
        maxs_shape = []
        for d in range(shapes.shape[0]):
            maxs_shape.append(shapes[d].max())
        new_entries = [None] * len(self.entries)
        for i, entry in enumerate(self.entries):
            current_shape = shapes[:, i]
            for d in dim:
                current_shape[d] = maxs_shape[d]
            new_entries[i] = type(entry)(entry, target_length=current_shape)
        self.entries = new_entries
        if self._shape:
            self._shape = (*self._shape[:-1], self.pad_dim)
        else:
            _ = self.shape

    def check_entries(self, entries):
        """
        check all the entries in the given list (an entry is considered wrong is it returns None once called)
        :param entries: list of entries to check
        :type entries: list(OfflineEntry)
        """
        invalid_entries = list(filter(lambda x: entries[x]() is None, range(len(entries))))
        for i in reversed(invalid_entries):
            del entries[i]

    def squeeze(self, dim):
        """
        add a squeeze transformation in the OfflineDataList
        :param dim: dimension to squeeze
        """
        if self._shape is None:
            raise Warning('Tried to squeeze %s, but shape is missing')
        if self._shape[dim] != 1:
            raise ValueError('cannot select an axis to squeeze out which has size not equal to one')
        if dim >= len(self._shape):
            raise np.AxisError('axis 4 is out of bounds for array of dimension 3' % (dim, len(self._shape)))
        self._transforms.append((torch.squeeze, {'dim': dim}))
        new_shape = list(self._shape)
        del new_shape[dim]
        self._shape = tuple(new_shape)

    def expand_dims(self, dim):
        """
        add an unsqueeze transformation in the OfflineDataList
        :param dim: dimension to squeeze
        """
        if self._shape is None:
            print('Tried to squeeze %s, but shape is missing')
        self._transforms.append((torch.unsqueeze, {'dim': dim}))
        if self._shape is not None:
            if dim >= 0:
                self._shape = (*self._shape[:dim], 1, *self._shape[dim:])
            else:
                self._shape = (*self._shape[:dim + 1], 1, *self._shape[dim + 1:])

    def __init__(self, *args, selector=Selector(), check=True, dtype=None, stride=None, transforms=None, padded=False):
        """
        Returns an OfflineDataList
        :param args:
        :param selector_gen:
        :param check:
        :param dtype:
        :param stride:
        :param transforms:
        :param padded:
        """
        self._shape = None
        self._update_shape = True
        self.pad_dim = None
        self._dtype = np.dtype(dtype) if dtype is not None else None
        self._transforms = []

        if len(args) == 1:
            if issubclass(type(args[0]), OfflineDataList):
                self.entries = list(args[0].entries)
                self._transforms = list(args[0]._transforms)
                self._shape = list(args[0]._shape)

                if dtype is None:
                    self._dtype = args[0]._dtype

            elif issubclass(type(args[0]), list):
                entries = []
                for elt in args[0]:
                    if issubclass(type(elt), self.EntryClass):
                        entries.append(elt)
                    elif issubclass(type(elt), OfflineDataList):
                        entries.extend(elt.entries)
                assert len(list(filter(lambda x: not issubclass(type(x), self.EntryClass), entries))) == 0
                self.entries = entries

                # if dtype is None:
                #     self._dtype = self.entries[0]._dtype
            else:
                raise ValueError('expected OfflineDataList or list, but got : %s' % type(args[0]))
        elif len(args) > 1:
            if issubclass(type(args[0]), str):
                if len(args) == 2:
                    file, data = args
                    range_ids = range(0, data.shape[0], stride)
                elif len(args) == 3:
                    file, data, range_ids = args
                self.entries = []
                if data.ndim == 1:
                    self.entries.append(OfflineEntry(file, selector_gen(0)))
                elif data.ndim == 2:
                    self.entries = [self.EntryClass(file, selector_gen(slice_id)) for slice_id in range_ids]
                    if check:
                        self.check_entries(self.entries)
            else:
                raise ValueError('expected (str, int), but got : (%s, %s)' % (type(args[0]), type(args[1])))

        if transforms is not None:
            self._transforms = transforms

        _ = self.shape

        if padded:
            self.pad_entries()


