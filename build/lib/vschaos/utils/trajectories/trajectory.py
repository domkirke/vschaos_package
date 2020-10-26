import numpy as np, abc
from numpy import ceil, floor, round
ROUND_STRATEGY = round

# dummy class that is used to ignore a given argument in trajectory classes
class Ignore(object):
    pass

class Trajectory(object):
    _callback = None
    @abc.abstractmethod
    def get_callback(self):
        return self._callback
    @abc.abstractmethod
    def set_callback(self):
        raise AttributeError('callback of trajectory cannot be changed')
    @abc.abstractmethod
    def del_callback(self):
        raise AttributeError('callback of trajectory cannot be deleted')
    trajectory_callback = property(get_callback, set_callback, del_callback, 'callback function of trajectory')

    def __init__(self, *args, **kwargs):
        self.call_args = args
        self.call_kwargs = kwargs

    def __call__(self, *args, **kwargs):
        init_args = self.call_args
        init_kwargs = dict(self.call_kwargs)
        for i in range(len(args)):
            if not issubclass(type(args[i]), Ignore):
                init_args[i] = args[i]
        for k, v in kwargs.items():
            init_kwargs[k] = v

        assert init_kwargs.get('n_steps') is not None, "Trajectory must have an n_steps keyword (int or float)"
        if init_kwargs.get('fs') is not None:
            init_kwargs['n_steps'] = ROUND_STRATEGY(init_kwargs['n_steps']*init_kwargs['fs']).astype(np.int)
        return self._callback(*init_args, **init_kwargs)

    def __getitem__(self, idx):
        if issubclass(type(idx), int):
            if idx > len(self.call_args):
                return AttributeError('itemargument %d of object %s does not exist'%(idx, self))
            return self.call_args[idx]
        elif issubclass(type(idx), str):
            if not idx in self.call_kwargs.keys():
                return AttributeError('keyword argument %s of object %s does not exist'%(idx, self))
            return self.call_kwargs[idx]
        else:
            return AttributeError('__getitem__ method of object %s can be int or str, but not %s'%(self, type(idx)))


    def __setitem__(self, idx, item):
        if issubclass(type(idx), int):
            current_args = list(self.call_args)
            if item >= len(current_args):
                current_args = current_args * [None]*(idx - len(current_args))
                current_args[idx] = item
        elif issubclass(type(idx), str):
            self.call_kwargs[idx] = item
        else:
            return AttributeError('__setitem__ method of object %s can be int or str, but not %s'%(self, type(idx)))




