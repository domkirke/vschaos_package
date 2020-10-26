
import numpy as np
from .trajectory import Trajectory
from .misc import scale_traj

def sawtooth_generator(*args, **kwargs):
    n_steps = kwargs.get('n_steps')
    print(n_steps)
    dim = kwargs.get('dim')
    freq = kwargs.get('freq')
    phase = kwargs.get('phase', 0)
    fs = kwargs.get('fs')
    amplitude = kwargs.get('amp', np.array([0, 1]))
    assert n_steps, "sawtooth_generator needs the n_steps argument"
    assert dim, "sawtooth_generator needs the dim argument"
    assert freq is not None, "sawtooth_generator needs the freq argument"

    if fs is not None:
        freq =  fs / freq

    freq = np.array(freq)
    phase = np.array(phase)
    if len(freq.shape) == 0:
        freq = np.ones(dim) * freq
    if len(phase.shape) == 0:
        phase = np.ones(dim) * phase
    if len(amplitude.shape) == 1:
        amplitude = amplitude[np.newaxis].repeat(dim, axis=0)

    traj = np.zeros((n_steps, dim))
    t = np.arange(n_steps)[:, np.newaxis].repeat(dim, axis=1)
    for i in range(dim):
        ramp_t = (np.array(t[:, i]/freq[i], dtype=np.float) - phase[i])
        traj[np.where(ramp_t>0), i] = np.fmod(ramp_t[np.where(ramp_t>0)], 1)
        traj[np.where(ramp_t<0), i] = 1 + np.fmod(ramp_t[np.where(ramp_t<0)], 1)
        traj[:, i] = scale_traj(traj[:, i], 0, 1, amplitude[i][0], amplitude[i][1])

    return traj

class Sawtooth(Trajectory):
    _callback = sawtooth_generator

def triangle_generator(*args, **kwargs):
    n_steps = kwargs.get('n_steps')
    print(n_steps)
    dim = kwargs.get('dim')
    freq = kwargs.get('freq')
    phase = kwargs.get('phase', 0)
    fs = kwargs.get('fs')
    amplitude = kwargs.get('amp', np.array([0, 1]))
    assert n_steps, "sawtooth_generator needs the n_steps argument"
    assert dim, "sawtooth_generator needs the dim argument"
    assert freq is not None, "sawtooth_generator needs the freq argument"

    if fs is not None:
        freq =  fs / freq

    freq = np.array(freq)
    phase = np.array(phase)
    if len(freq.shape) == 0:
        freq = np.ones(dim) * freq
    if len(phase.shape) == 0:
        phase = np.ones(dim) * phase
    if len(amplitude.shape) == 1:
        amplitude = amplitude[np.newaxis].repeat(dim, axis=0)

    traj = np.zeros((n_steps, dim))
    t = np.arange(n_steps)[:, np.newaxis].repeat(dim, axis=1)
    for i in range(dim):
        ramp_t = (np.array(t[:, i]/freq[i], dtype=np.float) - phase[i])
        traj[np.where(ramp_t>0), i] = np.fmod(ramp_t[np.where(ramp_t>0)], 1)
        traj[np.where(ramp_t<0), i] = 1 + np.fmod(ramp_t[np.where(ramp_t<0)], 1)
        traj[np.where(traj[:,i]>0.5), i] = 1 - traj[np.where(traj[:, i]>0.5), i]
        traj[:, i] = scale_traj(traj[:, i], 0, 0.5, amplitude[i][0], amplitude[i][1])

    return traj

class Triangle(Trajectory):
    _callback = triangle_generator


def sin_generator(*args, **kwargs):
    n_steps = kwargs.get('n_steps')
    print(n_steps)
    dim = kwargs.get('dim')
    freq = kwargs.get('freq')
    phase = kwargs.get('phase', 0)
    fs = kwargs.get('fs')
    amplitude = kwargs.get('amp', np.array([0, 1]))
    assert n_steps, "sawtooth_generator needs the n_steps argument"
    assert dim, "sawtooth_generator needs the dim argument"
    assert freq is not None, "sawtooth_generator needs the freq argument"

    if fs is not None:
        freq =  freq/fs

    freq = np.array(freq)
    phase = np.array(phase)
    if len(freq.shape) == 0:
        freq = np.ones(dim) * freq
    if len(phase.shape) == 0:
        phase = np.ones(dim) * phase
    if len(amplitude.shape) == 1:
        amplitude = amplitude[np.newaxis].repeat(dim, axis=0)

    traj = np.zeros((n_steps, dim))
    t = np.arange(n_steps)[:, np.newaxis].repeat(dim, axis=1)
    for i in range(dim):
        traj[:, i] = np.sin(2*np.pi*freq[i]*t[:, i] + phase[i])
        traj[:, i] = scale_traj(traj[:, i], -1, 1, amplitude[i][0], amplitude[i][1])

    return traj

class Sine(Trajectory):
    _callback = sin_generator

def square_generator(*args, **kwargs):
    n_steps = kwargs.get('n_steps')
    print(n_steps)
    dim = kwargs.get('dim')
    freq = kwargs.get('freq')
    phase = kwargs.get('phase', 0)
    pulse =  kwargs.get('pulse', 0.5)
    fs = kwargs.get('fs')
    amplitude = kwargs.get('amp', np.array([0, 1]))
    assert n_steps, "sawtooth_generator needs the n_steps argument"
    assert dim, "sawtooth_generator needs the dim argument"
    assert freq is not None, "sawtooth_generator needs the freq argument"

    if fs is not None:
        freq =  freq/fs

    freq = np.array(freq)
    phase = np.array(phase)
    pulse = np.array(pulse)
    if len(freq.shape) == 0:
        freq = np.ones(dim) * freq
    if len(phase.shape) == 0:
        phase = np.ones(dim) * phase
    if len(pulse.shape) == 0:
        pulse = np.ones(dim) * pulse
    pulse = -np.log(-1 + 1/np.clip(pulse, 0+1e-12, 1-1e-12))+1
    if len(amplitude.shape) == 1:
        amplitude = amplitude[np.newaxis].repeat(dim, axis=0)

    traj = np.zeros((n_steps, dim))
    t = np.arange(n_steps)[:, np.newaxis].repeat(dim, axis=1)
    for i in range(dim):
        traj[:, i] = np.exp(pulse[i]*np.log(np.cos(2*np.pi*freq[i]*t[:, i] + phase[i])/2+0.5))>0.5
        traj[:, i] = np.exp(pulse[i]*np.log(np.cos(2*np.pi*freq[i]*t[:, i] + phase[i])/2+0.5)) > 0.5
        #traj[:, i] = scale_traj(traj[:, i], 0, 1, amplitude[i][0], amplitude[i][1])

    return traj

class Square(Trajectory):
    _callback = square_generator