#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:09:11 2018

@author: chemla
"""
import numpy as np, pdb
import librosa, librosa.display, librosa.core
import torch, torchaudio
from scipy.signal import buttord, butter, lfilter, lfilter_zi
from .. import DatasetAudio
import matplotlib.pyplot as plt


class SynthesisError(Exception):
    def __init__(self, generator, params):
        self.params = params
        self.generator = generator
        
    def __repr__(self):
        return "erro while using %s with parameters : %s"%(self.generator, self.params)


def additive_generator(T=1, fs=44100, f0=440, n_partials=5, harmonic_decay = 1, inharmonicity=0, mode_phase="rand", fade_length=100, *args, **kwargs):
    t = np.linspace(0, T, round(T*fs))
    sig  = np.zeros(t.shape)
    for i in range(int(n_partials)):
        if mode_phase == "rand":
            phase = np.zeros((round(T*fs)))
            phase.fill(np.random.rand()*2*np.pi)
        else:
            phase=np.zeros((round(T*fs)))
        sig += np.exp(-i*harmonic_decay/n_partials)*np.cos(2*np.pi*f0*(i+1)*np.sqrt(1+inharmonicity*i**2)*t + phase)
    fade_length /= 1000. # fade length in milliseconds
    envelope = np.ones(t.shape)
    fade_length_samples = round(fs*fade_length)
    envelope[:fade_length_samples] = np.hanning(fade_length_samples*2)[:fade_length_samples]
    envelope[-fade_length_samples:] = np.hanning(fade_length_samples*2)[-fade_length_samples:]
    sig = sig * envelope
    sig /= np.max(sig)
    return sig

def pitched_additive_generator(T=1, fs=44100, midi=60, f0=440, n_partials=5, harmonic_decay = 1, inharmonicity=0, mode_phase="rand", fade_length=100, *args, **kwargs):

    f0 = f0 * 2**((midi - 69) / 12)
    t = np.linspace(0, T, round(T*fs))
    sig  = np.zeros(t.shape)
    for i in range(int(n_partials)):
        if mode_phase == "rand":
            phase = np.zeros((round(T*fs)))
            phase.fill(np.random.rand()*2*np.pi)
        else:
            phase=np.zeros((round(T*fs)))
        sig += np.exp(-i*harmonic_decay/n_partials)*np.cos(2*np.pi*f0*(i+1)*np.sqrt(1+inharmonicity*i**2)*t + phase)
    fade_length /= 1000. # fade length in milliseconds
    envelope = np.ones(t.shape)
    fade_length_samples = round(fs*fade_length)
    envelope[:fade_length_samples] = np.hanning(fade_length_samples*2)[:fade_length_samples]
    envelope[-fade_length_samples:] = np.hanning(fade_length_samples*2)[-fade_length_samples:]
    sig = sig * envelope
    sig /= np.max(sig)
    return sig


def fm_generator(T=1, fs=44100, f_carrier=440, f_multiplier=2, fm_ratio = 1, fade_length=50, *args, **kwargs):
    t = np.linspace(0, T, round(T*fs))
    sig  = np.zeros(t.shape)
        
    phase = np.zeros((round(T*fs)))
    phase.fill(np.random.rand()*2*np.pi)
    f_modulator = f_carrier * f_multiplier
    mod_sig = fm_ratio * np.cos(2*np.pi*f_modulator*t + phase)
    phase = np.zeros((round(T*fs)))
    phase.fill(np.random.rand()*2*np.pi)
    sig = np.cos(2*np.pi*(f_carrier + mod_sig)*t + phase)
    
    fade_length /= 1000. # fade length in milliseconds
    envelope = np.ones(t.shape)
    fade_length_samples = round(fs*fade_length)
    envelope[:fade_length_samples] = np.hanning(fade_length_samples*2)[:fade_length_samples]
    envelope[-fade_length_samples:] = np.hanning(fade_length_samples*2)[-fade_length_samples:]
    sig = sig * envelope
    if np.max(sig)==0.0:
        raise SynthesisError(fm_generator, (f_carrier, f_multiplier, fm_ratio))
    sig /= np.max(sig)
    return sig


def noise_generator(T=1, fs=44100, fc=100, fb=50, fe=200, gpass=3, gstop=40, fade_length=100):

    def get_w(f, fs):
        return (f / (fs/2))
    t = np.linspace(0, T, round(T*fs))
    sig = 1 - 2*np.random.ranf(t.shape)

    wp = [fc - fb/2, fc + fb/2]
    ws = [wp[0]-fe, wp[1]+fe]
    wp = [get_w(w,fs) for w in wp]
    ws = [get_w(w,fs) for w in ws]

    filter_type = 'band'
    if ws[0] <= 0 or wp[0] <= 0:
        filter_type = 'lowpass'
        wp = wp[1]; ws = ws[1]
    elif ws[1] >= 1 or wp[1] >= 1:
        wp = wp[0]; ws = ws[0]
        filter_type = 'highpass'

    print(wp, ws)
    N, wn = buttord(wp, ws, gpass, gstop, fs=fs)
    print(N,wn,filter_type)
    b, a = butter(N, wn, filter_type)
    zi = lfilter_zi(b,a)
    sig_filt,_ = lfilter(b,a,sig,zi=zi*sig[0])

    fade_length /= 1000. # fade length in milliseconds
    envelope = np.ones(t.shape)
    fade_length_samples = round(fs*fade_length)
    envelope[:fade_length_samples] = np.hanning(fade_length_samples*2)[:fade_length_samples]
    envelope[-fade_length_samples:] = np.hanning(fade_length_samples*2)[-fade_length_samples:]
    sig = sig * envelope
    print('max : ', np.amax(np.abs(sig)))
    if np.max(sig)==0.0:
        raise SynthesisError(noise_generator, (fs, fb, fe, gpass, gstop, fade_length))
    sig_filt /= np.max(sig_filt)
    return sig_filt

    

    


def dataset_from_generator(T, fs, generator, export=None, name=None, *args, **kwargs):
    generator_args = kwargs
    axis = [];  parameter_name = []; param_buffer = dict(); metadata=dict();
       
    if export!=None:
        import os, os.path
        if not os.path.isdir(export):
            os.makedirs(export)
        if not os.path.isdir(export+'/data/wav'):
            os.makedirs(export+'/data/wav')
        if not os.path.isdir(export+'/metadata'):
            os.makedirs(export+'/metadata')
        if os.path.isfile(export+'/metadata/inputs.txt'):
            os.remove(export+'/metadata/inputs.txt')
        
        if name == None:
            name = export.split('/')[-1]
        files = []
        
    for v, i in generator_args.items():
        if i == []:
            raise Exception("Parameter %s seems to be empty"%v)
        parameter_name.append(v)
        metadata[v]=[]
        param_buffer[v] = 0
        if export!=None:
            if not os.path.isdir(export+'/metadata/'+v):
                os.makedirs(export+'/metadata/'+v)
            if not os.path.isdir(export+'/metadata/'+v+'/tracks'):
                os.makedirs(export+'/metadata/'+v+'/tracks')
            if os.path.isfile(export+'/metadata/'+v+'/metadata.txt'):
                os.remove(export+'/metadata/'+v+'/metadata.txt')
        with open(export+'/metadata/'+v+'/callback.txt', 'w+') as f:
            f.write('importRawFloat')
        axis.append(i)
        
    axis = tuple(axis)
    meshes = np.meshgrid(*axis)
    iterator = np.nditer(meshes[0], flags=['multi_index'])
    data = np.zeros((meshes[0].size, round(T*fs)))
    running_id = 0    
    n_examples = meshes[0].size
    parameters = []
    for i in iterator:
        for a in range(meshes[0].ndim):
            param_buffer[parameter_name[a]] = meshes[a][iterator.multi_index]
            metadata[parameter_name[a]].append(meshes[a][iterator.multi_index])
            if export != None:
                with open('%s/metadata/%s/tracks/%s_%i.txt'%(export, parameter_name[a], name, running_id), 'w+') as f:
                    f.write("%f"%meshes[a][iterator.multi_index])
                with open('%s/metadata/%s/metadata.txt'%(export, parameter_name[a]), 'a+') as f:
                    f.write('data/wav/%s_%i.wav\t%f\n'%(name, running_id, meshes[a][iterator.multi_index]))
        
        try:
            data[running_id, :] = generator(T=T, fs=fs, **param_buffer)
            parameters.append({**param_buffer, "rate":fs})
        except SynthesisError as e:
            print(e)
        
        if export!= None:
            filename = '%s/data/wav/%s_%i.wav'%(export, name, running_id)
            print(filename)
            torchaudio.save(filename, torch.from_numpy(data[running_id, :]), fs); files.append(filename)
            with open('%s/metadata/inputs.txt'%export, 'a+') as f:
                f.write('data/wav/%s_%i.wav\n'%(name, running_id))
            
            
        running_id += 1
        
    if export != None:

        dataset = DatasetAudio(export)
        dataset.data = data
        dataset.data_properties = parameters
        dataset.metadata = metadata
        dataset.files = files

    return dataset, data

