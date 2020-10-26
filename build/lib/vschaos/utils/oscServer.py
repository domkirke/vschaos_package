import asyncio
from time import time

from pythonosc import udp_client
from pythonosc import dispatcher
from pythonosc import osc_server

import torch, numpy as np, torchaudio as ta
from scipy.signal import resample, resample_poly
from ..data import Dataset, DatasetAudio, ComposeTransform, Magnitude, Polar, PolarInst
from ..monitor import INVERTIBLE_DIMREDS
from . import checklist, checktuple, SerieDeformation
from ..monitor import visualize_dimred as dr
import dill, random, string

from ..vaes import *

SESSION_ID = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(10))
SESSION_HISTORY_LENGTH = 10
MAX_TRAJECTORY_LENGTH = 65

# phase reconstruction
phase_rand = {}


def random_static(magnitude):
    if not magnitude.shape[0] in phase_rand.keys():
        phase_rand[magnitude.shape[0]] = np.random.random(magnitude.shape[0]) * 2 * np.pi
    return phase_rand[magnitude.shape[0]]


def osc_attr(obj, attribute):
    def closure(*args):
        args = args[1:]
        if len(args) == 0:
            return getattr(obj, attribute)
        else:
            return setattr(obj, attribute, *args)

    return closure


class OSCServer(object):
    '''
    Key class for OSCServers linking Python and Max / MSP

    Example :
    >>> server = OSCServer(1234, 1235) # Creating server
    >>> server.run() # Running server

    '''
    # attributes automatically bounded to OSC ports
    osc_attributes = []

    # Initialization method
    def __init__(self, in_port, out_port, ip='127.0.0.1', verbose=True, *args):
        super(OSCServer, self).__init__()
        # OSC library objects
        self.verbose = verbose
        self.dispatcher = dispatcher.Dispatcher()
        self.client = udp_client.SimpleUDPClient(ip, out_port)

        self.init_bindings(self.osc_attributes)
        self.server = osc_server.BlockingOSCUDPServer((ip, in_port), self.dispatcher)

        self.in_port = in_port
        self.out_port = out_port
        self.ip = ip

    def init_bindings(self, osc_attributes=[]):
        '''Here we define every OSC callbacks'''
        self.dispatcher.map("/ping", self.ping)
        self.dispatcher.map("/stop", self.stop_server)
        for attribute in osc_attributes:
            print(attribute)
            self.dispatcher.map("/%s" % attribute, osc_attr(self, attribute))

    def stop_server(self, *args):
        '''stops the server'''
        self.client.send_message("/terminated", "bang")
        self.server.shutdown()
        self.server.socket.close()

    def run(self):
        '''runs the SoMax server'''
        self.server.serve_forever()

    def ping(self, *args):
        '''just to test the server'''
        print("ping", args)
        self.client.send_message("/fromServer", "pong")

    def send(self, address, content):
        '''global method to send a message'''
        self.client.send_message(address, content)

    def send_bach_series(self, message, dim, series):
        out = ['series']
        if (message == '/encode'):
            out.append(dim + 1)
            str_out = "(" + str(dim + 1)
        else:
            str_out = "(" + dim
        for s2 in range(series.shape[0]):
            str_out += ' ( ' + str(float(s2) / series.shape[0]) + ' ' + str(series[s2]) + ' 0. )'
        str_out += ' )'
        out.append(str_out)
        # print(str_out)
        self.send(message, out)

    def print(self, *args):
        if self.verbose:
            print(*args)
            self.send('/print', *args)


def selector_all(dataset):
    return np.arange(dataset.data.shape[0])


def selector_n(dataset, n_points):
    return np.random.permutation(dataset.data.shape[0])[:int(n_points)]


# OSC decorator
def osc_parse(func):
    '''decorates a python function to automatically transform args and kwargs coming from Max'''

    def func_embedding(address, *args):
        t_args = tuple();
        kwargs = {}
        for a in args:
            if issubclass(type(a), str):
                if "=" in a:
                    key, value = a.split("=")
                    kwargs[key] = value
                else:
                    t_args = t_args + (a,)
            else:
                t_args = t_args + (a,)
        return func(*t_args, **kwargs)

    return func_embedding


def max_format(v):
    '''Format some Python native types for Max'''
    if issubclass(type(v), (list, tuple)):
        if len(v) == 0:
            return ' "" '
        return ''.join(['%s ' % (i) for i in v])
    else:
        return v


def dict2str(dic):
    '''Convert a python dict to a Max message filling a dict object'''
    str = ''
    for k, v in dic.items():
        str += ', set %s %s' % (k, max_format(v))
    return str[2:]


class VAEServer(OSCServer):
    osc_attributes = ['model', 'projection', 'current_layer', 'dataset']
    min_threshold = 0

    # model attributes
    def getmodel(self):
        return self._model

    def delmodel(self):
        del self._model

    def setmodel(self, *args):
        self._model_path = None
        try:
            if len(args) == 1:
                if args[0] is None:
                    return
                if type(args[0]) == str:
                    self.print('Loading model %s...' % args[0])
                    loaded_data = torch.load(args[0], map_location='cpu')

                    self._model = loaded_data['class'].load(loaded_data)
                    self._transforms = loaded_data.get('full_transforms')
                    self._model_path = args[0]
                    # if 'preprocessing' in loaded_data.keys():
                    #     self.preprocessing = loaded_data['preprocessing']
            elif issubclass(type(args[0]), AbstractVAE):
                self._model = args[0]
            self.print('model loaded')
            self.send('/model_loaded', 'bang')
            self.current_projection = None
            self.get_state()
            self.init_interpolation()
            self.init_conditioning()
            if self._model:
                self._model.eval()
            self.print('Server ready')
        except Exception as e:
            self.print('Loading failed')
            print('-- Loading model failed')
            print(e)

    model = property(getmodel, setmodel, delmodel, "vae model attached to server")

    # projection attributes
    def getprojection(self):
        return self.model.manifolds.get(self.current_projection)

    def delprojection(self):
        if self.current_projection is None:
            raise Exception('[error] tried to delete current projection but is actually empty')
        del self.model.manifolds[self.current_projection]
        self.current_projection = None

    def setprojection(self, *args):
        if args[0] == "None":
            self.current_projection = None
        else:
            assert type(args[0]) == str and args[0] in self.model.manifolds.keys(), "projection %s not found" % args[0]
            self.current_projection = str(args[0])
        self.print('projection set to %s' % self.current_projection)
        self.get_state()

    projection = property(getprojection, setprojection, delprojection)


    def getdataset(self):
        return self._dataset
    def setdataset(self, dataset):
        if dataset is None:
            return
        if issubclass(type(dataset), Dataset):
            self._dataset = dataset
        elif isinstance(dataset, str):
            self.print('Loading dataset %s...' % dataset)
            if dataset[-1] == "/":
                dataset = dataset[:-1]
            if self._model is None:
                self.print('Load model before datasets')
            self._dataset = DatasetAudio(dataset, transforms=self._transforms)
            self._dataset.import_data(scale=False)
            self.send('/dataset_loaded', 'bang')
            self.print('Dataset loaded')
    def deldataset(self):
        del self._dataset
        self._dataset = None
    dataset = property(getdataset, setdataset, deldataset)


    def getlayer(self):
        return self._current_layer

    def dellayer(self):
        raise Exception("attribute current_layer cannot be deleted")

    def setlayer(self, *args):
        try:
            self._current_layer = int(args[0])
        except TypeError:
            print('current_layer attribute must be parsed as int')

    current_layer = property(getlayer, setlayer, dellayer)

    def __init__(self, *args, **kwargs):
        self._model = None
        self.model = kwargs.get('model')
        self.model_path = None
        self._dataset = None
        self.projections = {}
        self.current_projection = None
        self.current_gen_id = 0
        self.deformer = SerieDeformation()
        self.cd_deformer = {}

        self.condition_params = None
        self.current_phase = None
        self.phase_reconstruction = None

        self.dataset = kwargs.get('dataset')
        self.sr = 44100

        # navigation option
        self._current_layer = 0
        self._latent_noise_amount = 0.
        self.current_traj = None
        self.phase_callback = random_static
        self.current_conditioning = {}
        self._selector = selector_all(self.dataset) if self.dataset is not None else None

        # Sets of latent points for interpolation
        self.init_interpolation()
        self.interpolate_latent = 1
        super(VAEServer, self).__init__(*args)
        self.get_state()
        self.print('Server ready.')
        self.send('/reduction_types', [str(d.__name__) for d in INVERTIBLE_DIMREDS])
        self.original_shape = None

    def init_bindings(self, osc_attributes=[]):
        super(VAEServer, self).init_bindings(self.osc_attributes)

        # Offline methods
        self.dispatcher.map('/get_state', osc_parse(self.get_state))
        self.dispatcher.map('/resend_state', osc_parse(self.resend_state))
        # Change option flags
        # self.dispatcher.map('/set_freeze_mode', osc_parse(self.set_freeze_mode))
        # self.dispatcher.map('/set_scaling', osc_parse(self.set_scaling))
        # self.dispatcher.map('/set_path_use_z', osc_parse(self.set_path_use_z))
        # # Changing model and dataset
        # self.dispatcher.map('/select_model', osc_parse(self.select_model))
        # self.dispatcher.map('/set_model', osc_parse(self.set_model))
        # self.dispatcher.map('/set_dataset', osc_parse(self.set_dataset))
        # # Latent path operations
        # self.dispatcher.map('/clear_path', osc_parse(self.clear_path))
        # self.dispatcher.map('/add_free_path', osc_parse(self.add_free_path))
        # self.dispatcher.map('/add_preset_path', osc_parse(self.add_preset_path))
        # self.dispatcher.map('/play_path', osc_parse(self.play_path))
        # # Core functionalities (load, decode, encode)
        self.dispatcher.map('/condition', osc_parse(self.condition))
        self.dispatcher.map('/encode', osc_parse(self.encode))
        self.dispatcher.map('/decode', osc_parse(self.decode))
        self.dispatcher.map('/interpolate', osc_parse(self.interpolate))
        self.dispatcher.map('/set_interpolation_mode', osc_parse(self.set_interpolation_mode))
        self.dispatcher.map('/latent_sample', osc_parse(self.set_latent_noise))
        # Online methods

        self.dispatcher.map('/encode_spectrogram', osc_parse(self.encode_spectrogram))
        self.dispatcher.map('/decode_spectrogram', osc_parse(self.decode_spectrogram))
        self.dispatcher.map('/get_state', osc_parse(self.get_state))

        self.dispatcher.map('/get_spectrogram', osc_parse(self.get_spectrogram))
        self.dispatcher.map('/add_projection', osc_parse(self.add_projection))
        self.dispatcher.map('/load', osc_parse(self.load_projections))
        self.dispatcher.map('/save', osc_parse(self.save))
        self.dispatcher.map('/get_projections', osc_parse(self.get_projections))
        self.dispatcher.map('/add_anchor', osc_parse(self.add_anchor))
        self.dispatcher.map('/load_anchor', osc_parse(self.load_anchor))

    def resend_state(self):
        self.get_state()
        self.print('Server is ready.')

    def set_latent_noise(self, *args):
        self._latent_noise_amount = float(args[1])

    def init_conditioning(self):
        _, labels = self.get_condition_list()
        for k, v in labels.items():
            self.cd_deformer[k] = SerieDeformation()

    def get_condition_list(self):
        if self.model is None:
            return "None"
        hidden_params = []
        for ph in checklist(self.model.phidden):
            if ph.get('encoder') or ph.get('decoder'):
                if ph.get('encoder'):
                    hidden_params.append(ph['encoder'])
                if ph.get('decoder'):
                    hidden_params.append(ph['decoder'])
            else:
                hidden_params.extend(ph)

        labels = {}
        for ph in hidden_params:
            if ph.get('label_params') is not None:
                labels = {**labels, **ph['label_params']}

        conds = [[k, 'intlist', 0, v['dim']] for k, v in labels.items()]

        return conds, labels

    def get_state(self):
        conditions_list = "None";
        conditions_names = "None";
        latent_dim = 0;
        n_layers = 0
        if self.model is None:
            state = {}
        else:
            input_dim = self.model.pinput['dim']
            if self.current_projection is not None:
                latent_dim = self.projection.dim
            elif self.model is not None:
                current_layer = self.model.platent[self.current_layer]
                if isinstance(current_layer, list):
                    self.print('[Warning] model seems to have multiple latent variables at layer %d'%self.current_layer)
                    current_layer = self.model.platent[self.current_layer][0]
                latent_dim = current_layer['dim']

            conditions_list, self.condition_params = self.get_condition_list()
            conditions_names = [c[0] for c in conditions_list] if len(conditions_list) > 0 else ["None"]
            conditions_list = sum(conditions_list, [])
            n_layers = len(self.model.platent)
            state = {'current_projection': str(self.current_projection),
                     'projections': list(self.model.manifolds.keys()),
                     'conditioning': conditions_names,
                     'input_dim': input_dim, 'latent_dim': latent_dim}

            state['anchors'] = list(self.model.manifolds[self.current_projection].anchors.keys())

        state_str = dict2str(state)
        if len(state_str) != 0:
            self.send('/state', state_str)

        self.send('/conditions_list', conditions_list if len(conditions_list) > 0 else "None")
        self.send('/conditions_names', conditions_names)
        self.send('/latent_dims', latent_dim)
        self.send('/num_layers', n_layers)
        self.send('/interpolation_list', list(self.interpolate_points.keys()))
        # self.get_projections()
        return state

    def encode(self, *args, transforms=None):
        try:
            self.print('Computing transform...')
            transforms = transforms or self._transforms
            if type(args[0]) == str:
                file_path = args[0]
                x, sr = ta.load(file_path)
                transform = transforms(x)
            elif type(args[0]) == np.ndarray or torch.is_tensor(args[0]):
                transform = transforms(args[0])

            y = {}
            if self.condition_params != {}:
                for k, t in self.condition_params.items():
                    if self.current_conditioning.get(k) is not None:
                        if len(self.current_conditioning[k]) > 2:
                            y[k] = torch.from_numpy(
                                resample(np.array(self.current_conditioning[k]), transform.shape[0]))
                        else:
                            y[k] = torch.from_numpy(self.current_conditioning[k].repeat(transform.shape[0]))
                        # y[k] = self.cd_deformer[k].add_original(self.current_conditioning[k], MAX_TRAJECTORY_LENGTH)
                    else:
                        y[k] = torch.zeros(transform.shape[0]).int()
            #
            # self.print('Encoding...')
            self.print('Encoding...')
            with torch.no_grad():
                vae_out = self.model.encode(self.model.format_input_data(transform), y=y)
            latent = vae_out[self.current_layer]['out_params']
            if isinstance(latent, list):
                latent = latent[0]
            latent_std = latent.stddev
            latent = latent.mean


            if self.projection is not None:
                latent = self.projection(latent)
            self.original_shape = latent.shape

            if (len(latent.shape) == 1):
                self.send('/encode', ['point', latent])
            else:
                # PAD LATENT SERIE SO THAT IT'S A POWER OF 2
                latent = latent.T;
                target_dim = int(2 ** np.ceil(np.log2(latent.shape[1])))

                self.current_traj = latent
                self.current_traj_std = latent_std
                latent_rs = np.pad(latent, ((0, 0), (0, target_dim - latent.shape[1])), mode="constant")
                latent_rs = self.deformer.add_original(latent_rs, MAX_TRAJECTORY_LENGTH)
                for s in range(latent.shape[0]):
                    self.send_bach_series('/encode', s, latent_rs[s])
            if latent.shape[1] < MAX_TRAJECTORY_LENGTH:
                self.send('/latent', latent.squeeze().tolist())
            self.print('Server ready')
        except Exception as e:
            self.print('Encoding failed.')
            print('--Encoding failed.')
            print(e)

    def encode_spectrogram(self, *args):
        if self.model is None:
            return
        input = torch.Tensor(args).unsqueeze(0)
        if input.shape[1] < self.model.pinput['dim']:
            input = torch.cat([input, torch.zeros(1, self.model.pinput['dim'] - input.shape[1])], 1)
        transform = self._get_post_spectral_transform(self._transforms)
        return self.encode(input, transforms=transform)

    def decode(self, *args, write=True, send_spectrogram=False, transforms=None):
        if args[0] == "series":
            input_args = args[1].split(' ')
            dim_id = int(input_args[0]) - 1;
            dim_traj = np.array([float(a) for a in input_args[1:]])
            self.current_traj = self.deformer.deform(dim_id, dim_traj)

        if args[0] == "point":
            latent_dim = self.model.platent[self.current_layer]['dim']
            self.current_traj = np.zeros((latent_dim, 1))
            if len(args) - 1 >= latent_dim:
                self.current_traj[:, 0] = np.array(args[1:latent_dim + 1])
            else:
                self.current_traj[:len(args), 0] = np.array(args[1:])

        if args[0] in ["generate", "point"]:
            try:
                self.print('Decoding...')

                z = self.current_traj
                if self.projection is not None:
                    z = self.projection.invert(z)
                z = z.T
                z = self.model.format_input_data(z)
                if self.original_shape is not None:
                    z = z[:self.original_shape[0]]
                    if z.shape == self.current_traj_std.shape:
                        z = z + self.current_traj_std * self._latent_noise_amount * torch.randn_like(z)

                y = {}
                if self.condition_params != {}:
                    for k, t in self.condition_params.items():
                        if self.current_conditioning.get(k) is not None:
                            if len(self.current_conditioning[k]) == 1:
                                y[k] = torch.from_numpy(self.current_conditioning[k].repeat(z.shape[0]))
                            elif self.current_conditioning[k].shape[0] != z.shape[0]:
                                y[k] = self.get_deformed_serie(self.current_conditioning[k], target_length=z.shape[0],
                                                               axis=0, mode="constant").int()
                            else:
                                y[k] = torch.from_numpy(self.current_conditioning[k]).int()
                        else:
                            y[k] = torch.zeros(z.shape[:-1]).int()


                with torch.no_grad():
                    vae_out = self.model.decode(z, y=y)
                reco = vae_out[self.current_layer]['out_params'].mean.cpu()

                self.print('Inverting...')
                transforms = transforms or self._transforms
                raw_data = transforms.invert(reco)
                # if self.model.pinput['conv']:
                #     reco = reco.squeeze(-len(checktuple(self.model.pinput['dim'])) - 1)

                if write:
                    # assert self.transformOptions
                    # if self.transformOptions:
                        # raw_data = inverseTransform(reco.T, 'stft', {'transformParameters': self.transformOptions},
                        #                             iterations=30)
                    name = '/tmp/%s_%d.wav' % (SESSION_ID, self.current_gen_id);
                    self.current_gen_id = (self.current_gen_id + 1) % SESSION_HISTORY_LENGTH
                    if raw_data.abs().max() > 1.0:
                        raw_data /= raw_data.abs().max()
                    ta.save(name, raw_data.clamp(-1, 1), self.sr)
                    self.send('/decode', name)

                if send_spectrogram:
                    self.send('/spectrogram_out', reco.squeeze()[:1025].tolist())

                self.print('Server ready.')
            except Exception as e:
                self.print('Decoding failed.')
                # print('--Decoding failed.')
                print(e)

    def decode_spectrogram(self, *args):
        transform = self._get_post_spectral_transform(self._transforms)
        return self.decode(*args, write=False, send_spectrogram=True, transforms=transform)

    # Interpolation methods

    def init_interpolation(self):
        # Reset dictionnary
        self.interpolate_points = {}
        self.interpolate_conditions = {}

    def scale_interpolation(self):
        for k, v in self.interpolate_points.items():
            new_v = np.zeros((self._model.latent_dims, self.n_points * self.scaling))
            for d in range(v.shape[0]):
                new_v[d] = resample_poly(v[d], self.scaling * self.n_points, len(v[d]))
            self.interpolate_points[k] = new_v

    def get_deformed_serie(self, serie, target_serie=None, target_length=None, mode="smooth", axis=0):
        if target_serie is not None:
            target_length = target_serie.shape[1]
        if target_length is None:
            if self.current_traj is not None:
                target_length = self.current_traj.shape[1]
            else:
                return serie
        if torch.is_tensor(serie):
            serie = serie.numpy()
        if len(serie) > 2:
            if mode == "smooth":
                serie = resample(np.array(serie), target_length, axis=axis)
            elif mode == "constant":
                original_shape = serie.shape[0]
                new_serie = np.zeros(target_length, dtype=serie.dtype)
                steps = target_length // original_shape
                for i in range(original_shape):
                    new_serie[i*steps:(i+1)*steps] = serie[i]
                tail = target_length - steps * original_shape
                new_serie[-tail:] = serie[-1]
                serie =new_serie

            else:
                raise ValueError('deformation mode %s is not known'%mode)
        else:
            serie = np.array(serie).repeat(target_length, axis=axis)

        return torch.from_numpy(serie)

    def interpolate(self, val_list):
        print(val_list)
        self.print('Interpolating')
        interp = val_list.split(' ')
        if (interp[0] == 'current'):
            print('Adding current point as ' + interp[1])
            self.interpolate_points[interp[1]] = self.current_traj
            if self.current_conditioning is not None:
                self.interpolate_conditions[interp[1]] = dict(self.current_conditioning)
            print("Added !")
            return
        if (interp[0] == 'random'):
            print('Adding random point as ' + interp[1])
            self.interpolate_points[interp[1]] = np.random.randn(self._model.platent[self.current_layer]['dim'],
                                                                 self.current_traj[1])
            return

        interp_names = interp[::2]
        interp_vals = [float(x) for x in interp[1::2]]
        if (self.current_conditioning is not None):
            for k, v in self.current_conditioning.items():
                target_shape = np.max([x[k].shape[0] for x in self.interpolate_conditions.values()])
                z = np.zeros((target_shape))
                for n, v in zip(interp_names, interp_vals):
                    if n not in self.interpolate_conditions.keys():
                        continue
                    z += self.get_deformed_serie(self.interpolate_conditions[n][k], target_length=target_shape,
                                                 axis=0).numpy() * v
                self.condition(k, z)
                cond_ds = self.cd_deformer[k].add_original(z, target_shape)
                self.send_bach_series('/condition', k, cond_ds.astype('int'))

        valid_points = list(
            filter(lambda x: interp_names[x] in self.interpolate_points.keys(), range(len(interp_names))))
        interp_names = [interp_names[i] for i in valid_points]
        interp_vals = [interp_vals[i] for i in valid_points]
        if (self.interpolate_latent) and len(interp_names) > 0:
            # Create the empty point
            target_shape = np.max([x.shape[1] for x in self.interpolate_points.values()])
            z = np.zeros((self.model.platent[self.current_layer]['dim'], target_shape))

            for n, v in zip(interp_names, interp_vals):
                if not n in self.interpolate_points.keys():
                    continue
                z += self.get_deformed_serie(self.interpolate_points[n], target_length=target_shape, axis=1).numpy() * v

            # resamp = SerieDeformation().add_original(z, 65)
            # Compute decoding

            traj_ds = self.deformer.add_original(z, 65)
            for s in range(z.shape[0]):
                self.send_bach_series('/encode', s, traj_ds[s])

            self.current_traj = z
            result = self.decode('generate')

            # Write result to file
            # path = self.output_wav(result)
            # Send the path to interface
            # self.send('/decode', path)

    def set_interpolation_mode(self, version, value):
        if (version == 'latent'):
            self.interpolate_latent = value
        if (version == 'condition'):
            self.interpolate_condition = value

    def condition(self, *args):
        cond_type = str(args[0])
        if issubclass(type(args[1]), str):
            conditioning = args[1] if " " in args else args[1].split(' ')
            cond_series = np.array([int(a) for a in conditioning])
        elif issubclass(type(args[1]), np.ndarray):
            cond_series = args[1]
        else:
            try:
                cond_series = np.array(args[1:])
            except Exception as e:
                print('could not parse condition %s because of : %s' % (args[1:], e))
        self.current_conditioning[cond_type] = cond_series
        print("current conditioning : ", cond_series)


    def _get_post_spectral_transform(self, transforms):
        if isinstance(transforms, ComposeTransform):
            transforms  = transforms.transforms
        index = None
        for i, t in enumerate(transforms):
            if type(t) in [Magnitude, Polar, PolarInst]:
                index = i
                break
        if index is None:
            return None
        transforms = transforms[index:]
        return ComposeTransform(transforms)


    def get_spectrogram(self, *args, projection=None, filter=True):
        if self._model is None:
            print('d[Error] spectrogram requested by not any model loaded')
            self.send('/vae_error', '[Error] spectrogram requested by not any model loaded')
            return

        if projection is None:
            projection = self.projection

        z_input = np.array(list(args))[np.newaxis, :]
        input_dim = self._model.platent[-1]['dim'] if projection is None else projection.dim

        if z_input.shape[1] > input_dim:
            z_input = z_input[:, :input_dim]
        elif z_input.shape[1] < input_dim:
            z_input = np.concatenate([z_input, np.zeros((z_input.shape[0], input_dim - z_input.shape[1]))], axis=1)

        if projection is not None:
            z_input = projection.invert(z_input)

        with torch.no_grad():
            self._model.eval()
            vae_out = self._model.decode(self._model.format_input_data(z_input))

        spec_out = vae_out[0]['out_params'].mean.squeeze().numpy()
        stft_transform = self._get_spectral_transform(self._transforms)
        spec_out = stft_transform.invert(spec_out)
        # phase_out = self.phase_callback(spec_out)

        # spec_out = np.zeros_like(spec_out)
        # id_range = np.random.randint(spec_out.shape[0])
        # spec_out[id_range] = 1

        if filter:
            spec_out[spec_out < self.min_threshold] = 0.
        spec_out = spec_out[1:1024].tolist()
        # phase_out = phase_out[1:].tolist()

        self.send('/current_z', z_input.squeeze().tolist())
        self.send('/spectrogram_mag_out', spec_out)
        # self.send('/spectrogram_phase_out', phase_out)

    def add_projection(self, proj_name, dimred, save=True, **kwargs):
        if self.dataset is None:
            self.print('[Error] a dataset is needed to compute projections')

        proj_name = str(proj_name);
        dimred = int(dimred)
        if self._model is None:
            self.print('Please load a model first!')

        if self.dataset is None:
            self.print('Online dimensionality reduction methods require a -d option')

        if proj_name == "none":
            self.print('projection cannot be named none')

        dimred_method = getattr(dr, proj_name)
        if dimred_method is None:
            self.print('projection %s not found' % dimred_method)
        self.print('computing projection %s...' % proj_name)
        options = {'layer': self.current_layer, 'n_points':200, 'n_components':dimred}
        self._model.add_manifold(proj_name, self.dataset, dimred_method, **options)

        self.print('projection %s created' % (proj_name))
        self.get_state()
        self.get_projections()
        self.print('server ready!')

    def get_projections(self):
        if self._model is not None:
            projections = list(self.model.manifolds.keys())
            projections_str = [str(p) for p in projections]
            self.send('/projections', projections_str)

    def save(self, path):
        if self._model is None:
            self.print('load a model first!')
            return
        model_manifolds = self._model.manifolds
        torch.save(model_manifolds, path)

    def add_anchor(self, name, *args):
        self.model.manifolds[self.current_projection].add_anchor(name, *args)
        self.get_state()

    def load_anchor(self, name, *args):
        anchor = self.model.manifolds[self.current_projection].anchors[name]
        # self.get_spectrogram(*tuple(np.array(anchor).tolist()))
        self.send('/anchor', anchor)
        self.print('anchor %s loaded' % name)

        # with open(path+'.aego', 'wb') as f:
        #     dill.dump(self._projections, f)
        # self.print('projections saved at %s.aego'%path)

    def load_projections(self, path):
        with open(path, 'rb') as f:
            loaded_projections = torch.load(f)
        for k, v in loaded_projections.items():
            if k in self._projections.keys():
                self.print('[Warning] projection %s replaced' % k)
        self.print('projections loaded from %s' % path)
        self.model.manifolds = loaded_projections
        self.get_state()




