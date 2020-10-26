import torch, torchaudio
from tqdm import tqdm
from .data_generic import Dataset, checklist
from .data_transforms import ComposeAudioTransform
from .data_asynchronous import OfflineDataList
from . import data_utils



class DatasetAudio(Dataset):
    valid_types = ['wav', 'mp3', 'flac', 'ogg', 'aif', 'aiff']
    compose_function = ComposeAudioTransform

    def __repr__(self):
        return f"DatasetAudio(data_prefix={self.data_prefix})"

    def __init__(self, root, drop_time=None, time_transform={'time':None, 'sequence':None}, **options):
        super().__init__(root, **options)
        self.drop_time = drop_time
        self.time_transform = time_transform

    def __getitem__(self, item, drop_time=None, drop_sequence=None, preprocess=True, preprocess_time=True):
        drop_time = drop_time if drop_time is not None else self.drop_time
        drop_sequence = drop_sequence if drop_sequence is not None else self.drop_sequence
        if drop_time and drop_sequence in ["data", "both"]:
            (data, time, sequence), metadata = super(DatasetAudio, self).__getitem__(item, preprocess=True, drop_sequence=drop_sequence, drop_time=drop_time)
        elif drop_time:
            (data, time), metadata = super(DatasetAudio, self).__getitem__(item, preprocess=True, drop_sequence=drop_sequence, drop_time=drop_time)
        elif drop_sequence in ['data', 'both']:
            (data, sequence), metadata = super(DatasetAudio, self).__getitem__(item, preprocess=True, drop_sequence=drop_sequence, drop_time=drop_time)
        else:
            data, metadata = super(DatasetAudio, self).__getitem__(item, preprocess=True, drop_sequence=drop_sequence, drop_time=drop_time)
        if drop_sequence in ['data', 'both']:
            data = checklist(data)+[sequence]
        if self.time_transform.get('time') is not None and preprocess_time:
            time = self.time_transform['time'](time)
        if drop_time in ["data", "both"]:
            data = checklist(data)+[time]
        if drop_time in ['meta', 'both']:
            metadata['time'] = time

        return data, metadata

    def get_data(self, item, preprocess=True, drop_time=False, batch_first=False, **kwargs):
        if hasattr(item, "__iter__") or isinstance(item, slice):
            if isinstance(item, slice):
                data = self.data[item]
                props = self.data_properties[item]
            else:
                data = [self.data[int(i)] for i in item]
                props = [self.data_properties[int(i)] for i in item]

            if drop_time:
                if preprocess:
                    data_and_time = [self.transforms(data[i], time=props[i]['time'], sr=props[i]['rate']) for i in range(len(data))]
                else:
                    data_and_time = [(data[i], props[i]['time'].unsqueeze(0)) for i in range(len(data))]
                data = data_utils.dyn_collate([d[0] for d in data_and_time])
                time = data_utils.dyn_collate([d[1] for d in data_and_time])
                return data, time
            else:
                if preprocess:
                    data = data_utils.dyn_collate([self.transforms(data[i]) for i in range(len(data))])
                return data
        else:
            data = self.data[item]
            props = self.data_properties[item]
            if drop_time:
                if preprocess:
                    data, time = self.transforms(data, time=props['time'], sr=props['rate'])
                else:
                    data, time = data, props['time'].unsqueeze(0)
                return data, time
            else:
                if preprocess:
                    data = self.transforms(data)
                return data


    @property
    def options(self):
        return {**super(DatasetAudio, self).options, "resampleTo":self.sample_rate}

    @property
    def sample_rate(self):
        if hasattr(self, "resampleTo"):
            return self.resampleTo
        else:
            return None

    def _update_options(self, options):
        self._sample_rate = options.get('resampleTo')


    def import_callback(self, f, options={}):
        info, _ = torchaudio.info(f)
        properties = {'rate':info.rate, 'precision':info.precision}
        audio, _ = torchaudio.load(f)
        if options.get('resampleTo'):
            if properties['rate'] != options.get('resampleTo'):
                audio = torchaudio.transforms.Resample(info.rate, float(options.get('resampleTo')))(audio)
                properties['rate'] = options.get('resampleTo')
        properties['time'] = torch.zeros(audio.size(0))
        return audio, properties


    def apply_transforms(self, transform=None):
        transform = transform or self.transforms
        if isinstance(self.data, OfflineDataList):
            data = [None]*len(self.data)
        else:
            data = self.data
        for i in tqdm(range(len(self.data)), total=len(self.data), desc="applying transforms..."):
            data[i], self.data_properties[i]['time'] = transform(self.data[i], time=self.data_properties[i]['time'], sr=self.data_properties[i]['rate'])
        self.data = data
        self.transforms = self.compose_function()


    def scale_transforms(self, scale, transforms=None):
        if not scale:
            return
        if transforms is None:
            transforms = self.transforms
        if scale is True:
            data = self.data[:]
            ids = slice(None)
        elif isinstance(scale, int):
            ids = torch.randperm(len(self.data))[:scale]
            data = [self.data[i] for i in ids.tolist()]
        transforms.scale(data)
        drop_sequence = "meta" #if self.sequence_transform is not None else None
        drop_time = "meta" #if self.time_transform is not None else None
        _, metadata = self.__getitem__(ids, drop_time=drop_time, drop_sequence=drop_sequence, preprocess_time=None)
        if self.time_transform.get('time') is not None:
            self.time_transform['time'].scale(metadata['time'])
        if self.time_transform.get('sequence') is not None:
            self.time_transform['sequence'].scale(metadata['sequence'])


    def retrieve(self, idx):
        dataset = super().retrieve(idx)
        dataset.drop_time = self.drop_time
        return dataset

    def _scale_to_write(self, transforms, scale):
        if not scale:
            return
        if scale is True:
            data = self.__getitem__(slice(None, None, None), drop_time="none", drop_sequence="none")[0]
        elif isinstance(scale, int):
            ids = torch.randperm(len(self.data))[:scale]
            data = [self.__getitem__(int(i), drop_time="none", drop_sequence="none")[0] for i in ids]
        transforms.scale(data_utils.dyn_collate(data))

    def _get_data_to_write(self, item, transforms):
        (data, time), metadata = self.__getitem__(item, drop_time="data", drop_sequence="none")
        if transforms is not None:
            data, time = transforms(data, time=time, sr=self.data_properties[item]['rate'])
        return data, {'time':time}

