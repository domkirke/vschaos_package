import numpy as np, torch, torchvision, os, re, dill
from abc import abstractmethod
from . import checklist
from . import data_metadata as meta
from . import data_asynchronous as asyn
from .data_transforms import ComposeTransform
from . import data_utils
from tqdm import tqdm
from numbers import Number

class Dataset(torch.utils.data.Dataset):
    """
    data.Dataset object parses a dataset folder splitted in
    """
    valid_types = None # types valid for import
    compose_function = ComposeTransform
    def __repr__(self):
        return f"Dataset(data_prefix={self.data_prefix})"

    def __init__(self, root, **options):
        self.data_prefix = root
        self.input_file = options.get('input_file', None)
        self.data_directory = options.get("data_directory") or (self.data_prefix + '/data')
        self.analysis_directory = options.get("analysis_directory") or (self.data_prefix + '/analysis')
        self.metadata_directory = options.get("metadata_directory") or (self.data_prefix + '/metadata')
        self.verbose = options.get('verbose', False)
        # Tasks to import

        # metadata handling
        self.metadata = {}
        self.tasks = []
        self.metadata_callbacks = []
        self.classes = {}
        if options.get('tasks') is not None:
            self.tasks = options['tasks']
        else:
            self.import_metadata_tasks(options.get('sort_metadata', True))
        # Partitions in the dataset
        self.partitions = {}
        self.partitions_files = {}
        # Files tracking
        self.files = np.array([], dtype="object")
        self.hash = {}
        # data properties
        self.data = []
        self.data_properties = []
        self.import_callback = options.get('import_callback', type(self).import_callback)

        if options.get('list', True):
            if self.input_file is None:
                self.list_directory()
            else:
                if not os.path.isfile(f"{self.data_prefix}/{self.input_file}"):
                    self.list_directory()
                    self.write_inputs()
                else:
                    self.list_directory_from_file(self.input_file)

        transforms = options.get('transforms', [])
        if not isinstance(transforms, self.compose_function):
            transforms = self.compose_function(options.get('transforms', []))
        self.transforms = transforms

        self.target_transforms = {}
        target_transforms = options.get('target_transforms')
        if target_transforms is None:
            pass
        elif isinstance(target_transforms, dict):
            self.target_transforms = target_transforms
        else:
            if isinstance(target_transforms, (list, tuple)):
                target_transforms = self.compose_function(options.get('transforms', []))
            for t in self.tasks:
                self.target_transforms[t] = target_transforms

        self.drop_sequence = options.get('drop_sequence', None)
        self.sequence_dim = options.get('sequence_dim', 0)
        self.sequence_transform = options.get('sequence_transform', None)

    def __getitem__(self, item, preprocess=True, drop_sequence=None, **kwargs):
        """
        __getitem__ method of Dataset. Is used when combined with the native torch data loader.

        Parameters
        ----------
        options : dict
            native options:
                dataPrefix (str) : data roo<t of dataset
                dataDirectory (str) : audio directory of dataset (default: dataPrefix + '/data')
                analysisDirectory (str) : transform directory of dataset (default: dataPrefix + '/analysis')
                metadataDirectory (str) : metadata directory of dataset (default: dataPrefix + '/metadata')
                tasks [list(str)] : tasks loaded from the dataset
                taskCallback[list(callback)] : callback used to load metadata (defined in data_metadata)
                verbose (bool) : activates verbose mode
                forceUpdate (bool) : forces updates of imported transforms
                checkIntegrity (bool) : check integrity of  files

        Returns
        -------
        A new dataset object

        Example
        -------
        """
        # Retrieve data
        batch_first = hasattr(item, '__iter__') or isinstance(item, slice)
        data = self.get_data(item, preprocess=True, batch_first=batch_first, **kwargs)
        metadata = self.get_metadata(item, batch_first=batch_first, **kwargs)
        drop_sequence = drop_sequence or self.drop_sequence
        if drop_sequence is not None:
            sequence = self.get_sequence(data, batch_first=batch_first)
            if self.sequence_transform is not None:
                sequence = self.sequence_transform(sequence)
        if drop_sequence in ["data", "both"]:
            data = list(data)+[sequence]
        if drop_sequence in ['meta', 'both']:
            metadata['sequence'] = sequence
        return data, metadata

    def get_data(self, item, preprocess=True, batch_first=False, **kwargs):
        data = self.data[item]
        if isinstance(data, (list, tuple)) or isinstance(item, slice):
            data = data_utils.dyn_collate([self.transforms(data[i]) for i in range(len(data))])
        else:
            data = self.transforms(data)
        return data
        # batch_first = hasattr(item, '__iter__')
        # if self.drop_sequence is not None:
        #     sequence = self.get_sequence(data, batch_first)
        #     if self.sequence_transform is not None:
        #         sequence = self.sequence_transform(sequence)
        #     return data, sequence
        # else:
        #     return data

    def get_sequence(self, data, batch_first=False):
        if isinstance(data, tuple):
            data = data[0]
        sequence_dim = self.sequence_dim
        if batch_first:
            sequence_dim += 1
        if isinstance(data, list):
            data = data[0]
        shape = data.shape[:sequence_dim+1]
        sequence = torch.zeros(shape)
        for i in range(data.shape[sequence_dim]):
            sequence[..., i] = i
        sequence = sequence.int().unsqueeze(-1)
        return sequence


    def get_metadata(self, item, batch_first=False, **kwargs):
        metadata = {}
        for t in self.tasks:
            current_metadata = self.metadata[t][item]
            if self.target_transforms.get(t):
                current_metadata = self.target_transforms[t](current_metadata)
            metadata[t] = current_metadata
        return metadata

    def __len__(self):
        return len(self.data)

    def get_ids_from_class(self, meta_id, task, ids=None, exclusive=False):
        """return the data ids corresponding to a given class, for a given task.
        :param meta_id: desired class ids
        :type meta_id: int, list, np.ndarray
        :param str task: target task
        :param ids: (optional) constrains the search to the provided ids
        :type ids: np.ndarray or list(int)
        :param bool exclusive: only select data ids with every provided ids, in case of multi-label information (default: False)
        """
        current_metadata = self.metadata.get(task)
        if not hasattr(meta_id, '__iter__'):
            meta_id = checklist(meta_id)
        assert current_metadata is not None, "task %s not found" % task
        if ids is None:
            ids = range(len(current_metadata))
        valid_ids = list(filter(lambda x: current_metadata[x] is not None and x in ids, range(len(current_metadata))))
        meta_id = checklist(meta_id)
        ids = set(valid_ids) if exclusive else set()
        for m in meta_id:
            if exclusive:
                ids = set(filter(lambda x: current_metadata[x] == m or m in checklist(current_metadata[x]),
                                 valid_ids)).intersection(ids)
            else:
                ids = set(
                    filter(lambda x: current_metadata[x] == m or m in checklist(current_metadata[x]), valid_ids)).union(
                    ids)
        return np.array(list(ids))

    @property
    def options(self):
        return {'verbose':self.verbose, 'drop_sequence':self.drop_sequence}

    def retrieve(self, idx):
        """
        returns a sub-dataset from the actual one. If the main argument is a string, then returns the sub-dataset of the
        corresponding partition. Otherwise, it has to be a valid list of indices.
        :param idx: list of indices or partition
        :type idx: str, list(int), np.ndarray
        :return: the corresponding sub-dataset
        :rtype: :py:class:`Dataset`
        """


        if type(idx) == str:
            # retrieve the corresponding partition ids
            if not idx in self.partitions.keys():
                raise IndexError('%s is not a partition of current dataset' % idx)

            idx = self.partitions[idx]
            if type(idx[0]) == str or type(idx[0]) == np.str_:
                # if the partition are files, get ids from files
                idx = self.get_ids_from_files(idx)
        elif idx is None:
            if self.data != []:
                idx = np.arange(len(self))
            else:
                idx = np.arange(len(self.files))

        # create a new dataset
        newDataset = type(self)(self.data_prefix, list=False, **self.options)
        idx = np.array(idx)
        if len(self.data) > 0:
            if type(self.data) == list:
                # newDataset.data = np.array(self.data)[idx]
                # if self.has_sequences:
                newDataset.data = [self.data[i] for i in idx]
                # else:
                #    newDataset.data = [self.data[i][idx] for i in range(len(self.data))]
            elif type(self.data) == asyn.OfflineDataList:
                newDataset.data = self.data.take(idx)
            else:
                # pdb.set_trace()
                newDataset.data = self.data[idx]
            newDataset.data_properties = [self.data_properties[i] for i in idx]
        if self.metadata != {}:
            newDataset.metadata = {k: v[idx] for k, v in self.metadata.items()}
        if self.classes != {}:
            newDataset.classes = self.classes
        if len(self.files) != 0:
            newDataset.files = np.array(self.files)[idx].tolist()
        if self.hash != {}:
            new_files = np.array(list(set(newDataset.files)))
            new_hashes = {k:None for k in newDataset.files}
            for i, f in enumerate(newDataset.files):
                if new_hashes[f] is None:
                    new_hashes[f] = i
                elif isinstance(new_hashes[f], int):
                    new_hashes[f] = [new_hashes[f]]+[i]
            newDataset.hash = new_hashes
            #newDataset.hash = {i: np.where(new_files == i)[0].tolist() for i in list(set(newDataset.files))}

        if len(self.partitions) != 0:
            idx_hash = {idx[i]:i for i in range(len(idx))}
            newDataset.partitions = {}
            for name, indices in self.partitions.items():
                if len(indices) == 0:
                    continue
                if type(indices[0]) in [str, np.str_]:
                    indices = self.translate_files(indices)
                    newDataset.partitions[name] = list(filter(lambda x: x in newDataset.files, indices))
                else:
                    newDataset.partitions[name] = list(filter(lambda x: x is not None, [idx_hash.get(i) for i in indices]))
        newDataset.transforms = self.transforms
        newDataset.target_transforms = self.target_transforms
        newDataset.drop_sequence = self.drop_sequence
        newDataset.sequence_transform = self.sequence_transform
        newDataset.tasks = self.tasks
        return newDataset

    def retrieve_from_class_id(self, meta_id, task, ids=None, exclusive=False):
        """
        retrieves the sub-dataset containing the targeted classes ƒor a given task.
        :param meta_id: list of ids to filter
        :type meta_id: int, list(int), np.ndarray
        :param str task: task to filter
        :param ids: (optional) constrains the search to the provided ids
        :type ids: np.ndarray or list(int)
        :param bool exclusive: only select data ids with every provided ids, in case of multi-label information (default: False)
        :return: a filtered sub-dataset
        :rtype: :py:class:`Dataset`
        """
        ids = self.get_ids_from_class(meta_id, task, exclusive=exclusive, ids=ids)
        return self.retrieve(list(ids))

    def retrieve_from_class(self, meta_class, task, ids=None, exclusive=False):
        """
        retrieves the sub-dataset containing the targeted classes ƒor a given task.
        :param meta_id: list of ids to filter
        :type meta_id: int, list(int), np.ndarray
        :param str task: task to filter
        :param ids: (optional) constrains the search to the provided ids
        :type ids: np.ndarray or list(int)
        :param bool exclusive: only select data ids with every provided ids, in case of multi-label information (default: False)
        :return: a filtered sub-dataset
        :rtype: :py:class:`Dataset`
        """
        meta_id = [self.classes[task][m] for m in checklist(meta_class)]
        return self.retrieve_from_class_id(meta_id, task, exclusive=exclusive, ids=ids)

    #def has_transform(self):

    def list_directory(self):
        """
        Fill the list of files from a direct recursive path of a folder.
        This folder is specified at object construction with the
        data_directory attribute

        :param bool check: removes the file if the corresponding transform is not found (default: False)
        """
        # The final set of files
        filesList = []
        # Use glob to find all recursively
        valid_suffixes = "" if self.valid_types is None else tuple(self.valid_types)
        for dirpath, _, filenames in os.walk(self.data_directory):
            for f in filenames:
                if f.endswith(valid_suffixes) and f[0] != ".":
                    filesList.append(os.path.abspath(os.path.join(dirpath, f)))

        hashList = {}
        curFile = 0

        # Parse the list to have a hash
        for files in filesList:
            hashList[files] = curFile
            curFile = curFile + 1

        # Print info if verbose mode
        # Save the lists
        self.files = np.array(filesList);
        self.hash = hashList;
        if (self.verbose):
            print('[Dataset][List directory] Found ' + str(len(self.files)) + ' files.');

    def write_inputs(self):
        data_directory = self.data_directory or self.data_prefix
        with open(self.data_prefix+"/inputs.txt", 'w+') as inputs:
            for i, f in enumerate(self.files):
                inputs.write(re.sub(data_directory, "", f))
                if i != len(self.files) - 1:
                    inputs.write("\n")


    def list_directory_from_file(self, input_file):
        if not os.path.isfile(input_file):
            base_input_file = f"{self.data_prefix}/{input_file}"
            if not os.path.isfile(base_input_file):
                raise FileNotFoundError(f"neither {input_file} nor {base_input_file} has been found")
            else:
                input_file = base_input_file
        files = []; hash = {}
        with open(input_file, 'r') as inputs:
            for i, file in enumerate(inputs.readlines()):
                if file[-1] == "\n":
                    file = file[:-1]
                files.append(file)
                if hash.get(file) is None:
                    hash[self.data_directory+file] = i
                else:
                    hash[self.data_directory+file] = checklist(hash[file]+[i])
        data_directory = self.data_directory or self.data_prefix
        files = [f"{data_directory}{f}" for f in files]
        self.files = np.array(files)
        self.hash = hash


    def import_metadata_tasks(self, sort=True):
        """
        imports in the :py:class:`Dataset` object the metadata corresponding to the recorded tasks.
        :param bool sort: is metadata sorted (default : True)
        """
        tasks = []
        if not os.path.isdir(self.metadata_directory):
            if (self.verbose):
                print("no metadata found")
            return
        folders = list(
            filter(lambda x: os.path.isdir(self.metadata_directory + '/' + x), os.listdir(self.metadata_directory)))
        for f in folders:
            if os.path.isfile("%s/%s/metadata.txt" % (self.metadata_directory, f)):
                tasks.append(f)
        self.tasks = tasks
        self.metadata_callbacks = [None]*len(self.tasks)
        for t in range(len(self.tasks)):
            self.metadata_callbacks[t] = self.retrieve_callback_from_path(self.metadata_directory, self.tasks[t]) or meta.metadata_callbacks["default"] or []
        """
        if sort:
            for t in self.tasks:
                self.sort_classes(t)
        """

    def retrieve_callback_from_path(self, metadata_directory, task):
        """
        retrieve the appropriate callback for the given task.
        :param str metadata_directory: metadata directory
        :param str task: target task
        :return:
        """
        try:
            with open("%s/%s/callback.txt"%(metadata_directory, task)) as f:
                return getattr(meta, re.sub('\n', '', f.read()))
        except FileNotFoundError:
            pass
        return

    def sort_classes(self, tasks):
        """
        sort the given classes of the metadata *task*, such that class_ids are affected to the class names with a given
        order
        :param str task: sorted task
        """
        tasks = checklist(tasks)
        for task in tasks:
            private_keys = ['_length', None]
            private_dict = {p: self.classes[task].get(p) for p in private_keys}
            class_names = list(filter(lambda x: x not in private_keys, self.classes[task].keys()))
            clear_class_dict = {k: self.classes[task][k] for k in class_names}
            is_sortable = hasattr(class_names[0], '__gt__') or hasattr(class_names[0], '__lt__')
            if is_sortable:
                try:
                    class_names = clear_class_dict
                    class_keys = list(class_names.keys())
                    sorted_classes = np.argsort(np.array(class_keys))
                    inverse_class_hash = {v:k for k, v in class_names.items()}
                    for i in range(len(self.metadata[task])):
                        original_class = inverse_class_hash[self.metadata[task][i]]
                        class_idx = class_keys.index(original_class)
                        self.metadata[task][i] = np.where(sorted_classes == class_idx)[0]

                    new_classes = {c: i for i, c in enumerate(np.array(class_keys)[sorted_classes])}
                    self.classes[task] = {**new_classes, **private_dict}
                except Exception as e:
                    if self.verbose:
                        print('Task %s does not seem sortable.'%task)
                        pass

    @abstractmethod
    def _update_options(self, options):
        pass

    def import_data(self, ids=None, files=None, scale=True, options={}):
        if ids is not None and files is not None:
            raise ValueError('Dataset.import_data : whether ids or files must be given, but not both')
        # filter if asked
        if ids is not None:
            self.files = self.files[ids]
        elif files is not None:
            self.files = self.files[self.get_ids_from_files(files)]
        shapes = []
        self._update_options(options)
        # import
        for f in tqdm(self.files, total=len(self.files), desc="importing data"):
            data, metadata = self.import_callback(self, f, options=options)
            self.data.append(data)
            shapes.append(data.size())
            self.data_properties.append(metadata)
        for i in range(len(self.tasks)):
            metadata_file = f"{self.metadata_directory}/{self.tasks[i]}/metadata.txt"
            self.import_metadata(metadata_file, self.tasks[i], self.metadata_callbacks[i])
        self.sort_classes(self.tasks)
        self.scale_transforms(scale)

    def scale_transforms(self, scale=True, transforms=None):
        if not scale:
            return
        if transforms is None:
            transforms = self.transforms
        if scale is True:
            data = self.data[:]
        elif isinstance(scale, int):
            ids = torch.randperm(len(self.data))[:scale]
            data = [self.data[i] for i in ids]
        if not torch.is_tensor(data):
            data = data_utils.dyn_collate(data)
        transforms.scale(data)
        if self.sequence_transform is not None:
            sequence = self.get_sequence(data, batch_first=True)
            self.sequence_transform.scale(sequence)


    @abstractmethod
    def import_callback(self, f, options={}):
        raise NotImplementedError('import_callback function not implemented for a generic dataset. Please overload')

    def flatten_data(self, data=None, selector=lambda x: x, dim=0, merge_mode="min", stack=False):
        """
        if the data is built of nested arrays, flattens the data to be a single array. Typically, if each item of data
        is a sub-array of size (nxm), flatten_data concatenates among the first axis, and can optionally window among
        the second.
        If the second dimension of each sub-arrays do not match, it can be cropped (*merge_mode* min)
        or padded (*merge_mode* max)

        :param function selector: a lambda selector, that picks the wanted data in each sub-array
        :param int window: size of the window (default: None)
        :param float window_overlap: overlapping of windows
        :param str merge_mode: merging mode of nested array, if the dimensions are not matching ("min" or "max").
        """

        if isinstance(self.data, asyn.OfflineDataList):
            return self.flatten_data_offline(selector, stack)

        data = []; metadata = {k:[] for k in self.metadata.keys()}
        data_properties = []
        hash = {k: [] for k in self.hash.keys()}; files = []
        running_id = 0
        shapes = []
        original_data = data or self.data
        for i in tqdm(range(len(original_data)), total=len(original_data), desc="flattening data..."):
            new_data = selector(original_data[i])
            new_shape = (torch.prod(torch.Tensor(list(new_data.shape[:dim+1]))).int().item(),
                         *new_data.shape[dim+1:])
            shapes.append(new_shape)
            data.append(new_data.view(*new_shape))
        sizes = torch.Tensor(shapes).int()
        if merge_mode == "min":
            target_size, _ = sizes.min(dim=0)
        elif merge_mode == "max":
            target_size, _ = sizes.max(dim=0)

        running_id = 0
        for i, d in enumerate(data):
            data[i] = data_utils.dyn_expand(data[i], target_size)
            add_len = data[i].shape[0]
            hash[self.files[i]] = list(range(running_id, running_id+add_len))
            files.extend([self.files[i]]*add_len)
            data_properties.extend([self.data_properties[i]]*add_len)
            running_id += add_len

        data = torch.cat(data, dim=0)
        for k, v in metadata.items():
            metadata[k] = torch.cat([torch.Tensor([self.metadata[k][i]]).int().repeat(target_size[0]) for i in range(len(sizes))], 0)

        self.data = data; self.metadata = metadata
        self.files = files; self.hash = hash
        self.data_properties = data_properties


    def flatten_data_offline(self, selector=lambda x: x, padded=False):
        # initialize
        newData = []
        newMetadata = {}
        for k, v in self.metadata.items():
            newMetadata[k] = []
        newFiles = []
        newPartitions = {k:[] for k in self.partitions.keys()}
        revHash = {}
        # new hash from scratch
        newHash = {k:[] for k in self.hash.keys()}
        # filter dataset
        idx = 0
        for i in range(len(self.data)):
            # update minimum content shape
            chunk_to_add = selector(self.data.entries[i].split())
            newData.extend(chunk_to_add)
            for k, _ in newMetadata.items():
                if isinstance(self.metadata[k][i], Number):
                    newMetadata[k].extend([self.metadata[k][i]]*len(chunk_to_add))
                else:
                    newMetadata[k].extend(self.metadata[k][i])
            newFiles.extend([self.files[i]]*len(chunk_to_add))
            current_idxs = list(range(idx, idx + len(chunk_to_add)))
            newHash[self.files[i]].extend(current_idxs)
            for name, part in self.partitions.items():
                if i in part or self.files[i] in part:
                    if type(self.partitions[name][0])==int:
                        newPartitions[name].extend(current_idxs)
                    else:
                        newPartitions[name].append(self.files[i])
            idx += len(chunk_to_add)

        self.data = asyn.OfflineDataList(newData, dtype=newData[0]._dtype, padded=padded)
        self.metadata = newMetadata
        for k,v in newMetadata.items():
            newMetadata[k] = np.array(v)
        self.files = newFiles
        self.hash = newHash
        self.revHash = revHash
        self.partitions = newPartitions

    def import_metadata(self, fileName, task, callback):
        """
        Import the metadata given in a file, for a specific task
        The function callback defines how the metadata should be imported
        All these callbacks should be in the importUtils file

        Parameters
        ----------
        :param str fileName: Path to the file containing metadata
        :param str task: name of the metadata task
        :param function callback:  Callback defining how to import metadata

        """
        # Try to open the given file
        try:
            fileCheck = open(fileName, "r")
        except:
            print('[Dataset][List file] Error - ' + str(fileName) + ' does not exists !.')
            return None
        # Create data structures
        metaList = [None] * len(self.files)
        curFile, curHash = len(self.files), -1
        testFileID = None
        classList = {"_length": 0}

        for line in fileCheck:
            if line[-1] == '\n':
                line = line[:-1]
            if line[0] != "#" and len(line) > 1:
                vals = line.split('\t')  # re.search("^(.+)\t(.+)$", line)
                if len(vals) != 2:
                    continue
                audioPath, metaPath = vals[0], (vals[1] or "")  # vals.group(1), vals.group(2)
                if (audioPath is not None):
                    fFileName = os.path.abspath(self.data_prefix + '/' + audioPath);
                    try:
                        assert (fFileName in self.files)
                    except:
                        continue
                    try:
                        testFileID = open(fFileName, 'r')
                    except:
                        if self.verbose:
                            print(
                                '[Dataset][Metadata import] Warning loading task ' + task + ' - File ' + fFileName + ' does not exists ! (Removed from list)')
                        continue

                    if (testFileID):
                        testFileID.close()

                    if fFileName in self.hash.keys():
                        curHash = self.hash[fFileName]
                        if (len(metaList) - 1 < max(checklist(curHash))):
                            metaList.append("")
                        curHash = checklist(curHash)
                        for ch in curHash:
                            metaList[ch], classList = callback(metaPath, classList,
                                                               {"prefix": self.data_prefix, "files": self.files})
                    else:
                        pass
                else:
                    metaList.append({})

        # Save the lists
        if None in metaList:
            for i in range(len(metaList)):
                if metaList[i] is None:
                    metaList[i] = [-1]
            classList['None'] = -1

        self.metadata[task] = torch.Tensor(np.array(metaList))
        label_files = '/'.join(fileName.split('/')[:-1]) + '/classes.txt'
        classList = classList or {str(k): k for k in set(self.metadata[task])}
        if os.path.isfile(label_files):
            classes_raw = open(label_files, 'r').read().split('\n')
            classes_raw = [tuple(c.split('\t')) for c in classes_raw]
            classes_raw = list(filter(lambda x: len(x) == 2, classes_raw))
            imported_class_dict = dict(classes_raw)
            if classList.get('_length') is not None:
                del classList['_length']
            self.classes[task] = {imported_class_dict[label]: classList[label] for label in classList.keys()}
            self.classes[task]['_length'] = len(classes_raw)
        else:
            self.classes[task] = classList;

    def get_ids_from_files(self, files):
        ids = []; files = checklist(files)
        for f in files:
            current_ids = self.hash.get(f)
            if current_ids is None:
                if self.verbose:
                    print("[Warning] file %s not present in current dataset"%f)
                pass
            if issubclass(type(current_ids), (tuple, list)):
                ids.extend(list(current_ids))
            else:
                ids.append(current_ids)



    def construct_partition(self, partitionNames, partitionPercent, tasks=[], balancedClass=False, equalClass=False):
        """
        Construct a random/balanced partition set for each dataset
        Only takes indices with valid metadatas for every task
        now we can only balance one task

        :param tasks: balanced tasks
        :type tasks: str, list(str)
        :param list partitionNames: names of partition (list of str)
        :param list partitionPercent: relative proportion of each partition (summing up to 1)
        :param bool balancedClass: has the partition to be balanced for each class across partitions
        :param bool equalClass: enforces each class to be present with the same number of instances
        """
        #TODO balancing broken
        tasks = checklist(tasks)
        if tasks is None and balancedClass:
            tasks = self.tasks
        if (balancedClass is True):
            balancedClass = tasks[0]
        # Checking if tasks exist
        for t in tasks:
            if (self.metadata[t] is None):
                print("[Dataset] error creating partitions : " + t + " does not seem to exist")
                return None
        # making temporary index from mutal information between tasks
        mutual_ids = []

        data_shape = len(self.data)
        for i in range(data_shape):
            b = True
            for t in tasks:
                b = b and (self.metadata[t][i] is not None)
            if (b):
                mutual_ids.append(i)
        # Number of instances to extract
        nbInstances = len(mutual_ids)
        if (len(mutual_ids) == 0):
            if type(self.metadata[tasks[1]]) is np.ndarray:
                nbInstances = (self.metadata[tasks[0]].shape[0])
            else:
                nbInstances = len(self.metadata[tasks[0]])
            for i in range(nbInstances):
                mutual_ids[i] = i
        partitions = {}
        runningSum = 0
        partSizes = np.zeros(len(partitionNames))
        for p in range(len(partitionNames)):
            partitions[partitionNames[p]] = []
            if (p != len(partitionNames)):
                partSizes[p] = np.floor(nbInstances * partitionPercent[p])
                runningSum = runningSum + partSizes[p]
            else:
                partSizes[p] = nbInstances - runningSum;
        # Class-balanced version
        if balancedClass:
            # Perform class balancing
            curMetadata = self.metadata[balancedClass];
            curClasses = self.classes[balancedClass];
            nbClasses = curClasses["_length"];
            countclasses = np.zeros(nbClasses);
            classIDs = {};
            # Count the occurences of each class
            for idC in range(len(mutual_ids)):
                s = mutual_ids[idC]
                countclasses[curMetadata[s]] = countclasses[curMetadata[s]] + 1;
                # Record the corresponding IDs
                if (not classIDs.get(curMetadata[s])):
                    classIDs[curMetadata[s]] = [];
                classIDs[curMetadata[s]].append(s);
            if equalClass:
                minCount = np.min(countclasses)
                for c in range(nbClasses):
                    countclasses[c] = int(minCount);
            for c in range(nbClasses):
                if (classIDs[c] is not None):
                    curIDs = np.array(classIDs[c]);
                    classNb, curNb = 0, 0;
                    shuffle = np.random.permutation(int(countclasses[c]))
                    for p in range(len(partitionNames)):
                        if equalClass:
                            classNb = np.floor(partSizes[p] / nbClasses);
                        else:
                            classNb = np.floor(countclasses[c] * partitionPercent[p])
                        if (classNb > 0):
                            for i in range(int(curNb), int(curNb + classNb - 1)):
                                partitions[partitionNames[p]].append(curIDs[shuffle[np.min([i, shuffle.shape[0]])]])
                            curNb = curNb + classNb;
        else:
            # Shuffle order of the set
            shuffle = np.random.permutation(len(mutual_ids))
            curNb = 0
            for p in range(len(partitionNames)):
                part = shuffle[int(curNb):int(curNb+partSizes[p]-1)]
                for i in range(part.shape[0]):
                    partitions[partitionNames[p]].append(mutual_ids[part[i]])
                curNb = curNb + partSizes[p];
        for p in range(len(partitionNames)):
            self.partitions[partitionNames[p]] = np.array(partitions[partitionNames[p]])
            self.partitions[partitionNames[p]].sort()
        return partitions

    def translate_files(self, files):
        """
        translate an incoming list of data paths with the current root directory.
        :param list files: list of file paths to translate
        :return: translated files
        :rtype: list(str)
        """

        if self.data_prefix.split('/')[-1] not in files[0]:
            raise Exception('given files do not seem in current dataset')
        oldRoot = files[0][:re.search(self.data_prefix.split('/')[-1], files[0]).regs[0][1]]
        translated_files = list({re.sub(oldRoot, self.data_prefix, k) for k in files})
        translated_files = list(filter(lambda x: x in self.hash.keys(), translated_files))
        return translated_files

    def filter_files(self, files):
        """
        returns a sub-dataset containing data only extracted from the given array of files
        :param list files: list of files to retrieve
        :returns: sub-dataset
        :rtype: :py:class:`Dataset`
        """
        files = list(set(files))
        translated_files = self.translate_files(files)
        ids = list(set(sum([checklist(self.hash[k]) for k in translated_files], [])))
        return self.retrieve(ids)

    def filter_classes(self, class_names, task):
        class_names = checklist(class_names)
        ids = set();
        for cn in class_names:
            ids = ids.union(set(self.get_ids_from_class(self.classes[task][cn], task)))
        cur_length = len(self.data)
        if len(self.data) == 0:
            cur_length = len(self.metadata[list(self.metadata.keys())[0]])
        assert cur_length != 0, "please import data or metadata before filtering "

        valid_ids = list(set(list(range(cur_length))).difference(ids))
        return self.retrieve(valid_ids)

    def random_subset(self, n_files, shuffle=True):
        """
        just keep a given number of files and drop the rest
        :param n_files: number of kept audio files
        :param shuffle: randomize audio files (default: True)
        """
        files_set = set(self.files)
        assert n_files < len(files_set), "number of amputated files greater than actual number of files!"
        if shuffle:
            selected_files = np.random.choice(np.array(list(files_set)), n_files, replace=False)
        else:
            selected_files = list(files_set)[:n_files]
        return self.filter_files(selected_files)

    def apply_transforms(self, transform=None):
        transform = transform or self.transforms
        if isinstance(self.data, asyn.OfflineDataList):
            raise NotImplementedError
        else:
            for i in tqdm(range(len(self.data)), total=len(self.data), desc="applying transforms..."):
                self.data[i] = transform(self.data[i])
        self.transforms = self.compose_function()

    def _scale_to_write(self, transforms, scale):
        if not scale:
            return
        elif scale is True:
            data = self[:][0]
        elif isinstance(scale, int):
            ids = torch.randperm(len(self.data))[:scale]
            data = [self[i][0] for i in ids]
        transforms.scale(data)

    def _get_data_to_write(self, item, transforms=None):
        data = self.transforms(self.data[item])
        if transforms is not None:
            data = transforms(data)
        return data, {}

    def write_transform(self, name, transforms=None, scale=True):
        if not isinstance(transforms, self.compose_function) and transforms is not None:
            transforms = self.compose_function(transforms)
        target_dir = f"{self.analysis_directory}/{name}"
        parsing_hash = {}
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            print(self.analysis_directory, name, target_dir)

        if transforms is not None:
            self._scale_to_write(transforms, scale)

        for i, f in tqdm(enumerate(self.files.tolist()), total=self.files.shape[0], desc="exporting transform %s"%name):
            original_file = re.sub(os.path.abspath(self.data_directory), '', f)
            target_file = os.path.splitext(target_dir + re.sub(os.path.abspath(self.data_directory), '', f))[0] + ".dat"
            if not os.path.isdir(os.path.dirname(target_file)):
                os.makedirs(os.path.dirname(target_file))
            transformed_data, add_info = self._get_data_to_write(i, transforms)
            if isinstance(transformed_data, list):
                raise ValueError('Saving as memmap is not available with multi output')
            if torch.is_tensor(transformed_data):
                transformed_data = transformed_data.detach().cpu().numpy()
            elif not issubclass(type(transformed_data), np.ndarray):
                transformed_data = np.array(transformed_data)
            current_memmap = np.memmap(target_file, transformed_data.dtype, 'w+', shape=transformed_data.shape)
            current_memmap[:] = np.ascontiguousarray(transformed_data)[:]
            parsing_hash[original_file] = {'shape':transformed_data.shape, 'strides':transformed_data.strides,
                                  'dtype':transformed_data.dtype, **self.data_properties[i], **add_info}
        with open(target_dir+'/parsing.vs', 'wb+') as parsing_file:
            dill.dump(parsing_hash, parsing_file)
        with open(target_dir+'/transforms.vs', 'wb+') as parsing_file:
            if transforms is None:
                transforms = self.transforms
            else:
                transforms = self.transforms + transforms
            dill.dump(transforms, parsing_file)

    def load_transform(self, name, ids=None, offline=True, selector=asyn.Selector, selector_args={}):
        if ids is None:
            analysis_dir = f"{self.analysis_directory}/{name}"
            with open(analysis_dir+'/parsing.vs', 'rb') as f:
                parsing_hash = dill.load(f)
            try:
                with open(analysis_dir+'/transforms.vs', 'rb') as f:
                    transforms = dill.load(f)
            except FileNotFoundError:
                transforms = None
                if self.verbose:
                    print('[Warning] transforms not found for transform %s'%name)
            data = []
            data_properties = [None]*len(self.files)
            if isinstance(selector, str):
                selector = getattr(asyn, selector)
            for i, f in tqdm(enumerate(self.files), total=len(self.files), desc="loading transform %s"%name):
                target_file = os.path.splitext(re.sub(self.data_directory, analysis_dir, f))[0] + ".dat"
                original_file = re.sub(os.path.abspath(self.data_directory), '', f)
                current_entry = parsing_hash[original_file]
                if offline:
                    entry = asyn.OfflineEntry(target_file, selector=selector(**selector_args), shape=current_entry['shape'],
                                         dtype=current_entry['dtype'], strides=current_entry['strides'])
                    data.append(entry)
                else:
                    current_mm = np.memmap(target_file, dtype=current_entry['dtype'], shape=current_entry['shape'], mode="r")
                    data.append(torch.from_numpy(np.array(current_mm)))
                    del current_mm
                data_properties[i] = current_entry

            new_dataset = self.retrieve(None)
            if offline:
                new_dataset.data = asyn.OfflineDataList(data)
            else:
                new_dataset.data = data
            new_dataset.data_properties = data_properties

            if self.metadata == {}:
                for i in range(len(self.tasks)):
                    metadata_file = f"{self.metadata_directory}/{self.tasks[i]}/metadata.txt"
                    new_dataset.import_metadata(metadata_file, self.tasks[i], self.metadata_callbacks[i])
                new_dataset.sort_classes(self.tasks)
            else:
                new_dataset.metadata = self.metadata
                new_dataset.metadata_callbacks = self.metadata
                new_dataset.classes = self.classes
            return new_dataset, transforms
        else:
            return self.retrieve(ids).load_transform(name, offline=offline, selector=selector, selector_args=selector_args)

    def has_transform(self, name):
        has_transform = True
        if not os.path.isdir(f"{self.analysis_directory}/{name}"):
            has_transform = False
        if not os.path.isfile(f"{self.analysis_directory}/{name}/parsing.vs"):
            has_transform = False
        return has_transform


from torchvision.transforms import ToPILImage

def dataset_from_torchvision(name, *args, path=None, transforms=[], target_transforms=[], **kwargs):
    """
    Creates a :py:class:`Dataset` from a torchvision dataset.
    :param str name: name of torchvision module
    :param transform: transforms
    :param target_transform: target_transforms
    :return: torch dataset
    :rtype: :py:class:`Dataset`
    """
    current_path = path or os.path.dirname(__file__) + '/toys'
    full_dataset = Dataset(current_path)
    if not isinstance(transforms, ComposeTransform):
        transforms = ComposeTransform(transforms)
    if not isinstance(target_transforms, ComposeTransform):
        target_transform = ComposeTransform(target_transforms)
    if name in ["FakeData"]:
        full_dataset = getattr(torchvision.datasets, name)(**kwargs, transform=transforms, target_transform=target_transform)
        full_dataset.classes = {'class': {k: k for k in range(full_dataset.num_classes)}}
        full_dataset.data_properties = [None]*len(full_dataset)
        full_dataset.tasks = ['class']
        full_dataset.transforms = transforms
    elif name in ['SBU', 'CocoCaptions', 'CocoDetection', 'Flickr', 'Kinetics-400', 'HMDB51']:
        train_dataset = getattr(torchvision.datasets, name)(current_path, *args, **kwargs, transform=transforms, target_transform=target_transform)
        full_dataset.data = train_dataset[0]
        train_dataset.data_properties = [None]*full_dataset.data.size(0)
        train_dataset.metadata = {'class':train_dataset.targets}
        full_dataset.classes = {'class': {k: k for k in range(full_dataset.metadata['class'].min(),
                                                              full_dataset.metadata['class'].max() + 1)}}
        full_dataset.transforms = transforms
    else:
        if name in ['LSUN']:
            train_dataset = getattr(torchvision.datasets, name)(current_path, *args,
                                                                target_transform=target_transform, download=True,
                                                                classes="train", **kwargs)
            test_dataset = getattr(torchvision.datasets, name)(current_path, *args,
                                                               target_transform=target_transform, download=True,
                                                               classes="test", **kwargs)
        elif name in ['ImageNet', 'STL10', 'SVHN', 'Cityscapes', 'CelebA']:
            train_dataset = getattr(torchvision.datasets, name)(current_path, *args,
                                                                target_transform=target_transform, download=True,
                                                                split="train", **kwargs)
            test_dataset = getattr(torchvision.datasets, name)(current_path, *args,
                                                               target_transform=target_transform, download=True,
                                                               split ="test", **kwargs)
        elif name in ['VOCDetection', 'VOCSegmentation', 'SBD']:
            train_dataset = getattr(torchvision.datasets, name)(current_path, *args,
                                                                target_transform=target_transform, download=True,
                                                                image_set="train", **kwargs)
            test_dataset = getattr(torchvision.datasets, name)(current_path, *args,
                                                               target_transform=target_transform, download=True,
                                                               image_set ="test", **kwargs)
        else:
            train_dataset = getattr(torchvision.datasets, name)(current_path, *args,
                                                                target_transform=target_transform, download=True,
                                                                train=True, **kwargs)
            test_dataset = getattr(torchvision.datasets, name)(current_path, *args,
                                                               target_transform=target_transform, download=True,
                                                               train=False, **kwargs)
        if name in ["STL10", "SVHN"]:
            train_dataset.targets = train_dataset.labels
            test_dataset.targets = test_dataset.labels
        if name in ["CIFAR10", 'STL10', 'SVHN']:
            train_dataset.data = torch.from_numpy(train_dataset.data)
            test_dataset.data = torch.from_numpy(test_dataset.data)
        full_dataset.data = torch.cat(
            [train_dataset.data, test_dataset.data], axis=0)
        full_dataset.partitions = {'train': np.arange(train_dataset.data.shape[0]),
                                   'test': (train_dataset.data.shape[0] + np.arange(
                                       test_dataset.data.shape[0]))}
        # pdb.set_trace()
        full_dataset.metadata = {'class': torch.cat([torch.LongTensor(train_dataset.targets.long()),
                                torch.LongTensor(test_dataset.targets.long())])}
        full_dataset.classes = {'class': {k: k for k in range(full_dataset.metadata['class'].min(),
                                                              full_dataset.metadata['class'].max() + 1)}}
        full_dataset.classes['class']['_length'] = len(full_dataset.classes['class'].keys())
        full_dataset.tasks = ['class']
        full_dataset.data_properties = [None]*full_dataset.data.size(0)
        full_dataset.transforms = transforms
    return full_dataset




