import argparse
import torch
import numpy as np
from vschaos.data.data_audio import DatasetAudio
from vschaos.utils.oscServer import VAEServer
torch.init_num_threads()


parser = argparse.ArgumentParser()
parser.add_argument('--in_port', type=int)
parser.add_argument('--out_port', type=int)
parser.add_argument('-m', '--model', type=str, default=None)
parser.add_argument('-d', '--dbroot', type=str, default=None)
parser.add_argument('-t', '--transform', type=str, default=None)
parser.add_argument('--files', type=int, default=None)
parser.add_argument('--verbose', type=int, default=True)
parser.add_argument('--ip', type=str, default="127.0.0.1")
parser.add_argument('--frames', type=int, nargs="*", default=[])
args = parser.parse_args()


#%% Load model
model = None
preprocessing = None
if args.model:
    raw_data = torch.load(args.model, map_location="cpu")
    model = raw_data['class'].load(raw_data)
    args.frames = raw_data['script_args'].frames
    preprocessing = raw_data.get('preprocessing')
    # if preprocessing is None:
    #     preprocessing = Magnitude('none', 'none', 'none')
    # preprocessing = Magnitude('log1p', 'none', 'none')


audioSet = None
if args.dbroot:
    #%% Loader & loss function
    print('[Info] Loading data...')
    # import parameters
    audioOptions = {
        "dataPrefix":args.dbroot,
        "transformName":args.transform,
        "verbose":True,
        "forceUpdate":False,
        "forceRecompute":False,
    }


    # Create dataset object
    audioSet = DatasetAudio(audioOptions)
    # Recursively check in the given directory for data
    audioSet.list_directory()
    audioSet.retrieve_tasks()

    # import data
    if args.files:
        audioSet = audioSet.remove_files(args.files)
    audioSet.import_metadata_tasks()
    audioSet.import_data(None, audioOptions)

    if len(args.frames) == 0:
        print('taking the whole dataset...')
        audioSet.flatten_data(lambda x: x[:])
    elif  len(args.frames)==1:
        print('taking frame %d'%(args.frames[0]))
        audioSet.flatten_data(lambda x: x[args.frames[0]])
    elif len(args.frames) == 2:
        print('taking between %d and %d...'%(args.frames[0], args.frames[1]))
        audioSet.flatten_data(lambda x: x[args.frames[0]:args.frames[1]])


server = VAEServer(args.in_port, args.out_port, model=model, dataset=audioSet, preprocessing=preprocessing, verbose=args.verbose)
print('running server...')
server.run()
