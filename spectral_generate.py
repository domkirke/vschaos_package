import matplotlib; matplotlib.use('agg')
import vschaos
import numpy as np, torch, argparse, random, os, matplotlib.pyplot as plt, gc, pdb, dill
from scipy import ndimage
from vschaos.data import DatasetAudio, ComposeAudioTransform
from vschaos.utils.dataloader import DataLoader
from vschaos.monitor.audio_synthesize import resynthesize_files, trajectory2audio, interpolate_files
from vschaos.monitor.visualize_monitor import Monitor
from vschaos.monitor.visualize_dimred import PCA, ICA
from vschaos.monitor.evaluate_reconstruction import SpectralEvaluation
from vschaos.monitor.evaluate_latent import LatentEvaluation, DisentanglementEvaluation
from vschaos.utils.misc import concat_tensors, denest_dict, Logger, parse_folder, check_dir, checklist, decudify, merge_dicts, get_conditioning_params



#%% Script arguments
parser = argparse.ArgumentParser()
optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
required.add_argument('-m', '--models', type=str, nargs='+', required=True, help="models to test")
optional.add_argument('-c', '--cuda', type=int, nargs="*", default=[], help='cuda driver (-1 for cpu)')
optional.add_argument('-o', '--output', type=str, default=None, help="path for output figures (default is model path)")

# analysis tasks
optional.add_argument('--resynthesize', type=int, default=0, help="resynthesize full files")
optional.add_argument('--preprocessing', type=str, default=0, help="pickle for external preprocessing")
optional.add_argument('--evaluate', type=int, default=0)
optional.add_argument('--trajectories', type=int, default=0)
optional.add_argument('--traj_steps', type=int, default=400)

optional.add_argument('--cross_synthesis', type=str, default=None, nargs="*")
optional.add_argument('--interpolations', type=int, default=0)
optional.add_argument('--target_duration', type=float, default=3.0)

# misc parameters
parser.add_argument('--preload', type=int, default=0, help="")
parser.add_argument('--batch_size', type=int, default=None, help="batch size for forwards (default : full)")
parser.add_argument('--stat_points', type=int, default=None, help="batch size for forwards (default : full)")
parser.add_argument('--transform_type', type=str, default="stft")
parser.add_argument('--iterations', type=int, default=20, help="number of griffin-lim iterations")
parser.add_argument('--dis_epochs', type=int, default=500)
parser.add_argument('--n_interp', type=int, default=2)
args = parser.parse_args()

# Create dataset object

evaluations = {}
models = []

for model in args.models:
    if os.path.splitext(model)[1] == ".t7":
        models.append(model)
    elif os.path.isdir(model):
        for r,d,f in os.walk(model):
            valid_models = list(filter(lambda x: os.path.splitext(x)[1] in (".t7", ".pth"), f))
            models.extend([r+'/'+m for m in valid_models])

def get_label_params(init_args):
    label_params = {}
    hidden_params = checklist(init_args['hidden_params'])
    for hp in hidden_params:
        if hp.get('encoder'):
            if isinstance(hp['encoder'], list):
                for i in range(len(hp['encoder'])):
                    label_params = {**hp['encoder'][i].get('label_params', {}), **label_params}
            else:
                label_params = {**hp['encoder'].get('label_params', {}), **label_params}
            if isinstance(hp['decoder'], list):
                for i in range(len(hp['decoder'])):
                    label_params = {**hp['decoder'][i].get('label_params', {}), **label_params}
            else:
                label_params = {**hp['decoder'].get('label_params', {}), **label_params}

        else:
            label_params = {**hp.get('label_params', {}), **label_params}
    for lp in checklist(init_args['latent_params']):
        for lp_tmp in checklist(lp):
            if lp_tmp.get('task'):
                label_params = {lp_tmp['task']:{'dim':lp['dim', 'dist':lp['dist']]}, **label_params}
    return label_params


for current_path in args.models:
    try:
        print("-- model %s"%current_path)
        # import model data
        loaded_data = torch.load(current_path, map_location=torch.device('cpu'))

        if not 'state_dict' in loaded_data.keys():
            print('bypassing model %s'%current_path)
            continue

        script_args = loaded_data.get('script_args')
        current_epoch = loaded_data.get('epoch', -1)
        label_params = get_label_params(loaded_data['init_args'])

        transforms = loaded_data.get('full_transforms')
        target_transforms = loaded_data.get('target_transforms', {}) or {}

        losses = loaded_data.get('loss')
        # retrieve model
        print('retrieving model...')
        vae = loaded_data['class'].load(loaded_data)

        out_path = args.output or os.path.dirname(current_path)
        logger = Logger(verbose=True, log_file=out_path+'/%s_evaluation.txt'%(os.path.splitext(os.path.basename(current_path))[0]))
        if len(args.cuda) > 0:
            if args.cuda[0] >= 0:
                vae.cuda(args.cuda[0])
                if len(args.cuda) > 1:
                    vae = vschaos.DataParallel(vae, args.cuda, output_device=args.cuda[0])
        logger(vae)
        vae.eval()

        print('performing cross-synthesis...')
        if not issubclass(type(args.cross_synthesis), list):
            if args.cross_synthesis is not None:
                args.cross_synthesis = [args.cross_synthesis]
            else:
                args.cross_synthesis = []
        if len(args.cross_synthesis) != 0:
            files_to_resynthesize = []
            valid_exts = ['.wav', '.aif', '.aiff']
            for f in args.cross_synthesis:
                if os.path.isdir(f):
                    v = parse_folder(f, exts=valid_exts)
                    files_to_resynthesize.extend(v)
                else:
                    if os.path.splitext(f)[1] in valid_exts:
                        files_to_resynthesize.append(f)
            check_dir(out_path+'/cross')
            meta = {}
            for k, v in label_params.items():
                n_classes = v['dim']
                meta[k] = torch.randint(n_classes, (len(files_to_resynthesize),))
            resynthesize_files(None, vae, files=files_to_resynthesize, transforms=transforms,
                               sequence=script_args.sequence, out=out_path+'/cross', metadata=meta, retain_phase=True)


        print('processing random trajectories...')
        if args.trajectories:
            meta = {}
            for k, v in label_params.items():
                n_classes = v['dim']
                meta[k] = torch.randint(n_classes, (args.trajectories,))
            trajectory2audio(vae, ["line", "ellipse", "square", "sin", "triangle", "sawtooth"], transforms, n_trajectories = args.trajectories,
                             n_steps=args.traj_steps, out=out_path, sample=False, meta=meta)

        del vae;
        gc.collect(); gc.collect()



    except Exception as e:
        raise e
        print('There has been an excecption : %s'%e)
        print('for model : %s'%current_path)
        pass


