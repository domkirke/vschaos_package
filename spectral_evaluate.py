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
required.add_argument('-d', '--dbroot', type=str, required=True, help="path of corresponding database")
optional.add_argument('-t', '--transform', type=str, default=None, help="transform imported")
optional.add_argument('-f', '--files', type=int, default=None, help="number of files taken from the database")
optional.add_argument('-c', '--cuda', type=int, nargs="*", default=[], help='cuda driver (-1 for cpu)')
optional.add_argument('-o', '--output', type=str, default=None, help="path for output figures (default is model path)")
optional.add_argument('--offline', type=int, default=0, help="offline dataset import")

# analysis tasks
optional.add_argument('--tasks', type=str, nargs="*", default=[])
optional.add_argument('--resynthesize', type=int, default=0, help="resynthesize full files")
optional.add_argument('--preprocessing', type=str, default=0, help="pickle for external preprocessing")
optional.add_argument('--evaluate', type=int, default=0)
optional.add_argument('--trajectories', type=int, default=0)
optional.add_argument('--traj_steps', type=int, default=400)

optional.add_argument('--cross_synthesis', type=str, default=None, nargs="*")
optional.add_argument('--interpolations', type=int, default=0)
optional.add_argument('--target_duration', type=float, default=3.0)
# plot tasks
optional.add_argument('--plot_reconstructions', type=int, default=0)
optional.add_argument('--plot_latentspace', type=int, default=0)
optional.add_argument('--plot_dims', type=int, default=0)
optional.add_argument('--plot_consistency', type=int, default=0)
optional.add_argument('--plot_trajs', type=int, default=0)
optional.add_argument('--plot_statistics', type=int, default=0)
optional.add_argument('--plot_losses', type=int, default=0)
optional.add_argument('--plot_class_losses', type=int, default=0)
optional.add_argument('--plot_2descriptors', type=int, default=0)
optional.add_argument('--plot_3descriptors', type=int, default=0)
# misc parameters
parser.add_argument('--preload', type=int, default=0, help="")
parser.add_argument('--batch_size', type=int, default=None, help="batch size for forwards (default : full)")
parser.add_argument('--n_points', type=int, default=None, help="batch size for forwards (default : full)")
parser.add_argument('--stat_points', type=int, default=None, help="batch size for forwards (default : full)")
parser.add_argument('--transform_type', type=str, default="stft")
parser.add_argument('--iterations', type=int, default=20, help="number of griffin-lim iterations")
parser.add_argument('--plot_tasks', type=str, nargs='*', default=None)
parser.add_argument('--descriptor_scope', type=float, nargs=2, default=[-10, 10])
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


for current_path in args.models:
    currentSet = DatasetAudio(args.dbroot, input_file="inputs.txt")
    try:
        print("-- model %s"%current_path)
        # import model data
        loaded_data = torch.load(current_path, map_location=torch.device('cpu'))
        currentSet.partitions = loaded_data.get('partitions')[0]

        if 'files' in loaded_data.keys():
            currentSet = currentSet.filter_files(loaded_data['files'])
        if args.files:
            currentSet = currentSet.random_subset(args.files)

        if not 'state_dict' in loaded_data.keys():
            print('bypassing model %s'%current_path)
            continue

        script_args = loaded_data.get('script_args')
        current_epoch = loaded_data.get('epoch', -1)

        transform  = loaded_data.get('transform')
        if transform is None:
            currentSet.transforms = loaded_data.get('full_transforms')
            currentSet.target_transforms = loaded_data.get('target_transforms', {}) or {}
            currentSet.import_data(scale=False)
        else:
            currentSet, loaded_transforms = currentSet.load_transform(transform, offline=args.offline)
            full_transforms = loaded_data['full_transforms']
            dataset_transforms = full_transforms[len(loaded_transforms):]
            currentSet.transforms = ComposeAudioTransform(dataset_transforms)
            currentSet.target_transforms = loaded_data.get('target_transforms', {}) or {}

        if not script_args.sequence:
            currentSet.apply_transforms()
            currentSet.flatten_data(dim=0)


        losses = loaded_data.get('loss')
        tasks = args.tasks or currentSet.tasks
        plot_tasks = args.plot_tasks or tasks

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

        # get projections
        need_projections = args.plot_latentspace or args.plot_consistency or args.plot_dims or args.plot_2descriptors or args.plot_3descriptors or args.interpolations
        projection_types = [PCA, ICA]
        projections = []
        if need_projections:
            trainSet = currentSet.retrieve('train')
            ids = np.random.permutation(len(trainSet.data))[:args.n_points]
            loader = DataLoader(trainSet, args.batch_size, tasks=currentSet.tasks, ids=ids)
            outs = []
            with torch.no_grad():
                for x,y in loader:
                    vae_out = vae.encode(vae.format_input_data(x), y=y)
                    outs.append(decudify(vae_out))
                outs = merge_dicts(outs)
                for p_type in projection_types:
                    transforms = []
                    for l in range(len(checklist(vae.platent))):
                        transform = p_type(n_components=3)
                        transform.fit(outs[l])
                        transforms.append(transform)
                    projections.append(transforms)
            gc.collect(); gc.collect(); torch.cuda.empty_cache()


        # plotting
        print('plotting...')
        plots = {}
        label_params = get_conditioning_params(vae)
        if args.plot_reconstructions:
            plots['reconstructions'] = {"n_points":args.plot_reconstructions, "preprocessing":dataset.transforms, 'plot_multihead':True}
        if args.plot_latentspace:
            plots['latent_space'] = [{'transformation':projections[0], 'preprocessing':None, 'tasks':currentSet.tasks,
                                      'balanced':True, 'batch_size':args.batch_size, 'n_points':args.n_points, 'name':'PCA'},
                                     {'transformation':projections[1], 'preprocessing':None, 'tasks':currentSet.tasks,
                                      'balanced':True, 'batch_size':args.batch_size, 'n_points':args.n_points, 'name':'ICA'}]
        if args.plot_dims:
            plots['latent_dims'] = {'reductions':[None]+projections, 'preprocessing':None, 'tasks':currentSet.tasks,
                                    'balanced':True, 'batch_size':args.batch_size, 'n_points':args.n_points}
        if args.plot_consistency:
            plots['latent_consistency'] = {'reductions':[None]+projections, 'preprocessing':None, 'tasks':currentSet.tasks,
                                           'balanced':True, 'batch_size':args.batch_size, 'n_points':args.plot_consistency}
        if args.plot_trajs:
            plots['latent_trajs'] = {'preprocessing':None, 'tasks':args.tasks, 'batch_size':args.batch_size,
                                     'balanced':True, 'n_points':args.n_points}
        if args.plot_statistics:
            plots['statistics'] = {'tasks':currentSet.tasks, 'n_points':args.n_points, 'batch_size':args.batch_size,
                                   'n_points':args.stat_points}
        if args.plot_losses:
            plots['losses'] = {'loss': losses}
        if args.plot_class_losses:
            spectral_params = {'losses':['spec_conv', 'log_diff', 'log_isd', 'log_mag'],
                                        'n_fft':vae.pinput['dim'], 'transform':None, 'is_sequence':False}
            spectral_eval = SpectralEvaluation(spectral_params=spectral_params)
            latent_eval = LatentEvaluation(stats=False)
            plots['class_losses'] = {'tasks':args.tasks, 'evaluators':[spectral_eval, latent_eval],
                                     'batch_size':args.batch_size, 'n_points':args.n_points, 'preprocessing':None}
        if args.plot_2descriptors and vae.platent[-1]['dim']>=3:
            plots['descriptors_2d'] = {'projections':projections, 'transformOptions':transformOptions,
                                       'nb_planes':6, 'nb_points':20, 'bounds':args.descriptor_scope,
                                       'batch_size':args.batch_size, 'preprocessing':preprocessing,
                                       'labels':label_params, 'preprocess':False}
        if args.plot_3descriptors and vae.platent[-1]['dim']>=3:
            plots['descriptors_3d'] = {'projections':projections, 'transformOptions':transformOptions, 'nb_planes':2,
                                       'nb_points':5, 'labels':label_params, 'bounds':args.descriptor_scope,
                                       'batch_size':args.batch_size, 'preprocessing':preprocessing}

        monitor = Monitor(vae, currentSet , losses, tasks, plots=plots, partitions=list(currentSet.partitions.keys()), tasks=plot_tasks)
        monitor.plot(out=out_path+'/evaluation_plots', epoch=current_epoch, preprocessing=full_transforms)


        # perform evaluation
        print('evaluating...')
        if args.evaluate:
            evaluation_results = {}
            for p in currentSet.partitions.keys():
                spectral_params = {'losses':['spec_conv', 'log_diff', 'log_isd', 'log_mag'],
                                            'n_fft':vae.pinput['dim'], 'transform':None}
                spectral_eval = SpectralEvaluation(spectral_params=spectral_params)
                latent_eval = LatentEvaluation()
                evaluator = spectral_eval + latent_eval# + dis_eval
                loader = DataLoader(currentSet, args.batch_size,  tasks=currentSet.tasks, partition=p)
                evaluation_results[p] = evaluator(vae, loader, preprocessing=preprocessing, baselines=['PCA','ICA'])

            label_params = []
            for t in args.tasks:
                label_length = int(currentSet.metadata[t].max())+1
                label_params.append({'dim':label_length})
            dis_params = {'tasks':args.tasks, 'label_params':label_params, 'optim_params':{'epochs':args.dis_epochs}, 'cuda':args.cuda}
            dis_eval = DisentanglementEvaluation(dis_params=dis_params, latent_params=vae.platent)
            loader = DataLoader(currentSet, args.batch_size, tasks=currentSet.tasks, partition='train')
            disentanglement_results = dis_eval(vae, loader, baselines=['PCA', 'ICA'], preprocessing=preprocessing, out=out_path)
            evaluation_results['disentangling'] = disentanglement_results

            evaluations[current_path] = evaluation_results
            logger(evaluation_results)
            torch.save(evaluation_results, out_path+'/evaluation_results.vs')


        # make audio examples
        if args.resynthesize:
            ids = torch.randperm(len(currentSet.files))[:args.resynthesize]
            files = [currentSet.files[i] for i in ids]
            meta = {k: v[ids] for k, v in currentSet.metadata.items()}
            resynthesize_files(currentSet, vae,  files=files, transforms=full_transforms, 
                         out=out_path, sequence=script_args.sequence, metadata=meta)

        # transfer cross_synthesis
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
            for k, v in currentSet.metadata.items():
                n_classes = currentSet.classes[k].get('_length', v.max())
                meta[k] = torch.randint(n_classes, (len(files_to_resynthesize),))
            resynthesize_files(currentSet, vae, files=files_to_resynthesize, transforms=full_transforms,
                               sequence=script_args.sequence, out=out_path+'/cross', metadata=meta, retain_phase=True)


        print('processing random trajectories...')
        if args.trajectories:
            meta = {}
            for k, v in currentSet.metadata.items():
                n_classes = currentSet.classes[k].get('_length', v.max())
                meta[k] = torch.randint(n_classes, (args.trajectories,))
            trajectory2audio(vae, ["line", "ellipse", "square", "sin", "triangle", "sawtooth"], full_transforms, n_trajectories = args.trajectories,
                             n_steps=args.traj_steps, out=out_path, iterations=args.iterations, meta=meta)


        print('interpolating between files...')
        if args.interpolations:
            interpolate_files(currentSet, vae, n_files=args.interpolations, n_interp=args.n_interp, out=out_path, #projections=projections[1],
                              transforms=full_transforms, projections=projections[0])


        del vae; del currentSet
        gc.collect(); gc.collect()



    except Exception as e:
        raise e
        print('There has been an excecption : %s'%e)
        print('for model : %s'%current_path)


