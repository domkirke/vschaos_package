import torch, numpy as np, librosa, matplotlib.pyplot as plt
from .visualize_dimred import PCA, ICA
from ..utils.misc import check_dir
from ..utils import decudify, merge_dicts, checklist, CollapsedIds
from ..utils.dataloader import DataLoader
import matplotlib.gridspec as gridspec
reduction_hash = {'pca':PCA, 'ica':ICA}
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'


"""
###################################
#
# [Descriptor space computation]
#
###################################
"""
# Set of descriptors we will analyze
descriptors = ['loudness', 'centroid', 'bandwidth', 'flatness', 'rolloff']

zsh = None

def sampleBatchCompute(model, zs, pca, cond, targetDims=None, preprocessing=None, layer=-1):
    print(pca,type(pca))
    if (targetDims is not None):
        zs = zs[targetDims]
    else:
        zs = pca.inverse_transform(zs)
    # Compute inverse transform at this point
    zs = model.format_input_data(zs)
    with torch.no_grad():
        invVal = model.decode(zs, y=cond, from_layer=layer);
    invVal = invVal[0]['out_params'].mean.squeeze().cpu().numpy()
    zsh = invVal.copy()
    if preprocessing:
        invVal = preprocessing.invert(invVal)
    invVal[invVal < 0] = 0
    # Compute all descriptors
    descValues = {'loudness': np.zeros(zs.shape[0]),
                  'centroid': np.zeros(zs.shape[0]),
                  'flatness': np.zeros(zs.shape[0]),
                  'bandwidth': np.zeros(zs.shape[0]),
                  'rolloff': np.zeros(zs.shape[0])}
    for i in range(invVal.shape[0]):
        currentFrame = np.expand_dims(invVal[i], 1)
        descValues['loudness'][i] = librosa.feature.rmse(S=currentFrame)
        descValues['centroid'][i] = librosa.feature.spectral_centroid(S=currentFrame)[0][0]
        descValues['flatness'][i] = librosa.feature.spectral_flatness(S=currentFrame)[0][0]
        descValues['bandwidth'][i] = librosa.feature.spectral_bandwidth(S=currentFrame)[0][0]
        descValues['rolloff'][i] = librosa.feature.spectral_rolloff(S=currentFrame)[0][0]
    return descValues

def plot2Ddescriptors(dataset, vae, projections=[], label=None, nb_samples=10, layers=None, nb_planes=5, bounds=[-5,5],
                      sample_layer=0, preprocessing=None, n_points=None, batch_size=None, loader=None, preprocess=False,
                      out=None,  **kwargs):

    vae_out = None
    layers = None or range(len(vae.platent))
    need_forward = len(list(filter(lambda x: type(x) == type, projections))) != 0


    if issubclass(type(projections[0]), list):
        for proj in projections:
            plot2Ddescriptors(dataset, vae, projections=proj, label=label, nb_samples=nb_samples, nb_planes=nb_planes,
                              bounds=bounds, sample_layer=sample_layer, preprocessing=preprocessing, n_points=n_points,
                              batch_size=batch_size, loader=loader, preprocess=preprocess, out=out ,**kwargs)

    if need_forward:
        ### prepare data IDs
        ids = None # points ids in database
        full_ids = CollapsedIds()
        full_ids.add(None, ids if ids is not None else np.random.permutation(len(dataset.data))[:n_points])
        ### forwarding
        if not issubclass(type(label), list) and not label is None:
            label = [label]
        # preparing dataloader
        Loader = loader or DataLoader
        loader = Loader(dataset, batch_size, ids=full_ids.get_full_ids(), tasks = label)
        # forward!
        output = []
        with torch.no_grad():
            for x,y in loader:
                if not preprocessing is None and preprocess:
                    x = preprocessing(x)
                output.append(decudify(vae.encode(vae.format_input_data(x), return_shifts=None, y=y, **kwargs)))
        torch.cuda.empty_cache()
        vae_out = merge_dicts(output)

    if out is not None:
        out=  out+'/descriptor_2d/'
        check_dir(out)

    figs = []; axes = []
    projections = checklist(projections)
    for layer in layers:
        reduction = projections[layer]
        if type(reduction) == type:
            reduction = reduction_hash[reduction](n_components=3)
            reduction.fit(vae_out[-1])
        # First find boundaries of the space
        spaceBounds = np.zeros((3, 2))
        for i in range(3):
            spaceBounds[i, 0] = np.min(bounds[0])
            spaceBounds[i, 1] = np.max(bounds[1])
        # Now construct sampling grids for each axis
        samplingGrids = [None] * 3
        for i in range(3):
            samplingGrids[i] = np.meshgrid(np.linspace(-.9, .9, nb_samples), np.linspace(-.9, .9, nb_samples))
        # Create the set of planes
        planeDims = np.zeros((3, nb_planes))
        for i in range(3):
            curVals = np.linspace(spaceBounds[i, 0], spaceBounds[i, 1], nb_planes)
            for p in range(nb_planes):
                planeDims[i, p] = curVals[p]
        dimNames = ['X', 'Y', 'Z'];
        for dim in range(3):
            print('Dimension ' + str(dim))
            curSampling = samplingGrids[dim]
            resultMatrix = {}
            for d in descriptors:
                resultMatrix[d] = [None] * nb_planes
                for i in range(nb_planes):
                    resultMatrix[d][i] = np.zeros((nb_samples, nb_samples))
            for plane in range(nb_planes):
                print('Plane ' + str(plane))
                curPlaneVal = planeDims[dim, plane]
                points = np.zeros((nb_samples*nb_samples, 3))
                for x in range(nb_samples):
                    for y in range(nb_samples):
                        if (dim == 0):
                            curPoint = [curPlaneVal, curSampling[0][x, y], curSampling[1][x, y]]
                        if (dim == 1):
                            curPoint = [curSampling[0][x, y], curPlaneVal, curSampling[1][x, y]]
                        if (dim == 2):
                            curPoint = [curSampling[0][x, y], curSampling[1][x, y], curPlaneVal]
                        points[x*nb_samples + y] = np.array(curPoint)

                print(points.max(), points.min())
                descVals = sampleBatchCompute(vae, points, reduction, label, cond=label, layer=sample_layer)
                for x in range(nb_samples):
                    for y in range(nb_samples):
                        for d in descriptors:
                            resultMatrix[d][plane][x, y] = descVals[d][x*nb_samples+y]

            plt.figure();
            for dI in range(len(descriptors)):
                d = descriptors[dI]
                for i in range(nb_planes):
                    plt.subplot(len(descriptors), nb_planes, (dI * nb_planes) + i + 1)
                    plt.imshow(resultMatrix[d][i], interpolation="sinc");
                    plt.tick_params(which='both', labelbottom=False, labelleft=False)
                    if (i == 0):
                        plt.ylabel(d)
                        # plt.subplots_adjust(bottom=0.2, left=0.01, right=0.05, top=0.25)
            if (out is not None):
                plt.savefig(out + '_' + type(reduction).__name__ + '_' + dimNames[dim] + '.pdf', bbox_inches='tight');
                plt.close()

            figs.append(plt.gcf()); axes.append(plt.gcf().axes)
    return figs, axes



def getDescriptorGrid(sampleGrid3D, vae, pca, cond, preprocessing=None):
    # Resulting sampling tensors
    point_hash = {}
    zs = np.zeros((np.ravel(sampleGrid3D[0]).shape[0], 3))
    current_idx = 0
    for x in range(sampleGrid3D[0].shape[0]):
        for y in range(sampleGrid3D[0].shape[1]):
            for z in range(sampleGrid3D[0].shape[2]):
                curPoint = [sampleGrid3D[0][x, y, z], sampleGrid3D[1][x, y, z], sampleGrid3D[2][x, y, z]]
                zs[current_idx] = np.array(curPoint)
                point_hash[(x, y, z)] = current_idx
                current_idx += 1

    #    cond = vae.format_label_data(np.ones(zs.shape[0]))
    descVals = sampleBatchCompute(vae, zs, pca, cond, preprocessing=preprocessing)

    resultTensor = {}
    for d in descVals.keys():
        resultTensor[d] = np.zeros_like(sampleGrid3D[0])

    for x in range(sampleGrid3D[0].shape[0]):
        for y in range(sampleGrid3D[0].shape[1]):
            for z in range(sampleGrid3D[0].shape[2]):
                current_idx = point_hash[x, y, z]
                for d in descVals.keys():
                    resultTensor[d][x, y, z] = descVals[d][current_idx]

    return resultTensor

def plot3Ddescriptors(dataset, vae, projections=[], label=None, nb_samples=15, nb_planes=10, layers=None, bounds=[-5,5],
                      figName=None, loadFrom=None, saveAs=None, resultTensor=None, preprocessing=None, n_points=None,
                      batch_size=None, loader=None, preprocess=False, out=None, **kwargs):

    ### prepare data IDs
    # if some projections are just classes, compute them
    vae_out = None
    layers = layers or range(len(vae.platent))
    fig = plt.figure(figsize=(12, 6))
    need_forward = len(list(filter(lambda x: type(x) == type, projections))) != 0

    if issubclass(type(projections[0]), list):
        for proj in projections:
            plot3Ddescriptors(dataset, vae, projections=proj, label=label, nb_samples=nb_samples, nb_planes=nb_planes,
                              bounds=bounds,  preprocessing=preprocessing, n_points=n_points,
                              batch_size=batch_size, loader=loader, preprocess=preprocess, out=out ,**kwargs)

    if need_forward:
        ids = None # points ids in database
        full_ids = CollapsedIds()
        full_ids.add(None, ids if ids is not None else np.random.permutation(len(dataset.data))[:n_points])
        ### forwarding
        if not issubclass(type(label), list) and not label is None:
            label = [label]
        # preparing dataloader
        Loader = loader or DataLoader
        loader = Loader(dataset, batch_size, ids=full_ids.get_full_ids(), tasks = label)
        # forward!
        output = []
        with torch.no_grad():
            for x,y in loader:
                if not preprocessing is None and preprocess:
                    x = preprocessing(x)
                output.append(decudify(vae.encode(vae.format_input_data(x), y=y, **kwargs)))
        torch.cuda.empty_cache()
        vae_out = merge_dicts(output)

    # Create sampling grid
    bounds = np.array([bounds]*3).T
    samplingGrid3D = np.meshgrid(np.linspace(np.min(bounds[:, 0]), np.max(bounds[:, 0]), nb_samples),
                                 np.linspace(np.min(bounds[:, 1]), np.max(bounds[:, 1]), nb_samples),
                                 np.linspace(np.min(bounds[:, 2]), np.max(bounds[:, 2]), nb_samples))

    if out is not None:
        out=  out+'/descriptor_3d/'
        check_dir(out)

    for layer in layers:
        reduction = projections[layer]
        if type(reduction) == type:
            reduction = reduction_hash[reduction](n_components=3)
            reduction.fit(vae_out[-1])
        # Resulting sampling tensors
        if not loadFrom is None:
            print('loading from %s...' % loadFrom)
            resultTensor = np.load(loadFrom)[None][0]
        elif (resultTensor is None):
            resultTensor = getDescriptorGrid(samplingGrid3D, vae, reduction, label, preprocessing=preprocessing)
        #        for d in descriptors:
        #            resultTensor[d] = np.zeros((nb_samples, nb_samples, nb_samples))
        #        for x in range(nb_samples):
        #            for y in range(nb_samples):
        #                for z in range(nb_samples):
        #                    curPoint = [samplingGrid3D[0][x,y,z],samplingGrid3D[1][x,y,z],samplingGrid3D[2][x,y,z]]
        #                    descVals = sampleCompute(vae, torch.Tensor(curPoint), pca, cond, targetDims=[0, 1, 2])
        #                    for d in descriptors:
        #                        resultTensor[d][x, y, z] = descVals[d]
        if not saveAs is None:
            print('saving as %s...' % saveAs)
            np.save(saveAs, resultTensor)

        axNames = ['X', 'Y', 'Z']
        # Sets of planes
        Zp = bounds
        xVals = np.linspace(np.min(Zp[:, 0]), np.max(Zp[:, 0]), nb_samples)
        yVals = np.linspace(np.min(Zp[:, 1]), np.max(Zp[:, 1]), nb_samples)
        zVals = np.linspace(np.min(Zp[:, 2]), np.max(Zp[:, 2]), nb_samples)
        for dim in range(3):
            print('-- dim %d...' % dim)
            # For each descriptor
            for d in descriptors:
                print('descriptos %s...' % d)
                global i;
                i = 0;
                gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
                ax = fig.add_subplot(gs[0], projection='3d')
                plt.title('Projection ' + axNames[dim] + ' - Spectral ' + d)
                if (dim == 0):
                    surfYZ = np.array(
                        [[xVals[0], np.min(Zp[:, 1]), np.min(Zp[:, 2])], [xVals[0], np.max(Zp[:, 1]), np.min(Zp[:, 2])],
                         [xVals[0], np.max(Zp[:, 1]), np.max(Zp[:, 2])], [xVals[0], np.min(Zp[:, 1]), np.max(Zp[:, 2])]])
                if (dim == 1):
                    surfYZ = np.array(
                        [[np.min(Zp[:, 0]), yVals[0], np.min(Zp[:, 2])], [np.max(Zp[:, 0]), yVals[0], np.min(Zp[:, 2])],
                         [np.max(Zp[:, 0]), yVals[0], np.max(Zp[:, 2])], [np.min(Zp[:, 0]), yVals[0], np.max(Zp[:, 2])]])
                if (dim == 2):
                    surfYZ = np.array(
                        [[np.min(Zp[:, 0]), np.min(Zp[:, 1]), zVals[0]], [np.max(Zp[:, 0]), np.min(Zp[:, 1]), zVals[0]],
                         [np.max(Zp[:, 0]), np.max(Zp[:, 1]), zVals[0]], [np.min(Zp[:, 0]), np.max(Zp[:, 1]), zVals[0]]])
                # ax.scatter(zLatent[:, 0], zLatent[:, 1], zLatent[:, 2])
                lines = [None] * 4
                for j in range(4):
                    lines[j], = ax.plot([surfYZ[j, 0], surfYZ[(j + 1) % 4, 0]], [surfYZ[j, 1], surfYZ[(j + 1) % 4, 1]],
                                        zs=[surfYZ[j, 2], surfYZ[(j + 1) % 4, 2]], linestyle='--', color='k', linewidth=2)
                for v in range(nb_samples):
                    if (dim == 0):
                        surfYZ = np.array(
                            [[xVals[v], np.min(Zp[:, 1]), np.min(Zp[:, 2])], [xVals[v], np.max(Zp[:, 1]), np.min(Zp[:, 2])],
                             [xVals[v], np.max(Zp[:, 1]), np.max(Zp[:, 2])],
                             [xVals[v], np.min(Zp[:, 1]), np.max(Zp[:, 2])]])
                    if (dim == 1):
                        surfYZ = np.array(
                            [[np.min(Zp[:, 0]), yVals[v], np.min(Zp[:, 2])], [np.max(Zp[:, 0]), yVals[v], np.min(Zp[:, 2])],
                             [np.max(Zp[:, 0]), yVals[v], np.max(Zp[:, 2])],
                             [np.min(Zp[:, 0]), yVals[v], np.max(Zp[:, 2])]])
                    if (dim == 2):
                        surfYZ = np.array(
                            [[np.min(Zp[:, 0]), np.min(Zp[:, 1]), zVals[v]], [np.max(Zp[:, 0]), np.min(Zp[:, 1]), zVals[v]],
                             [np.max(Zp[:, 0]), np.max(Zp[:, 1]), zVals[v]],
                             [np.min(Zp[:, 0]), np.max(Zp[:, 1]), zVals[v]]])
                    for j in range(4):
                        ax.plot([surfYZ[j, 0], surfYZ[(j + 1) % 4, 0]], [surfYZ[j, 1], surfYZ[(j + 1) % 4, 1]],
                                zs=[surfYZ[j, 2], surfYZ[(j + 1) % 4, 2]], alpha=0.1, color='g', linewidth=2)
                ax1 = plt.subplot(gs[1])
                if (dim == 0):
                    im = ax1.imshow(resultTensor[d][i], animated=True)
                if (dim == 1):
                    im = ax1.imshow(resultTensor[d][:, i, :], animated=True)
                if (dim == 2):
                    im = ax1.imshow(resultTensor[d][:, :, i], animated=True)

                # Function to update
                def updatefig(*args):
                    global i
                    i += 1
                    try:
                        if (dim == 0):
                            im.set_array(resultTensor[d][i])
                        if (dim == 1):
                            im.set_array(resultTensor[d][:, i, :])
                        if (dim == 2):
                            im.set_array(resultTensor[d][:, :, i])
                        if (dim == 0):
                            surfYZ = np.array([[xVals[i], np.min(Zp[:, 1]), np.min(Zp[:, 2])],
                                               [xVals[i], np.max(Zp[:, 1]), np.min(Zp[:, 2])],
                                               [xVals[i], np.max(Zp[:, 1]), np.max(Zp[:, 2])],
                                               [xVals[i], np.min(Zp[:, 1]), np.max(Zp[:, 2])]])
                        if (dim == 1):
                            surfYZ = np.array([[np.min(Zp[:, 0]), yVals[i], np.min(Zp[:, 2])],
                                               [np.max(Zp[:, 0]), yVals[i], np.min(Zp[:, 2])],
                                               [np.max(Zp[:, 0]), yVals[i], np.max(Zp[:, 2])],
                                               [np.min(Zp[:, 0]), yVals[i], np.max(Zp[:, 2])]])
                        if (dim == 2):
                            surfYZ = np.array([[np.min(Zp[:, 0]), np.min(Zp[:, 1]), zVals[i]],
                                               [np.max(Zp[:, 0]), np.min(Zp[:, 1]), zVals[i]],
                                               [np.max(Zp[:, 0]), np.max(Zp[:, 1]), zVals[i]],
                                               [np.min(Zp[:, 0]), np.max(Zp[:, 1]), zVals[i]]])
                        for j in range(4):
                            lines[j].set_data([surfYZ[j, 0], surfYZ[(j + 1) % 4, 0]],
                                              [surfYZ[j, 1], surfYZ[(j + 1) % 4, 1]])
                            lines[j].set_3d_properties([surfYZ[j, 2], surfYZ[(j + 1) % 4, 2]])
                    except:
                        print('pass')
                    return im,

                # Set up formatting for the movie files
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=15, metadata=dict(artist='acids.ircam.fr'), bitrate=1800)
                ani = animation.FuncAnimation(fig, updatefig, frames=nb_samples, interval=50, blit=True)
                ani.save(out +'_' + type(reduction).__name__ + '_' + d + '_' + axNames[dim] + '.mp4', writer=writer)
                ani.event_source.stop()
                del ani
                plt.close()
    return fig, fig.axes
