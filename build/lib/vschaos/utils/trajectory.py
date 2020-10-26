import numpy as np, torch, scipy.ndimage as ndimage, pdb
from . import checklist


def line(z_dim, n_steps):
    origins = np.random.multivariate_normal(np.zeros((z_dim)), np.diag(3*np.ones((z_dim))), 2)
    coord_interp = np.linspace(0, 1, n_steps)
    z_interp = np.zeros((len(coord_interp), origins.shape[1]))
    for i,y in enumerate(coord_interp):
        z_interp[i] = ndimage.map_coordinates(origins, [y * np.ones(origins.shape[1]), np.arange(origins.shape[1])], order=2)
    z_traj = torch.from_numpy(z_interp)
    return z_traj


def get_random_trajectory(trajectory_type, z_dim, n=1, n_steps=1000, **kwargs):
    trajectories = []
    trajectory_type = checklist(trajectory_type)
    for traj_type in trajectory_type:
        if traj_type in trajectory_hash.keys():
            for i in range(n):
                trajectories.append(trajectory_hash[traj_type](z_dim, n_steps, **kwargs))
        else:
            raise LookupError('trajectory type %s not known'%traj_type)
    return trajectories


def get_interpolation(origins, n_steps=1000, interp_order=2, meta=None, **kwargs):
    if len(origins.shape) > 2:
        # is a sequence ; get homotopies
        trajs = []; full_meta = None if meta is None else {k:[] for k in meta.keys()}
        for i in range(origins.shape[1]):
            current_meta = None
            if meta is not None:
                current_meta = {k: v[:, i] for k, v in meta.items()}
            new_traj, new_cond = get_interpolation(origins[:,i], n_steps, meta=current_meta, interp_order=interp_order, **kwargs)
            trajs.append(new_traj)
            if meta is not None:
                {k: v.append(new_cond[k]) for k, v in full_meta.items()}
        return torch.stack(trajs, 1), {k:torch.stack(v, 1) for k, v in full_meta.items()}
    device = torch.device('cpu')
    if torch.is_tensor(origins):
        device = origins.device
        origins = origins.cpu().detach().numpy()
    coord_interp = np.linspace(0, 1, n_steps)
    z_interp = np.zeros((len(coord_interp), origins.shape[1]))
    for i,y in enumerate(coord_interp):
        z_interp[i] = ndimage.map_coordinates(origins, [y * np.ones(origins.shape[1]), np.arange(origins.shape[1])], order=interp_order)
    new_meta = {k:torch.linspace(meta[k][0], meta[k][1], len(coord_interp)).int() for k, v in meta.items()}
    z_traj = torch.from_numpy(z_interp).to(device=device)
    return z_traj, new_meta


trajectory_hash = {'line':line}
