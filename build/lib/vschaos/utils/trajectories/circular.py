import numpy as np
from .trajectory import Trajectory
from .points import Point, Origin, Uniform, check_point



def ellipse_generator(*args, **kwargs):
    n_steps = kwargs.get('n_steps')
    dim = kwargs.get('dim')
    origin = kwargs.get('origin', Origin)
    radius = kwargs.get('radius', np.ones(dim))
    plane = kwargs.get('plane', dim)
    assert plane <= dim, "plane has to be lesser than dim"
    assert n_steps, "line_generator needs the n_steps argument"
    assert dim, "line_generator needs the dim argument"

    origin = check_point(origin, dim=dim)

    angles = np.zeros((n_steps, plane-1))
    for i in range(plane-1):
        angles[:, i] = np.linspace(0, 2*np.pi, n_steps) if i < plane -2 else np.linspace(0, 2*np.pi, n_steps)

    traj = np.zeros((n_steps, plane))
    print(radius)
    for i in range(plane-1):
        traj[:, i] = radius[i]*np.cos(angles[:, i])
        if i > 0:
            traj[:, i] = traj[:, i] *  np.cumprod(np.sin(angles[:, :i]), axis=1)[:, 0]
    traj[:, -1] = np.cumprod(np.sin(angles), axis=1)[:, 0]

    if plane != dim:
        random_matrix = np.random.randn(plane, dim)
        traj = traj @ random_matrix
    traj = traj + origin
    return traj

class Ellipse(Trajectory):
    _callback = ellipse_generator

def circle_generator(*args, **kwargs):
    kwargs['radius'] = np.ones((kwargs.get('dim', 1)))
    return ellipse_generator(*args, **kwargs)

class Circle(Trajectory):
    _callback = circle_generator

