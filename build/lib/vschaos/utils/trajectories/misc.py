import numpy as np
from .points import Point

def scale_traj(traj, old_min, old_max, new_min, new_max):
    return (traj - old_min)/(old_max - old_min)*(new_max-new_min)+new_min
