from collections import namedtuple
import numpy as np

# if customize preference, use HoleSampling with the following config
Holes = namedtuple("Holes", "points radius")
HOLES = Holes(points=[np.array([0.5, 0.5])], radius=0.01)
HOLES_v2 = Holes(points=[np.array([0.45, 0.55])], radius=0.005)
HOLES_v3 = Holes(points=[np.array([1/3, 1/3, 1/3])], radius=0.01)

class RejectHole:
    def __init__(self, points, radius) -> None:
        self.ps = points
        self.r = radius

    def __contains__(self, coor):
        for p_coor in self.ps:
            if np.sum((p_coor - coor) ** 2) < self.r:  # reject if in the reject region
                return True
        return False
