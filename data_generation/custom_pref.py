from collections import namedtuple
import numpy as np

# if customize preference, use HoleSampling with the following config
Holes = namedtuple("Holes", "points radius prob")

def get_hole_config(tag: str='large'):
    ### large ood range ###
    if tag == 'large': 
        _TAG = 'large'
        _HOLES = Holes(points=[np.array([0.5, 0.5])], radius=0.06, prob=0)
        _HOLES_v2 = Holes(points=[np.array([0.45, 0.55])], radius=0.04, prob=0)   # for hopper-v2
        _HOLES_v3 = Holes(points=[np.array([1/3, 1/3, 1/3])], radius=0.04, prob=0) # for hopper-v3
    ### small ood range ###
    elif tag == 'small':
        _TAG = 'small'
        _HOLES = Holes(points=[np.array([0.3, 0.7]), np.array([0.4, 0.6]), np.array([0.5, 0.5])], radius=0.02, prob=0)
        _HOLES_v2 = Holes(points=[np.array([0.36, 0.64]), np.array([0.40, 0.60]), np.array([0.44, 0.56])], radius=0.01, prob=0)   # for hopper-v2
        _HOLES_v3 = Holes(points=[np.array([1/4, 1/4, 1/2]), np.array([1/4, 1/2, 1/4]), np.array([1/2, 1/4, 1/4])], radius=1/18, prob=0)
    ### few-shot scenarios ###
    elif tag == 'fewshot':
        _TAG = 'fewshot'
        _HOLES = Holes(points=[np.array([0.5, 0.5])], radius=0.06, prob=0.1)
        _HOLES_v2 = Holes(points=[np.array([0.45, 0.55])], radius=0.04, prob=0.1)   # for hopper-v2
        _HOLES_v3 = Holes(points=[np.array([1/3, 1/3, 1/3])], radius=0.04, prob=0.1) # for hopper-v3
    else:
        raise ValueError
    
    return _TAG, _HOLES, _HOLES_v2, _HOLES_v3

TAG, HOLES, HOLES_v2, HOLES_v3 = get_hole_config('large')

class RejectHole:
    def __init__(self, points, radius, prob) -> None:
        self.ps = points
        self.r = radius
        self.n_obj = len(points[0])
        self.prob = prob

    def __contains__(self, coor):
        for p_coor in self.ps:
            eps = np.random.rand(1)[0]
            if (np.sum(np.abs(p_coor - coor)) < self.n_obj * self.r) and (eps > self.prob):  
                # reject if in the reject region, for 2 obj, Hole([0.5, 0.5], r=0.01) means that data of pref == [0.49, 0.51] ~ [0.51, 0.49] is missing
                # reserve data in the range with probability PROB
                return True
        return False
    
