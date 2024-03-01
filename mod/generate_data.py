from diffuser.models import Inpaint
from mod.model import MODiffuser
import torch

class DataGenerator:
    def __init__(self, diffuser: MODiffuser, num_traj=10000) -> None:
        """Generate data using mod-dt. Assume s, a, r are concated with pref.

        Args:
            diffuser (MODiffuser): A trained modiffuser (mod-dt)
            num_traj (int, optional): traj num to generate. Defaults to 10000.
        """
        self.diffuser = diffuser
        self.num_traj = num_traj
        
        self.max_len = self.diffuser.max_length
        self.state_dim = self.diffuser.state_dim
        self.act_dim = self.diffuser.act_dim
        self.rtg_dim = self.diffuser.rtg_dim
        self.pref_dim = self.diffuser.pref_dim
        
    def _make_cond(self, pref):
        conds = {}
        
        for term, dim in zip(['a', 's', 'g'], [self.act_dim, self.state_dim, self.rtg_dim]):
            ph = torch.zeros((1, self.max_len, dim))
            ph[:, :, -self.pref_dim] = pref
            conds.update({term: Inpaint(0, self.max_len, ph)})
    
    def sample(self):
        pass