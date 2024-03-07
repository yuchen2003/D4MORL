from mod.model import Inpaint
from mod.model import MODiffuser
import torch
from collections import defaultdict
import numpy as np
import os
import pickle
from tqdm import tqdm

def collect_helper(args, all_data):
    data_path = f"data_generation/data_collected/{args['env_name']}"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    filename = f"{data_path}/{args['env_name']}_{args['collect_num']}_newsynthesis.pkl"
    
    with open(filename, "wb") as f:
        pickle.dump(all_data, f)
    
    print(f"Synthesized data saved to {filename}")

class DataGenerator:
    def __init__(self, diffuser: MODiffuser, max_r, num_traj=10000, max_ep_length=500) -> None:
        """Generate data using mod-dt. Assume s, a, r are concated with pref.

        Args:
            diffuser (MODiffuser): A trained modiffuser (mod-dt)
            num_traj (int, optional): traj num to generate. Defaults to 10000.
        """
        self.diffuser = diffuser
        self.num_traj = num_traj
        
        self.max_len = diffuser.max_length
        assert self.max_len == max_ep_length, "Should use max_ep_length as the generated traj length"
        self.state_dim = diffuser.state_dim
        self.act_dim = diffuser.act_dim
        self.rtg_dim = diffuser.rtg_dim
        self.pref_dim = diffuser.pref_dim
        self.scale = diffuser.scale
        self.device = diffuser.device
        self.verbose = diffuser.verbose
        
        self.max_r = torch.tensor(max_r, dtype=torch.float32, device=self.device)
        
    def _make_cond(self, pref): 
        conds = {}
        traj_start, traj_end = 0, self.max_len
        dim_start, dim_end = 0, 0
        
        for term, dim in zip(['a', 's', 'g'], [self.act_dim, self.state_dim, self.rtg_dim]):
            dim_end += dim
            dim_start = dim_end - self.pref_dim
            ph = pref.repeat(1, self.max_len, 1) # FIXME: not need this
            conds.update({term: Inpaint(traj_start, traj_end, dim_start, dim_end, ph)})
            
        return conds
    
    def _extract(self, traj):
        dim_start, dim_end = 0, 0
        res_triple = []
        for dim in [self.act_dim, self.state_dim, self.rtg_dim]:
            dim_start = dim_end
            dim_end += dim
            res_triple.append(traj[:, :, dim_start : dim_end - self.pref_dim].cpu().numpy())
            
        return tuple(res_triple)
    
    def sample(self, pref):
        pref = torch.tensor(pref, dtype=torch.float32, device=self.device)
        target_weighted_returns = torch.multiply(self.max_r, pref) / self.scale
        guidance_terms = torch.cat([target_weighted_returns], dim=-1).view(1, -1)
        cond = self._make_cond(pref)
        
        traj = self.diffuser.forward(cond, guidance_terms, verbose=self.verbose).trajectories
        actions, states, rewards = self._extract(traj)
        preference = pref.repeat(1, self.max_len, 1).cpu().numpy()
        
        traj_sample = dict(
            actions=actions[0],
            observations=states[0],
            preference=preference[0],
            raw_rewards=rewards[0] * self.scale,
        )
        
        return traj_sample
    
    def __call__(self, pref_grid, args):
        print("collecting data using diffuser...")
        all_traj = []
        
        pref_idx = np.random.randint(0, len(pref_grid), size=self.num_traj)
        prefs = pref_grid[pref_idx]
        
        for pref in tqdm(prefs):
            traj = self.sample(pref)
            all_traj.append(traj)
            
        collect_helper(args, all_traj)