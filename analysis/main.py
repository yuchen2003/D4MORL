import torch
from mod.model import MODiffuser
from diffuser import utils
import pickle
from analysis.jacobian import calc_jacobian

# Load diffuser, Unet
path = '../experiment_runs/mod_impl4/generalize/mixup_step/40w/mod/naive/bc/MO-HalfCheetah-v2/expert_custom_large/1/ckpt/step=400000.ckpt'

device = torch.device('cuda:0')
class Parser():
    config: str = "config.locomotion"
    savepath: str = "../experiment_runs/mod_analysis/",
    horizon: int = 8
    n_diffusion_steps:int = 8
    device: int = 0
    
diffuser_args = Parser()

diffuser = MODiffuser(
    state_dim=17,
    act_dim=6,
    pref_dim=2,
    hidden_size=512,
    max_length=8,
    eval_context_length=5,
    max_ep_len=500,
    act_scale=1,
    scale=1000,
    use_pref=1,
    concat_state_pref=0,
    concat_act_pref=0,
    concat_rtg_pref=0,
    diffuser_args=diffuser_args,
    mod_type='bc',
    infer_N=7,
    cond_M=1,
    batch_size=64,
    returns_condition=False,
    condition_guidance_w=0.1,
    concat_on='r', 
    verbose=False,
    warmup_steps=10000,
    id_prefs=None,
    mixup_step=100000,
    mixup_num=8,
    loading=True,
)
diffuser.load_model(path, device_idx=0, evaluate=True)
unet = diffuser.diffusion.model

# Load dataset
env = "Walker2d"
horizon = 8
data_path = f'/home/amax/xyc/D4MORL/data_generation/data_collected/MO-{env}-v2/MO-{env}-v2_50000_newexpert_uniform.pkl'
trajs = []
with open(data_path, 'rb') as f:
    trajs = pickle.load(f)
    
import numpy as np
idx = np.random.randint(0, 50000, 10)

obs, act, pref = [], [] ,[]

for i in idx:
    obs.append(trajs[i]['observations'])
    act.append(trajs[i]['actions'])
    pref.append(trajs[i]['preference'])
    
traj_max_len = np.max([len(s) for s in obs])
for items in [obs, act, pref]:
    for i, s in enumerate(items):
        if len(s) < traj_max_len:
            items[i] = np.pad(s, ((0, traj_max_len - len(s)), (0, 0)), mode='constant')
obs, act, pref = np.array(obs), np.array(act), np.array(pref)

obs_slice, act_slice, pref_slice = [], [], []

idx = np.random.randint(0, 500 - horizon, 10)

for i in idx:
    obs_slice.extend(obs[:, i:i+horizon])
    act_slice.extend(act[:, i:i+horizon])
    pref_slice.extend(pref[:, i:i+horizon])
    
obs_slice = np.array(obs_slice)
act_slice = np.array(act_slice)
pref_slice = np.array(pref_slice)

as_traj = np.concatenate([obs_slice], axis=-1)
pref_traj = pref_slice[:, 0, 0]

print(as_traj.shape, pref_traj.shape)

# Jacobian Analysis
inp = torch.from_numpy(as_traj).to(device) # B x H x T
prefs = torch.from_numpy(pref_traj).to(device)
K = as_traj.shape[1] * as_traj.shape[0]
I = torch.eye(K).to(device)
J = calc_jacobian(inp, prefs, unet)
U, S, V = torch.svd(I - J)
