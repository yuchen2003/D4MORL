from logging import warning
import numpy as np
import torch
import torch.nn as nn

from modt.models.model import TrajectoryModel
from diffuser import utils

from collections import namedtuple

DESIGNED_MOD_TYPE = ['bc', 'dd', 'dt']

Sample = namedtuple("Sample", "trajectories values chains")
Inpaint = namedtuple("InpaintConfig", "traj_start traj_end dim_start dim_end target")

class MODiffuser(TrajectoryModel):

    """
    This model uses Decision Diffuser to model ([state pref] action) * maxlen
    """

    def __init__(
        self,
        state_dim,
        act_dim,
        pref_dim,
        hidden_size,
        max_length,
        eval_context_length,
        max_ep_len,
        act_scale,
        scale,
        use_pref,
        concat_state_pref,
        concat_act_pref,
        concat_rtg_pref,
        diffuser_args=None,
        mod_type='bc',
        infer_N=1,
        cond_M=0,
        batch_size=64,
        returns_condition=False,
        condition_guidance_w=0.1,
        concat_on='r', 
        verbose=False,
        warmup_steps=10000,
        id_prefs=None,
        mixup_step=100000,
        mixup_num=8,
    ):
        super().__init__(state_dim, act_dim, pref_dim, max_length=max_length)

        self.hidden_size = hidden_size
        self.use_pref = use_pref
        self.act_scale = act_scale
        self.scale = scale
        self.concat_state_pref = concat_state_pref
        self.concat_act_pref = concat_act_pref
        self.concat_rtg_pref = concat_rtg_pref
        self.eval_context_length = eval_context_length
        self.verbose = verbose
        self.warmup_steps = warmup_steps
        self.mixup_step = mixup_step
        self.mixup_num = mixup_num

        self.state_dim = state_dim
        self.act_dim = act_dim + concat_act_pref * pref_dim   
        self.pref_dim = pref_dim
        self.rtg_dim = pref_dim + concat_rtg_pref * pref_dim 
        
        self.concat_on = concat_on 

        assert mod_type in DESIGNED_MOD_TYPE, f'Should set MO Diffuser type as one of {DESIGNED_MOD_TYPE}'
        self.ar_inv = False
        self.mod_type = mod_type
        if self.mod_type == 'bc':
            self.act_fn = self._bc_get_action
            trans_dim = self.act_dim + self.state_dim # a, s
        elif self.mod_type == 'dd':
            self.act_fn = self._dd_get_action
            trans_dim = self.state_dim + self.rtg_dim  # s, g (a from InvDyn)
            # self.ar_inv = True
        elif self.mod_type == 'dt':
            self.act_fn = self._dt_get_action
            trans_dim = self.act_dim + self.state_dim + self.rtg_dim  # a, s, g

        assert infer_N + cond_M == max_length
        self.infer_N = infer_N
        self.cond_M = cond_M
        self.gen_H = max_length # used for generation, can be different from that in training
        
        self.batch_size = batch_size

        self.args = args = diffuser_args
        self.device = args.device
        
        sort_idx = np.argsort(id_prefs[:, 0])
        self.id_prefs = torch.from_numpy(id_prefs[sort_idx]).to(dtype=torch.float32, device=self.device)
        self.pref_rec = {}
        
        model_config = utils.Config(
            args.model,
            savepath=(args.savepath, "model_config"),
            horizon=max_length,
            transition_dim=trans_dim,
            dim=args.dim,
            cond_dim=self.pref_dim, # weighted returns, rtg, pref(and dont use concat_rtg_pref)
            pref_dim=self.pref_dim,
            dim_mults=args.dim_mults,
            attention=args.attention,
            device=args.device,
            returns_condition=returns_condition,
        )  # Unet

        diffusion_config = utils.Config(
            args.diffusion,
            savepath=(args.savepath, "diffusion_config"),
            horizon=args.horizon,
            cond_M = self.cond_M,
            observation_dim=self.state_dim,
            action_dim=self.act_dim,
            pref_dim=self.pref_dim,
            rtg_dim=self.rtg_dim,
            trans_dim=trans_dim,
            hidden_dim=hidden_size,
            mod_type=mod_type,
            n_timesteps=args.n_diffusion_steps,
            loss_type=args.loss_type,
            clip_denoised=args.clip_denoised,
            predict_epsilon=args.predict_epsilon,
            # loss weighting
            action_weight=args.action_weight,
            loss_weights=args.loss_weights,
            loss_discount=args.loss_discount,
            returns_condition=returns_condition,
            condition_guidance_w=condition_guidance_w,
            ar_inv=self.ar_inv,
            device=args.device,
        )

        self._model = model_config()

        self.diffusion = diffusion_config(self._model)
        
        self.n_diffsteps = args.n_diffusion_steps

    def forward(self, cond, prefs, target_return, gen_H, n_timestep_start=None, n_timestep_end=None, **kwargs) -> Sample:
        # return -> Sample(<denoised traj>, <some? value>, <trajs chain>)
        # Just for sampling
        batch_size = target_return.shape[0]
        return self.diffusion.forward(cond, batch_size, prefs, target_return, gen_H, n_timestep_start, n_timestep_end, **kwargs)
    
    def sample_start_from_latent(self, cond, prefs, returns, n_timestep_start=None, n_timestep_end=None, latent=None, **kwargs) -> Sample:
        return self.diffusion.sample_start_from_latent(cond, prefs, returns, n_timestep_start, n_timestep_end, latent, **kwargs)

    def get_action(self, states, actions, rtg, rewards, prefs, timesteps, max_r):
        ''' Plan using s_{t-N:t} and a_{t-N:t-1} (or also r_{t-N:t-1}), return the first action in the plan '''
        
        return self._get_id_action(states, actions, rtg, rewards, prefs, timesteps, max_r)
        
        # target_id_prefs = self._get_id_pref(prefs[-1])
        # if target_id_prefs is None:
        #     return self._get_id_action(states, actions, rtg, rewards, prefs, timesteps, max_r)
        # else:
        #     return self._get_ood_action(states, actions, rtg, rewards, prefs, target_id_prefs, timesteps, max_r)
    
    def _preprocess(self, states, actions, rtg, rewards, prefs, timesteps, max_r):
        rtg = rtg[-self.max_length : ]
        max_r = torch.multiply(max_r / self.scale, prefs[-1])
        
        if self.concat_on == 'r':
            target_r = rewards
        elif self.concat_on == 'g':
            target_r = rtg
        else:
            raise ValueError
        
        if self.concat_rtg_pref != 0:
            target_r = torch.cat((target_r, torch.cat([prefs] * self.concat_rtg_pref, dim=1)), dim=1)
        if self.concat_act_pref != 0:
            actions = torch.cat((actions, torch.cat([prefs] * self.concat_rtg_pref, dim=1)), dim=1)
        
        guidance_terms = torch.cat([max_r], dim=-1).view(1, -1) # target weighted returns
        # guidance_terms = rtg[[-1]]
        
        return target_r, guidance_terms
    
    def _get_ood_action(self, states, actions, rtg, rewards, prefs, target_id_prefs, timesteps, max_r):
        # currently: mod-dt, cond_M == 1
        states = self._pad_or_clip(states)
        actions = self._pad_or_clip(actions)
        
        t1 = int(self.n_diffsteps * 0.9)
        
        target_r0, guidance_terms0 = self._preprocess(states, actions, rtg, rewards, target_id_prefs[0].view(1, -1), timesteps, max_r)
        target_r0 = self._pad_or_clip(target_r0)
        conds0 = self._make_cond(actions, states, target_r0)
        traj0 = self.forward(conds0, target_id_prefs[0].view(1, -1), guidance_terms0, self.gen_H, n_timestep_start=0, n_timestep_end=self.n_diffsteps - t1, verbose=self.verbose).trajectories
        
        target_r1, guidance_terms1 = self._preprocess(states, actions, rtg, rewards, target_id_prefs[1].view(1, -1), timesteps, max_r)
        target_r1 = self._pad_or_clip(target_r1)
        conds1 = self._make_cond(actions, states, target_r1)
        traj1 = self.forward(conds1, target_id_prefs[1].view(1, -1), guidance_terms1, self.gen_H, n_timestep_start=0, n_timestep_end=self.n_diffsteps - t1, verbose=self.verbose).trajectories
        
        # with torch.no_grad():
        #     t = torch.tensor(t1, dtype=torch.long, device=self.device).view(1)
        #     traj0 = self.diffusion.q_sample(traj0, t)
        #     traj1 = self.diffusion.q_sample(traj1, t)
        
        coef = (prefs[-1, 0] - target_id_prefs[1][0]) / (target_id_prefs[0][0] - target_id_prefs[1][0])
        traj_tilde = coef * traj0 + (1. - coef) * traj1
        
        # print(f"coef={coef}")
        
        # NOTE: should use conds0 == conds1, or just condition solely on current state
        max_r = torch.multiply(max_r / self.scale, prefs[-1])
        guidance_terms = torch.cat([max_r], dim=-1).view(1, -1) # target weighted returns
        traj_gen = self.sample_start_from_latent(conds0, prefs[[-1]], guidance_terms, self.n_diffsteps - t1, t1, traj_tilde, verbose=self.verbose).trajectories
        action = traj_gen[0, self.cond_M - 1, : self.act_dim]
        
        if self.concat_act_pref:
            return action[ : -self.pref_dim]
        else:
            return action
        
    
    def _get_id_action(self, states, actions, rtg, rewards, prefs, timesteps, max_r):
        target_r, guidance_terms = self._preprocess(states, actions, rtg, rewards, prefs, timesteps, max_r)
        action = self.act_fn(states, actions, target_r, prefs[[-1]], timesteps, guidance_terms)
        # action = torch.zeros((1, self.act_dim), dtype=torch.float32, device=self.device)
        if self.concat_act_pref:
            return action[ : -self.pref_dim]
        else:
            return action

    def _get_id_pref(self, target_pref, eps=0.02):
        n_obj = target_pref.shape[-1]
        assert n_obj == 2, 'Currrently only support 2 objectives.'
        
        param = round(target_pref[0].item(), 3)
        if param in self.pref_rec:
            return self.pref_rec[param]
        
        lo = self.id_prefs[target_pref[0] >= self.id_prefs[:, 0]]
        hi = self.id_prefs[target_pref[0] < self.id_prefs[:, 0]]
        if len(hi) == 0 or len(lo) == 0: # extra-ood
            target_id_prefs = None
        else:
            hi = hi[0]
            lo = lo[-1]
            min_dist = min((hi - target_pref).abs().sum(), (lo - target_pref).abs().sum())
            if min_dist > eps: # intra-ood
                target_id_prefs = [lo, hi]
            else: # id
                target_id_prefs = None
            
        self.pref_rec.update({param: target_id_prefs})
        
        return target_id_prefs
    
    def _pad_or_clip(self, traj):
        traj_dim = traj.shape[-1]
        traj = traj.reshape(1, -1, traj_dim)
        if traj.shape[1] < self.max_length:
            traj = torch.cat(
                [
                    torch.zeros(
                        (1, self.max_length - traj.shape[1], traj_dim),
                        dtype=torch.float32,
                        device=traj.device,
                    ),
                    traj,
                ],
                dim=1,
            )
        elif traj.shape[1] > self.max_length:
            traj = traj[:, -self.max_length:, :]

        return traj

    def _make_cond(self, a=None, s=None, g=None):
        conds = {}
        dim_start, dim_end = 0, 0
        
        if a is not None:
            dim_start = dim_end
            dim_end += self.act_dim
            a = a[:, -(self.cond_M - 1):]
            conds.update({'a' : Inpaint(0, self.cond_M - 1, dim_start, dim_end, a)})
        else:
            conds.update({'a': None})
            
        if s is not None:
            dim_start = dim_end
            dim_end += self.state_dim
            s = s[:, -(self.cond_M):]
            conds.update({'s' : Inpaint(0, self.cond_M, dim_start, dim_end, s)})
        else:
            conds.update({'s': None})
            
        if g is not None:
            dim_start = dim_end
            dim_end += self.rtg_dim
            g = g[:, -(self.cond_M - 1):]
            conds.update({'g' : Inpaint(0, self.cond_M - 1, dim_start, dim_end, g)})
        else:
            conds.update({'g': None})

        return conds

    def _bc_get_action(self, states, actions, target_r, pref, t, target_return):
        # Preprocess
        actions = self._pad_or_clip(actions)
        states = self._pad_or_clip(states)
        
        conds = self._make_cond(actions, states, None)
        traj_gen = self.forward(conds, pref, target_return, self.gen_H, verbose=self.verbose).trajectories
        action = traj_gen[0, self.cond_M - 1, : self.act_dim]

        return action

    def _dd_get_action(self, states, actions, target_r, pref, t, target_return): # Refer DD:Alg.1
        # Preprocess
        states = self._pad_or_clip(states)
        target_r = self._pad_or_clip(target_r)
        
        conds = self._make_cond(None, states, target_r)
        traj_gen = self.forward(conds, pref, target_return, self.gen_H, verbose=self.verbose).trajectories
        state_traj = traj_gen[:, :, :self.state_dim]
        s_t, s_t_1 = state_traj[[-1], self.cond_M - 1, :], state_traj[[-1], self.cond_M, :]
        s_comb_t = torch.cat([s_t, s_t_1], dim=-1)
        action = self.diffusion.inv_model(s_comb_t)[-1]
        return action

    def _dt_get_action(self, states, actions, target_r, pref, t, target_return):
        # Preprocess
        actions = self._pad_or_clip(actions)
        states = self._pad_or_clip(states)
        target_r = self._pad_or_clip(target_r)
        
        conds = self._make_cond(actions, states, target_r)
        traj_gen = self.forward(conds, pref, target_return, self.gen_H, verbose=self.verbose).trajectories
        action = traj_gen[0, self.cond_M - 1, : self.act_dim]
        
        return action
    
    # Save model parameters
    def save_model(self, file_name):
        # logger.info('Saving models to {}'.format(file_name))
        print("Saving models to {}".format(file_name))
        torch.save(
            {
                "diffusion": self.diffusion,
            },
            file_name,
        )

    # Load model parameters
    def load_model(self, filename, device_idx=0, evaluate=False):
        # logger.info(f'Loading models from {filename}')
        print(f"Loading models from {filename}")
        if filename is not None:
            if device_idx == -1:
                checkpoint = torch.load(filename, map_location=f"cpu")
            else:
                checkpoint = torch.load(filename, map_location=f"cuda:{device_idx}")

            self.diffusion = checkpoint["diffusion"]
            if evaluate:
                self.diffusion.eval()
            else:
                self.diffusion.train()
