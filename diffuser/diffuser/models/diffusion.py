from collections import namedtuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pdb

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    VP_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)


Sample = namedtuple('Sample', 'trajectories values chains')


@torch.no_grad()
# this sample fn dont use cond (depend on the unet)
def default_sample_fn(model, x, cond, t):
    model_mean, _, model_log_variance = model.p_mean_variance(
        x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values


def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t

# Inpaint = namedtuple("InpaintConfig", "traj_start traj_end dim_start dim_end target")
Inpaint = namedtuple("InpaintConfig", "start end target")

class MOGaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, pref_dim, rtg_dim, trans_dim, hidden_dim, mod_type='bc', n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None, returns_condition=False, condition_guidance_w=0.1, ar_inv=False, train_only_inv=False):
        super().__init__()
        self.horizon = horizon
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.pref_dim = pref_dim
        self.rtg_dim = rtg_dim
        self.transition_dim = trans_dim
        
        self.model = model
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w
        
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # log calculation clipped because the posterior variance
        # is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        # get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(
            action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)
        
        self.mod_type = mod_type
        
        
    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        # set loss coefficients for dimensions of observation
        if weights_dict is None:
            weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        # decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        # manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    def _apply_inpaint_cond(self, x, conditions):
        ''' Suppose traj (x) = [a, s, g], 
        conditions={
            "a" : Inpaint(a_traj_start, a_traj_end, a_dim_start, a_dim_end, a) | None
            "s" : Inpaint(s_traj_start, s_traj_end, s_dim_start, s_dim_end, s) | None
            "g" : Inpaint(g_traj_start, g_traj_end, g_dim_start, g_dim_end, g) | None
        }'''
        dim_start, dim_end = 0, self.action_dim
        if conditions['a'] is not None:
            traj_start, traj_end, value = conditions['a']
            x[:, traj_start: traj_end, dim_start: dim_end] = value[:, traj_start: traj_end, :].clone()
            
        dim_start = dim_end
        dim_end += self.observation_dim
        if conditions['s'] is not None:
            traj_start, traj_end, value = conditions['s']
            x[:, traj_start: traj_end, dim_start: dim_end] = value[:, traj_start: traj_end, :].clone()
            
        dim_start = dim_end
        dim_end += self.rtg_dim
        if conditions['g'] is not None:
            traj_start, traj_end, value = conditions['g']
            x[:, traj_start: traj_end, dim_start: dim_end] = value[:, traj_start: traj_end, :].clone()
            
        return x

    # ------------------------------------------ sampling ------------------------------------------#
    
    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w * (epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5 * torch.randn(shape, device=device)
        x = self._apply_inpaint_cond(x, cond)

        chain = [x] if return_chain else None

        progress = utils.Progress(
            self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            # x, values = sample_fn(self, x, cond, t, **sample_kwargs)
            x = self.p_sample(x, cond, t, returns)
            x = self._apply_inpaint_cond(x, cond)

            progress.update({'t': i})
            
            if return_chain: chain.append(x)

        progress.stamp()

        # x, values = sort_by_values(x, values)
        values = None
        if return_chain:
            chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)

    @torch.no_grad()
    def conditional_sample(self, cond, batch_size, returns, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, returns, **sample_kwargs)

    # ------------------------------------------ training ------------------------------------------#
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, returns=None):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = self._apply_inpaint_cond(x_noisy, cond)

        x_recon = self.model(x_noisy, cond, t, returns) # train return-cond and un-return-cond model simutaneously
        if not self.predict_epsilon:
            x_recon = self._apply_inpaint_cond(x_recon, cond)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond, returns=None):  # x -> traj; args -> cond
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,),
                          device=x.device).long()
        return self.p_losses(x, cond, t, returns)

    def forward(self, cond, batch_size, returns, *args, **kwargs):  # not used in training
        # args -> horizon, return -> Sample(<denoised traj>, <some? value>, <trajs chain>); kwargs -> verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs(-> sample_fn(**))
        return self.conditional_sample(cond, batch_size, returns, *args, **kwargs)

class ARInvModel(nn.Module):
    def __init__(self, hidden_dim, observation_dim, action_dim, low_act=-1.0, up_act=1.0):
        super(ARInvModel, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.action_embed_hid = 128
        self.out_lin = 128
        self.num_bins = 80

        self.up_act = up_act
        self.low_act = low_act
        self.bin_size = (self.up_act - self.low_act) / self.num_bins
        self.ce_loss = nn.CrossEntropyLoss()

        self.state_embed = nn.Sequential(
            nn.Linear(2 * self.observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.lin_mod = nn.ModuleList([nn.Linear(i, self.out_lin) for i in range(1, self.action_dim)])
        self.act_mod = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, self.action_embed_hid), nn.ReLU(),
                                                    nn.Linear(self.action_embed_hid, self.num_bins))])

        for _ in range(1, self.action_dim):
            self.act_mod.append(
                nn.Sequential(nn.Linear(hidden_dim + self.out_lin, self.action_embed_hid), nn.ReLU(),
                              nn.Linear(self.action_embed_hid, self.num_bins)))

    def forward(self, comb_state, deterministic=False):
        state_inp = comb_state

        state_d = self.state_embed(state_inp)
        lp_0 = self.act_mod[0](state_d)
        l_0 = torch.distributions.Categorical(logits=lp_0).sample()

        if deterministic:
            a_0 = self.low_act + (l_0 + 0.5) * self.bin_size
        else:
            a_0 = torch.distributions.Uniform(self.low_act + l_0 * self.bin_size,
                                              self.low_act + (l_0 + 1) * self.bin_size).sample()

        a = [a_0.unsqueeze(1)]

        for i in range(1, self.action_dim):
            lp_i = self.act_mod[i](torch.cat([state_d, self.lin_mod[i - 1](torch.cat(a, dim=1))], dim=1))
            l_i = torch.distributions.Categorical(logits=lp_i).sample()

            if deterministic:
                a_i = self.low_act + (l_i + 0.5) * self.bin_size
            else:
                a_i = torch.distributions.Uniform(self.low_act + l_i * self.bin_size,
                                                  self.low_act + (l_i + 1) * self.bin_size).sample()

            a.append(a_i.unsqueeze(1))

        return torch.cat(a, dim=1)

    def calc_loss(self, comb_state, action):
        eps = 1e-8
        action = torch.clamp(action, min=self.low_act + eps, max=self.up_act - eps)
        l_action = torch.div((action - self.low_act), self.bin_size, rounding_mode='floor').long()
        state_inp = comb_state

        state_d = self.state_embed(state_inp)
        loss = self.ce_loss(self.act_mod[0](state_d), l_action[:, 0])

        for i in range(1, self.action_dim):
            loss += self.ce_loss(self.act_mod[i](torch.cat([state_d, self.lin_mod[i - 1](action[:, :i])], dim=1)),
                                     l_action[:, i])

        return loss / self.action_dim

class MOGaussianInvDynDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, pref_dim, rtg_dim, trans_dim, hidden_dim, mod_type='bc', n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None, returns_condition=False, condition_guidance_w=0.1, ar_inv=False, train_only_inv=False):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = trans_dim
        self.pref_dim = pref_dim
        self.rtg_dim = rtg_dim
        self.model = model
        self.ar_inv = ar_inv
        self.train_only_inv = train_only_inv
        if self.ar_inv:
            self.inv_model = ARInvModel(hidden_dim=hidden_dim, observation_dim=observation_dim, action_dim=action_dim)
        else:
            self.inv_model = nn.Sequential(
                nn.Linear(2 * self.observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.action_dim),
            )
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w
        
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # log calculation clipped because the posterior variance
        # is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        # get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(loss_discount)
        self.loss_fn = Losses['state_l2'](loss_weights)
        
        self.mod_type = mod_type
        
        
    def get_loss_weights(self, discount):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = 1
        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        # Cause things are conditioned on t=0
        if self.predict_epsilon:
            loss_weights[0, :] = 0

        return loss_weights

    def _apply_inpaint_cond(self, x, conditions):
        ''' Suppose traj (x) = [a, s, g], 
        conditions={
            "a" : Inpaint(a_start, a_end, a) | None
            "s" : Inpaint(s_start, s_end, s) | None
            "g" : Inpaint(g_start, g_end, g) | None
        }'''
        dim_start, dim_end = 0, 0
        if conditions['a'] is not None:
            dim_end += self.action_dim
            traj_start, traj_end, value = conditions['a']
            x[:, traj_start: traj_end, dim_start: dim_end] = value[:, traj_start: traj_end, :].clone()
            
        dim_start = dim_end
        if conditions['s'] is not None:
            dim_end += self.observation_dim
            traj_start, traj_end, value = conditions['s']
            x[:, traj_start: traj_end, dim_start: dim_end] = value[:, traj_start: traj_end, :].clone()
            
        dim_start = dim_end
        if conditions['g'] is not None:
            dim_end += self.rtg_dim
            traj_start, traj_end, value = conditions['g']
            x[:, traj_start: traj_end, dim_start: dim_end] = value[:, traj_start: traj_end, :].clone()
            
        return x

    # ------------------------------------------ sampling ------------------------------------------#
    
    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w * (epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5 * torch.randn(shape, device=device)
        x = self._apply_inpaint_cond(x, cond)

        chain = [x] if return_chain else None

        progress = utils.Progress(
            self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            # x, values = sample_fn(self, x, cond, t, **sample_kwargs)
            x = self.p_sample(x, cond, t, returns)
            x = self._apply_inpaint_cond(x, cond)

            progress.update({'t': i})
            
            if return_chain: chain.append(x)

        progress.stamp()

        # x, values = sort_by_values(x, values)
        if return_chain:
            chain = torch.stack(chain, dim=1)
        return Sample(x, None, chain)

    @torch.no_grad()
    def conditional_sample(self, cond, batch_size, returns=None, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, returns, **sample_kwargs)

    # ------------------------------------------ training ------------------------------------------#
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, returns=None):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = self._apply_inpaint_cond(x_noisy, cond)

        x_recon = self.model(x_noisy, cond, t, returns)
        if not self.predict_epsilon:
            x_recon = self._apply_inpaint_cond(x_recon, cond)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, actions, cond, returns=None):  # x -> traj; actions -> ground truth actions
        s_t = x[:, :-1, :self.observation_dim]
        a_t = actions[:, :-1, :].reshape(-1, self.action_dim)
        s_t_1 = x[:, 1:, :self.observation_dim]
        s_comb_t = torch.cat([s_t, s_t_1], dim=-1).reshape(-1, 2 * self.observation_dim)
        if self.ar_inv:
            inv_loss = self.inv_model.calc_loss(s_comb_t, a_t)
        else:
            pred_a_t = self.inv_model(s_comb_t)
            inv_loss = F.mse_loss(pred_a_t, a_t)
            
        info = {}
        if self.train_only_inv:
            loss = inv_loss
            info.update({"a0_loss: loss"})
        else:
            batch_size = len(x)
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
            diffuse_loss, diffuse_info = self.p_losses(x, cond, t, returns)
            
            loss = (1 / 2) * (diffuse_loss + inv_loss)
            info.update(diffuse_info)
            
        return loss, info

    def forward(self, cond, batch_size, returns, *args, **kwargs):  # not used in training
        # args -> horizon, return -> Sample(<denoised traj>, <some? value>, <trajs chain>); kwargs -> verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs(-> sample_fn(**))
        return self.conditional_sample(cond, batch_size, returns, *args, **kwargs)