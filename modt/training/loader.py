import numpy as np
import torch
import random

class GetBatch:
    def __init__(
        self,
        batch_size,
        max_len,
        max_ep_len,
        num_trajectories,
        p_sample,
        trajectories,
        sorted_inds,
        state_dim,
        act_dim,
        pref_dim,
        rtg_dim,
        state_mean,
        state_std,
        scale,
        device,
        act_low,
        act_high,
        avg_rtg=False,
        use_obj=-1,
        concat_state_pref=0,
        **kwargs,
    ):
        self.batch_size = batch_size
        self.max_len = max_len
        self.max_ep_len = max_ep_len
        self.num_trajectories = num_trajectories
        self.p_sample = p_sample
        self.trajectories = trajectories
        self.sorted_inds = sorted_inds
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.pref_dim = pref_dim
        self.rtg_dim = rtg_dim
        self.state_mean = state_mean
        self.state_std = state_std
        self.scale = scale
        self.device = device
        self.act_low = act_low
        self.act_high = act_high
        self.avg_rtg = avg_rtg
        self.use_obj = use_obj
        self.gamma = 1.0
        self.concat_state_pref = concat_state_pref

    def discount_cumsum(self, x):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + self.gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def discount_cumsum_mo(self, x_mo):
        return np.transpose(
            np.array([self.discount_cumsum(x_mo[:, i]) for i in range(x_mo.shape[1])])
        )

    def find_avg_rtg(self, x):
        return np.mean(x)

    def find_avg_rtg_mo(self, x_mo):
        return np.mean(x_mo, axis=0)

    def __call__(self):
        batch_inds = np.random.choice(
            np.arange(self.num_trajectories),
            size=self.batch_size,
            replace=True,
            p=self.p_sample,
        )
        s, a, pref, rtg, timesteps, mask = [], [], [], [], [], []
        raw_r = []
        for i in batch_inds:
            # randomly get the traj from all trajectories
            traj = self.trajectories[int(self.sorted_inds[i])]
            # randomly get the starting idx
            step_start = random.randint(0, traj["rewards"].shape[0] - 1)
            step_end = step_start + self.max_len

            s.append(
                traj["observations"][step_start:step_end].reshape(1, -1, self.state_dim)
            )
            a.append(
                np.maximum(
                    np.minimum(
                        traj["actions"][step_start:step_end].reshape(
                            1, -1, self.act_dim
                        ),
                        self.act_high,
                    ),
                    self.act_low,
                )
                / self.act_high
            )  # assume scale if relflective to 0 (-x, x)
            raw_r_to_add = traj["raw_rewards"][step_start:step_end].reshape(
                1, -1, self.pref_dim
            )
            raw_r.append(raw_r_to_add)
            pref.append(
                traj["preference"][step_start:step_end].reshape(1, -1, self.pref_dim)
            )
            timesteps.append(
                np.arange(step_start, step_start + s[-1].shape[1]).reshape(1, -1)
            )
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = (
                self.max_ep_len - 1
            )  # padding cutoff

            # non-rvs: use discount cumsum
            if not self.avg_rtg:
                if self.rtg_dim == 1:
                    rtg.append(
                        self.discount_cumsum(
                            traj["rewards"][step_start:step_end]
                        ).reshape(1, -1, self.rtg_dim)
                    )
                else:
                    rtg.append(
                        self.discount_cumsum_mo(
                            traj["raw_rewards"][step_start:step_end]
                        ).reshape(1, -1, self.rtg_dim)
                    )

                if rtg[-1].shape[1] <= s[-1].shape[1]:
                    rtg[-1] = np.concatenate(
                        [rtg[-1], np.zeros((1, 1, self.rtg_dim))], axis=1
                    )
            # rvs: use future avg, and look until the end
            else:
                if self.rtg_dim == 1:
                    rtg.append(
                        self.find_avg_rtg(
                            traj["rewards"][step_start : self.max_ep_len]
                        ).reshape(1, -1, self.rtg_dim)
                    )
                else:
                    rtg.append(
                        self.find_avg_rtg_mo(
                            traj["raw_rewards"][step_start : self.max_ep_len]
                        ).reshape(1, -1, self.rtg_dim)
                    )
            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate(
                [np.zeros((1, self.max_len - tlen, self.state_dim)), s[-1]], axis=1
            )
            a[-1] = np.concatenate(
                [np.ones((1, self.max_len - tlen, self.act_dim)) * -0.0, a[-1]], axis=1
            )
            raw_r[-1] = np.concatenate(
                [np.zeros((1, self.max_len - tlen, self.pref_dim)), raw_r[-1]], axis=1
            )
            pref[-1] = np.concatenate(
                [np.zeros((1, self.max_len - tlen, self.pref_dim)), pref[-1]], axis=1
            )
            rtg[-1] = np.concatenate(
                [np.zeros((1, self.max_len - tlen, self.rtg_dim)), rtg[-1]], axis=1
            )
            timesteps[-1] = np.concatenate(
                [np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1
            )
            mask.append(
                np.concatenate(
                    [np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1
                )
            )

        s = np.clip((s - self.state_mean) / self.state_std, -10, 10)
        s = torch.from_numpy(np.concatenate(s, axis=0)).to(
            dtype=torch.float32, device=self.device
        )
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(
            dtype=torch.float32, device=self.device
        )
        raw_r = (
            torch.from_numpy(np.concatenate(raw_r, axis=0)).to(
                dtype=torch.float32, device=self.device
            )
            / self.scale
        )
        pref = torch.from_numpy(np.concatenate(pref, axis=0)).to(
            dtype=torch.float32, device=self.device
        )
        rtg = (
            torch.from_numpy(np.concatenate(rtg, axis=0)).to(
                dtype=torch.float32, device=self.device
            )
            / self.scale
        )
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long, device=self.device
        )
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=self.device)
        return s, a, raw_r, rtg, timesteps, mask, pref

class AugGetBatch:
    def __init__(
        self,
        batch_size,
        max_len,
        max_ep_len,
        num_trajectories,
        p_sample,
        trajectories,
        sorted_inds,
        state_dim,
        act_dim,
        pref_dim,
        rtg_dim,
        state_mean,
        state_std,
        scale,
        device,
        act_low,
        act_high,
        avg_rtg=False,
        use_obj=-1,
        concat_state_pref=0,
    ):
        self.batch_size = batch_size
        self.max_len = max_len
        self.max_ep_len = max_ep_len
        self.num_trajectories = num_trajectories
        self.p_sample = p_sample
        self.trajectories = trajectories
        self.sorted_inds = sorted_inds
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.pref_dim = pref_dim
        self.rtg_dim = rtg_dim
        self.state_mean = state_mean
        self.state_std = state_std
        self.scale = scale
        self.device = device
        self.act_low = act_low
        self.act_high = act_high
        self.avg_rtg = avg_rtg
        self.use_obj = use_obj
        self.gamma = 1.0
        self.concat_state_pref = concat_state_pref
        print('[ loader.py ] Using Mixup Getbatch.')

    def discount_cumsum(self, x):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + self.gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def discount_cumsum_mo(self, x_mo):
        return np.transpose(
            np.array([self.discount_cumsum(x_mo[:, i]) for i in range(x_mo.shape[1])])
        )

    def find_avg_rtg(self, x):
        return np.mean(x)

    def find_avg_rtg_mo(self, x_mo):
        return np.mean(x_mo, axis=0)

    def __call__(self, use_mixup=False, mixup_num=8):
        batch_inds = np.random.choice(
            np.arange(self.num_trajectories),
            size=self.batch_size,
            replace=True,
            p=self.p_sample,
        )
        s, a, pref, rtg, timesteps, mask = [], [], [], [], [], []
        raw_r = []
        for i in batch_inds:
            # randomly get the traj from all trajectories
            traj = self.trajectories[int(self.sorted_inds[i])]
            # randomly get the starting idx
            step_start = random.randint(0, traj["rewards"].shape[0] - 1)
            step_end = step_start + self.max_len

            s.append(
                traj["observations"][step_start:step_end].reshape(1, -1, self.state_dim)
            )
            a.append(
                np.maximum(
                    np.minimum(
                        traj["actions"][step_start:step_end].reshape(
                            1, -1, self.act_dim
                        ),
                        self.act_high,
                    ),
                    self.act_low,
                )
                / self.act_high
            )  # assume scale if relflective to 0 (-x, x)
            raw_r_to_add = traj["raw_rewards"][step_start:step_end].reshape(
                1, -1, self.pref_dim
            )
            raw_r.append(raw_r_to_add)
            pref.append(
                traj["preference"][step_start:step_end].reshape(1, -1, self.pref_dim)
            )
            timesteps.append(
                np.arange(step_start, step_start + s[-1].shape[1]).reshape(1, -1)
            )
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = (
                self.max_ep_len - 1
            )  # padding cutoff

            # non-rvs: use discount cumsum
            if not self.avg_rtg:
                if self.rtg_dim == 1:
                    rtg.append(
                        self.discount_cumsum(
                            traj["rewards"][step_start:step_end]
                        ).reshape(1, -1, self.rtg_dim)
                    )
                else:
                    rtg.append(
                        self.discount_cumsum_mo(
                            traj["raw_rewards"][step_start:step_end]
                        ).reshape(1, -1, self.rtg_dim)
                    )

                if rtg[-1].shape[1] <= s[-1].shape[1]:
                    rtg[-1] = np.concatenate(
                        [rtg[-1], np.zeros((1, 1, self.rtg_dim))], axis=1
                    )
            # rvs: use future avg, and look until the end
            else:
                if self.rtg_dim == 1:
                    rtg.append(
                        self.find_avg_rtg(
                            traj["rewards"][step_start : self.max_ep_len]
                        ).reshape(1, -1, self.rtg_dim)
                    )
                else:
                    rtg.append(
                        self.find_avg_rtg_mo(
                            traj["raw_rewards"][step_start : self.max_ep_len]
                        ).reshape(1, -1, self.rtg_dim)
                    )
            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate(
                [np.zeros((1, self.max_len - tlen, self.state_dim)), s[-1]], axis=1
            )
            a[-1] = np.concatenate(
                [np.ones((1, self.max_len - tlen, self.act_dim)) * -0.0, a[-1]], axis=1
            )
            raw_r[-1] = np.concatenate(
                [np.zeros((1, self.max_len - tlen, self.pref_dim)), raw_r[-1]], axis=1
            )
            pref[-1] = np.concatenate(
                [np.zeros((1, self.max_len - tlen, self.pref_dim)), pref[-1]], axis=1
            )
            rtg[-1] = np.concatenate(
                [np.zeros((1, self.max_len - tlen, self.rtg_dim)), rtg[-1]], axis=1
            )
            timesteps[-1] = np.concatenate(
                [np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1
            )
            mask.append(
                np.concatenate(
                    [np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1
                )
            )

        s = np.clip((s - self.state_mean) / self.state_std, -10, 10)
        s = torch.from_numpy(np.concatenate(s, axis=0)).to(
            dtype=torch.float32, device=self.device
        )
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(
            dtype=torch.float32, device=self.device
        )
        raw_r = (
            torch.from_numpy(np.concatenate(raw_r, axis=0)).to(
                dtype=torch.float32, device=self.device
            )
            / self.scale
        )
        pref = torch.from_numpy(np.concatenate(pref, axis=0)).to(
            dtype=torch.float32, device=self.device
        )
        rtg = (
            torch.from_numpy(np.concatenate(rtg, axis=0)).to(
                dtype=torch.float32, device=self.device
            )
            / self.scale
        )
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long, device=self.device
        )
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=self.device)
        
        # Aug through direct mixup
        if use_mixup:
            # lamda = np.random.beta(1, 1, 1)[0]
            alpha0 = 0.5 # NOTE can also be modified
            lamda = np.random.uniform(-alpha0, 1 + alpha0, 1)[0]
            idx = np.random.randint(0, len(s), 2 * mixup_num)
            idx1, idx2 = idx[:mixup_num], idx[mixup_num:]
            
            s_tilde = lamda * s[idx1] + (1. - lamda) * s[idx2]
            a_tilde = lamda * a[idx1] + (1. - lamda) * a[idx2]
            rtg_tilde = lamda * rtg[idx1] + (1. - lamda) * rtg[idx2]
            raw_r_tilde = lamda * raw_r[idx1] + (1. - lamda) * raw_r[idx2]
            pref_tilde = lamda * pref[idx1] + (1. - lamda) * pref[idx2]
            
            s = torch.cat([s, s_tilde], dim=0)
            a = torch.cat([a, a_tilde], dim=0)
            raw_r = torch.cat([raw_r, raw_r_tilde], dim=0)
            rtg = torch.cat([rtg, rtg_tilde], dim=0)
            pref = torch.cat([pref, pref_tilde], dim=0)
            
            # for testing mixup on MODT
            timesteps_tilde = timesteps[idx1]
            mask_tilde = mask[idx1]
            timesteps = torch.cat([timesteps, timesteps_tilde], dim=0)
            mask = torch.cat([mask, mask_tilde], dim=0)
            
        return s, a, raw_r, rtg, timesteps, mask, pref

class QGetBatch:
    """For CQL."""

    def __init__(
        self,
        batch_size,
        max_len,
        max_ep_len,
        num_trajectories,
        p_sample,
        trajectories,
        sorted_inds,
        state_dim,
        act_dim,
        pref_dim,
        rtg_dim,
        state_mean,
        state_std,
        scale,
        device,
        act_low,
        act_high,
        avg_rtg=False,
        use_obj=-1,
        concat_state_pref=0,
    ):
        self.batch_size = batch_size
        self.max_len = max_len
        self.max_ep_len = max_ep_len
        self.num_trajectories = num_trajectories
        self.trajectories = trajectories
        self.p_sample = p_sample
        self.sorted_inds = sorted_inds
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.pref_dim = pref_dim
        self.rtg_dim = rtg_dim
        self.state_mean = state_mean
        self.state_std = state_std
        self.scale = scale
        self.device = device
        self.act_low = act_low
        self.act_high = act_high
        self.avg_rtg = avg_rtg
        self.use_obj = use_obj
        self.gamma = 1.0
        self.concat_state_pref = concat_state_pref

    def __call__(self):
        batch_inds = np.random.choice(
            np.arange(self.num_trajectories),
            size=self.batch_size,
            replace=True,
            p=self.p_sample,
        )
        # state_batch, action_batch, reward_batch, next_state_batch, mask, pref
        s, a, r, sp, p, done = [], [], [], [], [], []
        for i in batch_inds:
            # randomly get the traj from all trajectories
            traj = self.trajectories[int(self.sorted_inds[i])]
            # randomly get the step idx
            step_start = random.randint(0, traj["rewards"].shape[0] - 1)
            step_end = step_start + self.max_len

            s.append(
                traj["observations"][step_start:step_end].reshape(1, -1, self.state_dim)
            )
            a.append(
                np.maximum(
                    np.minimum(
                        traj["actions"][step_start:step_end].reshape(
                            1, -1, self.act_dim
                        ),
                        self.act_high,
                    ),
                    self.act_low,
                )
                / self.act_high
            )  # assume scale if relflective to 0 (-x, x)
            sp.append(
                traj["next_observations"][step_start:step_end].reshape(
                    1, -1, self.state_dim
                )
            )

            p.append(
                traj["preference"][step_start:step_end].reshape(1, -1, self.pref_dim)
            )
            r.append(
                traj["rewards"][step_start:step_end].reshape(1, -1, 1)
            )  # as unary vector
            done.append(
                traj["terminals"][step_start:step_end].reshape(1, -1, 1)
            )

            tlen = s[-1].shape[1]
            s[-1] = np.concatenate(
                [np.zeros((1, self.max_len - tlen, self.state_dim)), s[-1]], axis=1
            )
            a[-1] = np.concatenate(
                [np.ones((1, self.max_len - tlen, self.act_dim)) * -0.0, a[-1]], axis=1
            )
            r[-1] = np.concatenate(
                [np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1
            )
            tlen_sp = sp[-1].shape[1]
            sp[-1] = np.concatenate(
                [np.zeros((1, self.max_len - tlen_sp, self.state_dim)), sp[-1]], axis=1
            )
            p[-1] = np.concatenate(
                [np.zeros((1, self.max_len - tlen, self.pref_dim)), p[-1]], axis=1
            )
            done[-1] = np.concatenate(
                [np.zeros((1, self.max_len - tlen, 1)), done[-1]], axis=1
            )

        s = np.clip((s - self.state_mean) / self.state_std, -10, 10)
        s = torch.from_numpy(np.concatenate(s, axis=0)).to(
            dtype=torch.float32, device=self.device
        )
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(
            dtype=torch.float32, device=self.device
        )
        r = (
            torch.from_numpy(np.concatenate(r, axis=0)).to(
                dtype=torch.float32, device=self.device
            )
            / self.scale
        )
        sp = torch.from_numpy(np.concatenate(sp, axis=0)).to(
            dtype=torch.float32, device=self.device
        )
        p = torch.from_numpy(np.concatenate(p, axis=0)).to(
            dtype=torch.float32, device=self.device
        )
        done = torch.from_numpy(np.concatenate(done, axis=0)).to(
            dtype=torch.int32, device=self.device
        )

        return s, a, r, sp, p, done
