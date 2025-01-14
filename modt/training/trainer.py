import numpy as np
import torch
import time
from tqdm import tqdm
from modt.training.visualizer import visualize
from modt.models.cql import CQLModel
from mod.model import MODiffuser

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        get_batch,
        loss_fn,
        dataset_min_prefs,
        dataset_max_prefs,
        dataset_min_raw_r,
        dataset_max_raw_r,
        dataset_min_final_r,
        dataset_max_final_r,
        scheduler=None,
        eval_fns=[],
        max_iter=0,
        n_steps_per_iter=0,
        eval_only=False,
        concat_rtg_pref=0,
        concat_act_pref=0,
        logsdir='./',
        use_p_bar=False,
        datapath=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        # for plotting purposes
        self.dataset_min_prefs = dataset_min_prefs
        self.dataset_max_prefs = dataset_max_prefs
        self.dataset_min_raw_r = dataset_min_raw_r # weighted
        self.dataset_max_raw_r = dataset_max_raw_r
        self.dataset_min_final_r = dataset_min_final_r
        self.dataset_max_final_r = dataset_max_final_r
        self.scheduler = scheduler
        self.eval_fns = eval_fns
        self.max_iter = max_iter
        self.n_steps_per_iter = n_steps_per_iter
        self.eval_only = eval_only
        self.concat_rtg_pref = concat_rtg_pref
        self.concat_act_pref = concat_act_pref
        self.logsdir = logsdir
        self.diagnostics = dict()
        self.start_time = time.time()
        self.use_p_bar = use_p_bar
        self.datapath = datapath

    def train_iteration(self, ep):
        cur_step = (ep+1) * self.n_steps_per_iter
        log_file_name = f'{self.logsdir}/step={cur_step}.txt'
        with open(log_file_name, 'w') as f:
            f.write("\n")
            
        is_cql = False
        is_mod = False
        if isinstance(self.model, CQLModel):
            is_cql = True
            # print("[CQL loss contains qf_loss, policy_loss]")
        elif isinstance(self.model, MODiffuser):
            is_mod = True
            
        # 1. Training
        train_losses = []
        logs = dict()
        
        train_start = time.time()
        if not self.eval_only:
            print("training: iter =", ep)
            self.model.train()
            for ite in tqdm(range(self.n_steps_per_iter), disable=not self.use_p_bar):
                if is_mod:
                    train_loss, infos = self.train_step(ite)
                else:
                    train_loss = self.train_step()
                train_losses.append(train_loss)
                if self.scheduler is not None:
                    self.scheduler.step()
        logs['time/training'] = time.time() - train_start
        
        # 2. Evaluating
        eval_start = time.time()
        self.model.eval()
        cur_step = (ep+1) * self.n_steps_per_iter

        set_final_return, set_unweighted_raw_return, set_weighted_raw_return, set_cum_r_original = [], [], [], []
        with tqdm(self.eval_fns) as t:
            for eval_fn in t:
                t.set_postfix_str(f"eval: {eval_fn.target_pref}")
                outputs, final_returns, unweighted_raw_returns, weighted_raw_returns, cum_r_original = eval_fn(self.model, cur_step)
                set_final_return.append(np.mean(final_returns, axis=0))
                set_unweighted_raw_return.append(np.mean(unweighted_raw_returns, axis=0))
                set_weighted_raw_return.append(np.mean(weighted_raw_returns, axis=0))
                set_cum_r_original.append(np.mean(cum_r_original, axis=0))
                for k, v in outputs.items():
                    logs[f'evaluation/{k}'] = v


        rollout_unweighted_raw_r = np.array(set_unweighted_raw_return)
        rollout_weighted_raw_r = np.array(set_weighted_raw_return)
        rollout_original_raw_r = np.array(set_cum_r_original)
        target_prefs = np.array([eval_fn.target_pref for eval_fn in self.eval_fns])
        target_returns = np.array([eval_fn.target_reward for eval_fn in self.eval_fns]) # target returns are weighted
        
        
        n_obj = self.model.pref_dim
        # rollout_ratio = rollout_original_raw_r / np.sum(rollout_original_raw_r, axis=1, keepdims=True)
        rollout_logs = {
            'n_obj': n_obj,
            'target_prefs': target_prefs,
            'target_returns': target_returns,
            'dataset_min_prefs': self.dataset_min_prefs,
            'dataset_max_prefs': self.dataset_max_prefs,
            'dataset_min_raw_r': self.dataset_min_raw_r,
            'dataset_max_raw_r': self.dataset_max_raw_r,
            'dataset_min_final_r': self.dataset_min_final_r,
            'dataset_max_final_r': self.dataset_max_final_r,
            'rollout_unweighted_raw_r': rollout_unweighted_raw_r,
            'rollout_weighted_raw_r': rollout_weighted_raw_r, # for finding [achieved return vs. target return]
            'rollout_original_raw_r': rollout_original_raw_r, # unnormalized raw_r, for calculating roll-out ratio
        }
        infos = {
        "env": 'unspecified',
        "dataset": 'unspecified',
        "num_traj": 'unspecified',
        'datapath': self.datapath,
        'eps': 0.02,
        'is_custom': ('custom' in self.datapath),
        }
        visualize(rollout_logs, self.logsdir, cur_step, infos=infos, draw_ood=True)
        
        if not self.eval_only:
            cur_step = (ep+1) * self.n_steps_per_iter
            log_file_name = f'{self.logsdir}/step={cur_step}.txt'
            with open(log_file_name, 'a') as f:
                s = ''
                s += f"\n\n\n------------------> epoch: {ep} <------------------"
                if is_cql:
                    s += f"\nloss = {np.mean(train_losses, axis=1)}" # qf_loss, policy_loss
                elif is_mod:
                    s += f"\nloss = {np.mean(train_losses)}, infos = {infos}"
                else:
                    s += f"\nloss = {np.mean(train_losses)}"
                for k in self.diagnostics:
                    s += f"\n{k} = {self.diagnostics[k]}"
                f.write(s)
            
            logs['time/total'] = time.time() - self.start_time
            logs['time/evaluation'] = time.time() - eval_start
            logs['training/train_loss_mean'] = np.mean(train_losses) if not is_cql else np.mean(train_losses, axis=1)
            logs['training/train_loss_std'] = np.std(train_losses) if not is_cql else np.std(train_losses, axis=1)

            for k in self.diagnostics:
                logs[k] = self.diagnostics[k]
        return logs, rollout_logs


    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch()
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)
        
        
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:,1:], action_target, reward_target[:,1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
