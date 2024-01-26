import numpy as np
import torch
from modt.training.trainer import Trainer
from modt.models.cql import hard_update, soft_update
import torch.nn.functional as F


class CQLTrainer(Trainer):
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
        eval_fns=...,
        max_iter=0,
        n_steps_per_iter=0,
        eval_only=False,
        concat_rtg_pref=0,
        concat_act_pref=0,
        logsdir="./",
        use_p_bar=False,
    ):
        super().__init__(
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
            scheduler,
            eval_fns,
            max_iter,
            n_steps_per_iter,
            eval_only,
            concat_rtg_pref,
            concat_act_pref,
            logsdir,
            use_p_bar,
        )
        self.critic_optim = self.model.critic_optim
        self.policy_optim = self.model.policy_optim
        self.critic = self.model.critic
        self.critic_target = self.model.critic_target
        self.policy = self.model.policy
        self.critic_sched = self.model.critic_sched
        self.policy_sched = self.model.policy_sched
        self.alpha = self.model.alpha
        self.gamma = self.model.gamma
        self.tau = self.model.tau
        self.action_space = self.model.action_space
        self.step = 0

    def train_step(self):
        states, actions, rewards, next_states, preferences, dones = self.get_batch()
        # rewards = rewards.squeeze(-1) # mean|sum|max|last over contiguous states in trajs
        
        # Network updating
        act_shape = actions.shape
        single_act_shape = (actions.shape[0], 1, actions.shape[-1])
        # 1. Actor
        pred_actions, log_pi, _ = self.policy.sample(states)
        qf1_pi, qf2_pi = self.critic(states, pred_actions)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # [?]
        
        # 2. Critic
        qf1, qf2 = self.critic(states, actions)
        
        with torch.no_grad():
            next_actions, next_state_log_pis, _ = self.policy.sample(
                next_states
            )
            qf1_next_targets, qf2_next_targets = self.critic_target(
                next_states, next_actions
            )
            min_qf_next_target = (
                torch.min(qf1_next_targets, qf2_next_targets)
                - self.alpha * next_state_log_pis
            )
            next_q_value = rewards[:, -1, :] + self.gamma * (1. - dones[:, -1, :]) * self.gamma * min_qf_next_target # target Q
        
        # next_q_value = torch.clamp(next_q_value, -10, None)
        
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        
        # 1.1. Conservative term loss: qf_loss
        consQ = self.model.Conservative_Q
        if consQ == 0:
            qf_loss = qf1_loss + qf2_loss
        elif consQ == 1: # Conservative term is Q itself
            qf1_cql_loss = qf1.mean()
            qf2_cql_loss = qf2.mean()
            qf_loss = qf1_loss + qf2_loss + qf1_cql_loss + qf2_cql_loss
        elif consQ == 2 or consQ == 3:
            SP_act = 10
            cql_temp = 1
            cql_clip_diff_min = -1e6
            cql_clip_diff_max = 1e6
            cql_weight = 5.0
            
            with torch.no_grad():
                act_shape = next_actions.shape
                cql_random_actions = next_actions.new_empty(
                    (*act_shape, SP_act),
                    requires_grad=False,
                ).uniform_(-1, 1)
                cql_current_actions = next_actions.new_empty(
                    (*act_shape, SP_act),
                    requires_grad=False,
                )
                cql_current_log_pis = next_actions.new_empty(
                    (*act_shape, SP_act), requires_grad=False
                )
                cql_next_actions = next_actions.new_empty(
                    (*act_shape, SP_act),
                    requires_grad=False,
                )
                cql_next_log_pis = next_actions.new_empty(
                    (*act_shape, SP_act),
                    requires_grad=False,
                )
                for k in range(SP_act):
                    (
                        cql_current_actions[..., k],
                        cql_current_log_pis[..., k],
                        _,
                    ) = self.policy.sample(states)
                    (
                        cql_next_actions[..., k],
                        cql_next_log_pis[..., k],
                        _,
                    ) = self.policy.sample(next_states)
            q_shape = qf1.shape # bs, 1
            cql_q1_rand = qf1.new_empty((*q_shape, SP_act))
            cql_q2_rand = qf2.new_empty((*q_shape, SP_act))
            cql_q1_current = qf1.new_empty((*q_shape, SP_act))
            cql_q2_current = qf2.new_empty((*q_shape, SP_act))
            cql_q1_next = qf1.new_empty((*q_shape, SP_act))
            cql_q2_next = qf2.new_empty((*q_shape, SP_act))

            for k in range(SP_act):
                cql_q1_rand[..., k], cql_q2_rand[..., k] = self.critic(
                    states, cql_random_actions[..., k]
                )
                cql_q1_current[..., k], cql_q2_current[..., k] = self.critic(
                    states, cql_current_actions[..., k]
                )
                cql_q1_next[..., k], cql_q2_next[..., k] = self.critic(
                    states, cql_next_actions[..., k]
                )

            # -> cql_cat_q<1,2>
            cql_cat_q1 = torch.cat(
                [cql_q1_rand, torch.unsqueeze(qf1, -1), cql_q1_next, cql_q1_current],
                dim=-1,
            )
            cql_cat_q2 = torch.cat(
                [cql_q2_rand, torch.unsqueeze(qf2, -1), cql_q2_next, cql_q2_current],
                dim=-1,
            )
            if consQ == 3:
                random_density = np.log(0.5 ** self.action_space.shape[0])
                cql_cat_q1 = torch.cat(
                    [
                        cql_q1_rand - random_density,
                        cql_q1_next - torch.mean(cql_next_log_pis, dim=-2, keepdim=True),
                        cql_q1_current - torch.mean(cql_current_log_pis, dim=-2, keepdim=True),
                    ],
                    dim=-1,
                )
                cql_cat_q2 = torch.cat(
                    [
                        cql_q2_rand - random_density,
                        cql_q2_next - torch.mean(cql_next_log_pis, dim=-2, keepdim=True),
                        cql_q2_current - torch.mean(cql_current_log_pis, dim=-2, keepdim=True),
                    ],
                    dim=-1,
                )
            
            cql_qf1_ood = torch.logsumexp(cql_cat_q1 / cql_temp, dim=-1) * cql_temp
            cql_qf2_ood = torch.logsumexp(cql_cat_q2 / cql_temp, dim=-1) * cql_temp

            # Subtract the log likelihood of data
            cql_qf1_diff = torch.clamp(
                cql_qf1_ood - qf1, cql_clip_diff_min, cql_clip_diff_max
            ).mean()
            cql_qf2_diff = torch.clamp(
                cql_qf2_ood - qf2, cql_clip_diff_min, cql_clip_diff_max
            ).mean()
            
            cql_min_qf1_loss = cql_qf1_diff * cql_weight
            cql_min_qf2_loss = cql_qf2_diff * cql_weight
            
            qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss
        else:
            raise ValueError('invalid conservative Q.')
        
        # 1.2. update
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        
        # 3. Target Critic
        soft_update(self.critic_target, self.critic, self.tau)
        
        # 4. Scheduler
        # self.critic_sched.step()
        # self.policy_sched.step()
        
        if self.step % 10000 == 0:
            print(f'qf_loss:{qf_loss.item()}, policy_loss:{policy_loss.item()}')
        self.step += 1
        
        return qf_loss.item(), policy_loss.item()
