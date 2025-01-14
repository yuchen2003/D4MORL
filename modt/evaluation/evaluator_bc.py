from copy import deepcopy
import numpy as np
import torch
from modt.evaluation import Evaluator

class EvaluatorBC(Evaluator):
    
    def __call__(self, model, target_return, target_pref, cur_step):
        model.eval()
        model.to(device=self.device)

        with torch.no_grad():
            init_target_return = deepcopy(target_return)
            init_target_pref = deepcopy(target_pref)
            state_mean = torch.from_numpy(self.state_mean).to(device=self.device, dtype=torch.float32)
            state_std = torch.from_numpy(self.state_std).to(device=self.device, dtype=torch.float32)
            
            seed = np.random.randint(0, 10000)
            self.eval_env.seed(seed) # fixed seeding in evaluation to visualize
            
            state_np = self.eval_env.reset()
            state_np = np.concatenate((state_np, np.tile(init_target_pref, self.concat_state_pref)), axis=0)
            state_tensor = torch.from_numpy(state_np).to(device=self.device, dtype=torch.float32).reshape(1, self.state_dim)
            state_tensor = torch.clip((state_tensor - state_mean) / state_std, -10, 10)
            states = state_tensor
            actions = []
                
            pref_np = np.array(target_pref)

            episode_return, episode_length = 0, 0
            unweighted_raw_reward_cumulative = np.zeros(shape=(self.pref_dim), dtype=np.float32)
            cum_r_original = np.zeros(shape=(self.pref_dim), dtype=np.float32) # no scaling, no normalization
            for t in range(self.max_ep_len):
                action = model.get_action(states.to(dtype=torch.float32))
                action = action.detach().cpu().numpy()
                action = np.multiply(action, self.act_scale)
                actions.append(action)


                state_np, _, done, info = self.eval_env.step(action)
                
                cum_r_original += info['obj']
                
                raw_rewards = info['obj'] / self.scale
                if self.normalize_reward:
                    raw_rewards = (info['obj'] - self.min_each_obj_step) / (self.max_each_obj_step - self.min_each_obj_step) / self.scale
                
                state_np = np.concatenate((state_np, np.tile(init_target_pref, self.concat_state_pref)), axis=0)
                state_tensor = torch.from_numpy(state_np).to(device=self.device, dtype=torch.float32).reshape(1, self.state_dim)
                state_tensor = torch.clip((state_tensor - state_mean) / state_std, -10, 10)
                states = torch.cat([states, state_tensor], dim=0)
                
                    
                unweighted_raw_reward_cumulative += raw_rewards
                final_reward = np.dot(pref_np, raw_rewards)

                episode_return += final_reward
                episode_length += 1

                if done:
                    break
            
            
            target_ret_scaled_back = np.round(init_target_return * self.scale, 3) # this is normalized
            weighted_raw_reward_cumulative_eval = np.round(np.multiply(unweighted_raw_reward_cumulative * self.scale, init_target_pref), 3)
            unweighted_raw_return_cumulative_eval = np.round(unweighted_raw_reward_cumulative * self.scale, 3)
            total_return_scaled_back_eval = np.round(np.sum(weighted_raw_reward_cumulative_eval), 3)
            if not self.eval_only:
                log_file_name = f'{self.logsdir}/step={cur_step}.txt'
                with open(log_file_name, 'a') as f:
                    f.write(f"\ntarget return: {target_ret_scaled_back} ------------> {weighted_raw_reward_cumulative_eval}\n")
                    f.write(f"target pref: {np.round(init_target_pref, 3)} ------------> {np.round(unweighted_raw_return_cumulative_eval / np.sum(unweighted_raw_return_cumulative_eval), 3)}\n")
                    f.write(f"\tunweighted raw returns: {unweighted_raw_return_cumulative_eval}\n")
                    f.write(f"\tweighted raw return: {weighted_raw_reward_cumulative_eval}\n")
                    f.write(f"\tweighted final return: {total_return_scaled_back_eval}\n")
                    f.write(f"\tlength: {episode_length}\n")

            # use this to save the videos
            # self.decide_save_video(np.multiply(actions.detach().cpu().numpy(), self.act_scale), raw_rewards_cumulative, init_target_return, init_target_pref, seed)
            
            # print(f"target_pref:{target_pref}, target_ret:{target_return}, eps_return:{episode_return}, cum_r_original:{cum_r_original}")
            return episode_return, episode_length, unweighted_raw_reward_cumulative, weighted_raw_reward_cumulative_eval, cum_r_original