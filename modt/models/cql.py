import numpy as np
import torch
import torch.nn as nn
from modt.models.model import TrajectoryModel

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import wandb

LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0
epsilon = 1e-6
# *** code is modified from DMBP (DMBP: Diffusion model based predictor for robust offline reinforcement learning against state observation perturbations. (2023).) ***

CQL_config = {
    "eval_freq": int(1e4),
    "max_timestep": int(1e6),
    "checkpoint_start": int(9e5),
    "checkpoint_every": int(1e4),
    "policy": "Gaussian",
    # "policy": "Deterministic",
    "automatic_entropy_tuning": False,  # Always False for good results
    "iter_repeat_sampling": True,
    "gamma": 0.99,
    "tau": 0.005,
    "q_lr": 3e-4,
    "policy_lr": 3e-4,
    "alpha": 0.2,
    "target_update_interval": 1,
    "normalize": True,
    "optimizer": 'adam',
}

if CQL_config['optimizer'] == 'adam':
    from torch.optim import Adam as Optim
elif CQL_config['optimizer'] == 'adamw':
    from torch.optim import AdamW as Optim
elif CQL_config['optimizer'] == 'SGD':
    from torch.optim import SGD as Optim
        


#  Part 1 Global Function Definition
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def soft_update(
    target, source, tau
):  # Target will be updated but Source will not change
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):  # Target will be updated but Source will not change
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


#  Part 2 Network Structure Definition
class QNetwork(
    nn.Module
):  # Critic (Judge the S+P+A with Double Q) -> Double Q are independent
    def __init__(
        self, num_inputs, num_actions, hidden_dim, max_length, dropout, n_layer
    ):
        """ [(SP)A] -> 1, just input the last transition"""
        super(QNetwork, self).__init__()
        self.max_length = max_length
        self.num_outputs = 1

        # Q1 architecture
        q1_layers = [nn.Linear(num_inputs + num_actions, hidden_dim)]
        for _ in range(n_layer - 1):
            q1_layers.extend(
                [nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)]
            )
        q1_layers.extend(
            [
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_outputs),
                nn.Tanh(),
            ]
        )
        self.q1_model = nn.Sequential(*q1_layers)

        # Q2 architecture
        q2_layers = [nn.Linear(num_inputs + num_actions, hidden_dim)]
        for _ in range(n_layer - 1):
            q2_layers.extend(
                [nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)]
            )
        q2_layers.extend(
            [
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_outputs),
                nn.Tanh(),
            ]
        )
        self.q2_model = nn.Sequential(*q2_layers)

        self.apply(weights_init_)

    def forward(self, states, actions):
        if len(actions.shape) == 2:
            actions = actions.unsqueeze(1)

        # Network input is [ [State + Preference + Act ] * max_length ], output is Q(s_t, a_t, omega)
        state = states[:, [-1], :]
        action = actions[:, [-1], :]
        xu = torch.cat([state, action], dim=-1) # (bs, 1, in_dim + act_dim)
        xu = xu.reshape(states.shape[0], -1)

        x1 = self.q1_model(xu)
        x2 = self.q2_model(xu)

        return x1, x2


class GaussianPolicy(
    nn.Module
):  # Gaussian Actor -> log_prod when sampling is NOT 0
    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_dim,
        max_length,
        dropout,
        n_layer,
        action_space=None,
    ):
        super(GaussianPolicy, self).__init__()
        self.max_length = max_length
        self.num_actions = num_actions
        self.num_outputs = num_actions
        enc_layers = [nn.Linear(max_length * num_inputs, hidden_dim)]
        for _ in range(n_layer - 1):
            enc_layers.extend(
                [nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)]
            )
        self.enc = nn.Sequential(*enc_layers)  # encoder: SP * len -> hidden_dim

        self.mean_linear = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_outputs),
        )
        self.log_std_linear = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_outputs),
        )

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )

    def forward(self, states):
        states = states[:, -self.max_length :].reshape(
            states.shape[0], -1
        )  # (bs, maxlen * state_dim)
        x = self.enc(states)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, states):
        mean, log_std = self.forward(states)
        std = log_std.exp()
        # logger.info(f"Mean is {mean} and Std is {std}")
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(
    nn.Module
):  # Deterministic Actor -> log_prod when sampling is 0
    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_dim,
        max_length,
        dropout,
        n_layer,
        action_space=None,
    ):
        super(DeterministicPolicy, self).__init__()
        # self.linear1 = nn.Linear(num_inputs, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.max_length = max_length
        self.num_actions = num_actions
        self.num_outputs = num_actions
        enc_layers = [nn.Linear(max_length * num_inputs, hidden_dim)]
        for _ in range(n_layer - 1):
            enc_layers.extend(
                [nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)]
            )
        self.enc = nn.Sequential(*enc_layers)  # encoder: SP * len -> hidden_dim

        # self.mean = nn.Linear(hidden_dim, self.num_outputs)
        self.mean = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_outputs),
            nn.Tanh(),
        )
        self.noise = torch.Tensor(1, max_length, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.0
            self.action_bias = 0.0
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )

    def forward(self, states):
        states = states[:, -self.max_length :].reshape(
            states.shape[0], -1
        )  # (bs, maxlen * state_dim)
        x = self.enc(states)
        mean = self.mean(x) * self.action_scale + self.action_bias
        return mean

    def sample(self, states):
        means = self.forward(states)
        noise = self.noise.normal_(0.0, std=0.1)  # Noise is Normal(mu=0, std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        actions = means + noise
        return actions, torch.tensor(0.0), means

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


#  Part 3 Agent Class Definition
class CQLModel(TrajectoryModel):
    def __init__(
        self,
        state_dim,
        act_dim,
        pref_dim,
        n_layer=3,
        dropout=0.1,
        max_length=1,  # K=20
        hidden_size=512,
        action_space=None,
        cons_q=3,
        device="cuda",
        config=CQL_config,
        concat_state_pref=False,
        warmup_steps=10000,
        *args,
        **kwargs,
    ):
        # 0. hyper-parameters
        super().__init__(state_dim, act_dim, pref_dim)  # -> self.~ = ~
        self.gamma = config["gamma"]  # 0.99
        self.tau = config["tau"]  # 0.005
        self.alpha = config["alpha"]  # 0.2
        self.policy_type = config["policy"]  # Gaussian or Deterministic
        self.action_space = action_space
        self.target_update_interval = config["target_update_interval"]  # 1
        self.automatic_entropy_tuning = config["automatic_entropy_tuning"]  # False
        self.device = device  # cuda or cpu
        self.Conservative_Q = cons_q
        self.max_length = max_length

        num_inputs = state_dim

        # 1. Critic network -> critic, critic_target, critic_optim
        self.critic = QNetwork(
            num_inputs,
            action_space.shape[0],
            hidden_size,
            max_length,
            dropout,
            n_layer,
        ).to(device=self.device)
        self.critic_optim = Optim(self.critic.parameters(), lr=config["q_lr"])
        self.critic_sched = torch.optim.lr_scheduler.LambdaLR(
            self.critic_optim, lambda steps: min((steps+1)/warmup_steps, 1)
        )

        self.critic_target = QNetwork(
            num_inputs,
            action_space.shape[0],
            hidden_size,
            max_length,
            dropout,
            n_layer,
        ).to(self.device)

        hard_update(self.critic_target, self.critic)

        # 2. Actor/Policy network -> policy, policy_optim
        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(
                    torch.Tensor(action_space.shape).to(self.device)
                ).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Optim([self.log_alpha], lr=config["policy_lr"])

            self.policy = GaussianPolicy(
                num_inputs,
                action_space.shape[0],
                hidden_size,
                max_length,
                dropout,
                n_layer,
                action_space,
            )
            self.policy.to(self.device)
            self.policy_optim = Optim(self.policy.parameters(), lr=config["policy_lr"])

        else:  # Deterministic Do not allow autotune alpha value as alpha will always be 0
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(
                num_inputs,
                action_space.shape[0],
                hidden_size,
                max_length,
                dropout,
                n_layer,
                action_space,
            )
            self.policy.to(self.device)
            self.policy_optim = Optim(self.policy.parameters(), lr=config["policy_lr"])
        self.policy_sched = torch.optim.lr_scheduler.LambdaLR(
            self.policy_optim, lambda steps: min((steps+1)/warmup_steps, 1)
        )

    def get_action(self, states):
        states = states.reshape(states.shape[0], -1, self.state_dim)
        
        if states.shape[1] < self.max_length:
            states = torch.cat(
                [
                    torch.zeros(
                        (states.shape[0], self.max_length - states.shape[1], self.state_dim),
                        dtype=torch.float32,
                        device=states.device,
                    ),
                    states, # the last one(s)
                ],
                dim=1,
            ).to(dtype=torch.float32)
        
        # states_compli = states[:, [-1], :].repeat((1, self.max_length - states.shape[1], 1))
        # states = torch.cat([states_compli, states], dim=1)
        
        _, _, action = self.policy.sample(states)  # -> actions, log_probs, means
        action = action.reshape(states.shape[0], 1, self.act_dim)
        return action[0, -1]

    def forward(self, states):
        action, _, _ = self.policy.sample(states)
        action = action.reshape(states.shape[0], 1, self.act_dim)

        return action
    
    def train(self) -> None:
        self.policy.train()
        self.critic.train()
        self.critic_target.train()

    def eval(self) -> None:
        self.policy.eval()
        self.critic.eval()
        self.critic_target.eval()

    # Save model parameters
    def save_model(self, file_name):
        # logger.info('Saving models to {}'.format(file_name))
        print("Saving models to {}".format(file_name))
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "critic_optimizer_state_dict": self.critic_optim.state_dict(),
                "policy_optimizer_state_dict": self.policy_optim.state_dict(),
                "critic_scheduler_state_dict": self.critic_sched.state_dict(),
                "policy_scheduler_state_dict": self.policy_sched.state_dict(),
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

            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.critic_optim.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            self.policy_optim.load_state_dict(checkpoint["policy_optimizer_state_dict"])
            self.critic_sched.load_state_dict(checkpoint["critic_scheduler_state_dict"])
            self.policy_sched.load_state_dict(checkpoint["policy_scheduler_state_dict"])
            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
