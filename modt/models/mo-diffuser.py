from logging import warning
import numpy as np
import torch
import torch.nn as nn

import transformers
from modt.models.model import TrajectoryModel
from diffuser import utils
from diffuser.models import GaussianDiffusion


class DecisionDiffuser(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
        self,
        state_dim,
        act_dim,
        pref_dim,
        rtg_dim,
        hidden_size,
        act_scale,
        use_pref=False,
        concat_state_pref=0,
        concat_rtg_pref=0,
        concat_act_pref=0,
        max_length=None,
        eval_context_length=None,
        max_ep_len=4096,
        action_tanh=True,
        *args,
        **kwargs
    ):
        super().__init__(state_dim, act_dim, pref_dim, max_length=max_length)
        
        self.hidden_size = hidden_size
        self.use_pref = use_pref
        self.act_scale = act_scale
        self.concat_state_pref = concat_state_pref
        self.concat_rtg_pref = concat_rtg_pref
        self.concat_act_pref = concat_act_pref
        self.eval_context_length = eval_context_length
        
        self.rtg_dim = rtg_dim + concat_rtg_pref * pref_dim
        self.act_dim = act_dim + concat_act_pref * pref_dim
        
        self.init_temperature=0.1
        
        # config = transformers.GPT2Config(
        #     vocab_size=1,  # doesn't matter -- we don't use the vocab
        #     n_embd=hidden_size,
        #     **kwargs
        # )

        # # note: the only difference between this GPT2Model and the default Huggingface version
        # # is that the positional embeddings are removed (since we'll add those ourselves)
        # # self.transformer = GPT2Model(config)

        # self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        # # return and preference should have the same dimension for linear preference
        # self.embed_return = torch.nn.Linear(self.rtg_dim, hidden_size)
        # self.embed_pref = torch.nn.Linear(self.pref_dim, hidden_size, bias=False)
        # self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        # self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        # self.embed_ln = nn.LayerNorm(hidden_size)
        # self.predict_action = nn.Sequential(
        #     *([nn.Linear(hidden_size, act_dim)] + ([nn.Tanh()] if action_tanh else []))
        # )
        
        # # note: we don't predict return nor pref for the paper
        # # but you can try to see if training these jointly can improve stability
        # self.predict_return = nn.Linear(hidden_size * 2, self.pref_dim)
        # self.predict_pref = nn.Sequential(
        #     *([nn.Linear(hidden_size * 2, self.pref_dim), nn.Softmax(dim=2)])
        # )


    def forward(self, states, actions, returns_to_go, pref, timesteps, attention_mask=None):


        return action_preds, return_preds, pref_preds

    def get_action(self, states, actions, returns_to_go, pref, timesteps, **kwargs):

        return action_preds[0, -1]
