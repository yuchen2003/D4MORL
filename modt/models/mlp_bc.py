import numpy as np
import torch
import torch.nn as nn
from modt.models.model import TrajectoryModel


class MLPBCModel(TrajectoryModel):

    """
    Simple MLP that predicts next action a from past states s.
    """

    def __init__(
        self,
        state_dim,
        act_dim,
        pref_dim,
        hidden_size,
        n_layer,
        dropout=0.1,
        max_length=1, # K=20
        *args,
        **kwargs
    ):
        super().__init__(state_dim, act_dim, pref_dim)

        self.hidden_size = hidden_size
        self.max_length = max_length

        layers = [nn.Linear(max_length * self.state_dim, hidden_size)]
        for _ in range(n_layer - 1):
            layers.extend(
                [nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size, hidden_size)]
            )
        layers.extend(
            [
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, self.act_dim),
                nn.Tanh(),
            ]
        )

        self.model = nn.Sequential(*layers)

    def forward(
        self,
        states,
        actions=None,
        rewards=None,
        attention_mask=None,
        target_return=None,
    ):
        states = states[:, -self.max_length :].reshape(
            states.shape[0], -1
        )  # concat states
        actions = self.model(states).reshape(states.shape[0], 1, self.act_dim)

        return actions

    def get_action(self, states, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        if states.shape[1] < self.max_length:
            states = torch.cat(
                [
                    torch.zeros(
                        (1, self.max_length - states.shape[1], self.state_dim),
                        dtype=torch.float32,
                        device=states.device,
                    ),
                    states, # the last one(s)
                ],
                dim=1,
            )
        states = states.to(dtype=torch.float32)
        actions = self.forward(states, **kwargs)
        return actions[0, -1]
