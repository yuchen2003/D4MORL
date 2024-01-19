import numpy as np
import torch
from modt.training.trainer import Trainer

class CQLTrainer(Trainer):
    def train_step(self):        
        qf_loss, policy_loss, alpha_loss, alpha_value = self.model.cql.train_step()
        return (qf_loss, policy_loss, alpha_loss, alpha_value)
