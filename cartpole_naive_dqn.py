# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 12:52:36 2021

@author: IntrovertedBot
"""

import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import Torch as T

class LinearDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(LinearDeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)
        
        return actions