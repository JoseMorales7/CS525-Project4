#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class DQN(nn.Module):
    def __init__(self, inDim, numActions, hiddenDim = 512):
        super(DQN, self).__init__()
        self.fc1 = torch.nn.Linear(inDim, hiddenDim)
        self.fc2 = torch.nn.Linear(hiddenDim, hiddenDim)
        self.fc3 = torch.nn.Linear(hiddenDim, numActions)



    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
