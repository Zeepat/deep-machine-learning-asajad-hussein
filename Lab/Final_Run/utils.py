import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, FrameStackObservation
import matplotlib.pyplot as plt
import random
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def select_action(state, policy_net, epsilon, action_space):
    if random.random() <= epsilon:
        return action_space.sample()

    if isinstance(state, np.ndarray):
        state = torch.tensor(state, dtype=torch.float32, device=device)
    else:
        state = state.to(device=device, dtype=torch.float32)
    if state.ndim == 3:
        state = state.unsqueeze(0)

    with torch.no_grad():
        q_values = policy_net(state)
    action = torch.argmax(q_values, dim=1).item()
    return action

def make_env():
    """
    Creates and returns a single Atari environment with necessary preprocessing.
    """
    env = gym.make('ALE/SpaceInvaders-v5', render_mode='human', frameskip=1)
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env)
    env = FrameStackObservation(env, stack_size=4)
    return env

class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, action_size)
        
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='linear')
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = x / 255.0  # Normalize input to [0, 1]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)