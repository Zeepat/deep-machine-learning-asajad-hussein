import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import threading
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, FrameStackObservation
import ale_py

# Register ALE environments
gym.register_envs(ale_py)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

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
        x = x / 255.0  
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.lock = threading.Lock()

    def push(self, state, action, reward, next_state, done):
        with self.lock:
            self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        with self.lock:
            if len(self.memory) < batch_size:
                return None
            samples = random.sample(self.memory, batch_size)
        batch = list(zip(*samples))
        states = np.array(batch[0])
        actions = np.array(batch[1])
        rewards = np.array(batch[2])
        next_states = np.array(batch[3])
        dones = np.array(batch[4], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        with self.lock:
            return len(self.memory)

def select_action(state, policy_net, epsilon, action_space):
    if random.random() <= epsilon:
        return action_space.sample()

    # Convert state to torch tensor if needed
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

def optimize_model(batch_size, memory, policy_net, target_net, optimizer, criterion, gamma):
    sample = memory.sample(batch_size)
    if sample is None:
        return None

    states, actions_batch, rewards_batch, next_states, dones_batch = sample

    states_tensor = torch.FloatTensor(states).to(device)
    actions_tensor = torch.LongTensor(actions_batch).unsqueeze(1).to(device)
    rewards_tensor = torch.FloatTensor(rewards_batch).to(device)
    next_states_tensor = torch.FloatTensor(next_states).to(device)
    dones_tensor = torch.FloatTensor(dones_batch).to(device)

    current_q_values = policy_net(states_tensor).gather(1, actions_tensor).squeeze(1)
    next_q_values = target_net(next_states_tensor).max(dim=1)[0]
    expected_q_values = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)

    loss = criterion(current_q_values, expected_q_values.detach())

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10)
    optimizer.step()

    return loss.item()

def main():
    set_seed(963)

    # Hyperparameters
    num_episodes = 5000          # Reduced episodes for faster experimentation
    batch_size = 64
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995
    target_update = 100        # Update target less frequently to reduce overhead
    save_frequency = 100       # Save model less frequently
    memory_capacity = 100000   # Reduced memory if needed
    max_steps_per_episode = 500  # Reduced steps per episode

    # Create the environment
    # Using rgb_array mode (no human rendering) for speed
    env = gym.make('ALE/SpaceInvaders-v5', render_mode='rgb_array')
    env = ResizeObservation(env, (84,84))
    env = GrayscaleObservation(env)
    env = FrameStackObservation(env, stack_size=4)

    action_size = env.action_space.n

    policy_net = DQN(action_size).to(device)
    target_net = DQN(action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.SmoothL1Loss()

    memory = ReplayMemory(memory_capacity)

    epsilon = epsilon_start

    total_rewards = []
    loss_values = []
    epsilon_values = []
    best_reward = float('-inf')

    # Print less frequently, only show progress bar updates
    episode_bar = tqdm(range(1, num_episodes + 1), desc="Training Progress", unit="episode")

    for episode in episode_bar:
        state, info = env.reset()
        total_reward = 0
        done = False
        truncated = False
        steps = 0

        while not done and not truncated and steps < max_steps_per_episode:
            action = select_action(state, policy_net, epsilon, env.action_space)
            next_state, reward, done, truncated, info = env.step(action)
            memory.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            # Only optimize if we have enough samples
            if len(memory) > batch_size:
                loss = optimize_model(batch_size, memory, policy_net, target_net, optimizer, criterion, gamma)
                if loss is not None:
                    loss_values.append(loss)
            else:
                loss = None

        total_rewards.append(total_reward)

        # Epsilon decay
        if epsilon > epsilon_end:
            epsilon *= epsilon_decay
            epsilon = max(epsilon, epsilon_end)
        epsilon_values.append(epsilon)

        # Update target network less frequently
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Save the model less frequently
        if episode % save_frequency == 0:
            torch.save(policy_net.state_dict(), f'policy_net_episode_{episode}.pth')

        # Track best reward
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(policy_net.state_dict(), 'best_policy_net.pth')

        loss_str = f"{loss_values[-1]:.4f}" if loss_values else "N/A"
        episode_bar.set_postfix({
            'Avg Reward': f"{np.mean(total_rewards[-10:]):.2f}",  # Last 10 episodes avg
            'Epsilon': f"{epsilon:.4f}",
            'Best Reward': f"{best_reward:.2f}",
            'Loss': f"{loss_str}"
        })

    env.close()

    # Plot and save data after training for efficiency
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(total_rewards, label='Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward over Episodes')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(loss_values, label='Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Loss over Training Steps')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_plots.png')
    # plt.show() # Remove this to avoid blocking

    df = pd.DataFrame({
        'Episode': range(1, len(total_rewards) + 1),
        'Reward': total_rewards,
        'Epsilon': epsilon_values[:len(total_rewards)],  # match length if needed
        'Loss': loss_values + [None]*(len(total_rewards)-len(loss_values))
    })
    df.to_csv('info.csv', index=False)

if __name__ == "__main__":
    main()
