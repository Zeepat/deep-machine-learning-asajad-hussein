# test_dqn_space_invaders.py

import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import imageio
import pandas as pd
import ale_py
from utils import select_action, make_env, DQN
from collections import Counter
import os


gym.register_envs(ale_py)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model(model_path, num_tests=10, max_steps_per_episode=10000, save_gifs=False, render_episodes=[]):
    """
    Tests the trained DQN agent with detailed logging.
    
    Args:
        model_path (str): Path to the saved model weights.
        num_tests (int): Number of test episodes to run.
        max_steps_per_episode (int): Maximum steps per test episode.
        save_gifs (bool): Whether to save gameplay as GIFs.
        render_episodes (list): List of episode numbers to render and save as GIFs.
    """
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Please provide a valid path.")
        return

    env = make_env()
    action_size = env.action_space.n

    policy_net = DQN(action_size).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    total_rewards = []
    success_count = 0
    action_counts = Counter()  # To track action distribution

    print(f"\nStarting Testing with {num_tests} Episodes...\n")

    for test in range(1, num_tests + 1):
        try:
            state, info = env.reset()
        except Exception as e:
            print(f"Error resetting environment: {e}")
            break

        total_reward = 0
        done = False
        truncated = False
        steps = 0
        frames = []

        while not done and not truncated and steps < max_steps_per_episode:
            try:
                action = select_action(state, policy_net, epsilon=0.0, action_space=env.action_space)
                action_counts[action] += 1
                next_state, reward, done, truncated, info = env.step(action)
            except Exception as e:
                print(f"Error during environment step: {e}")
                break

            if save_gifs and test in render_episodes:
                try:
                    frame = env.render()
                    frames.append(frame)
                except Exception as e:
                    print(f"Error rendering frame: {e}")

            state = next_state
            total_reward += reward
            steps += 1

        termination_reason = ""
        if done:
            termination_reason = "Agent lost all lives (done=True)"
        elif truncated:
            termination_reason = "Episode truncated (truncated=True)"
        elif steps >= max_steps_per_episode:
            termination_reason = "Reached maximum steps per episode"
        else:
            termination_reason = "Unknown termination reason"

        print(f"Test Episode {test}: Total Reward = {total_reward}, Steps = {steps}, Termination Reason: {termination_reason}")

        total_rewards.append(total_reward)

        if total_reward >= 1000:
            success_count += 1

        if save_gifs and test in render_episodes and frames:
            gif_filename = f'test_episode_{test}.gif'
            try:
                imageio.mimsave(gif_filename, frames, fps=30)
                print(f"Saved gameplay GIF to '{gif_filename}'")
            except Exception as e:
                print(f"Error saving GIF: {e}")

    env.close()

    average_reward = np.mean(total_rewards) if total_rewards else 0
    max_reward = np.max(total_rewards) if total_rewards else 0
    min_reward = np.min(total_rewards) if total_rewards else 0

    print("\nTesting Completed.")
    print(f"Average Reward over {num_tests} Episodes: {average_reward:.2f}")
    print(f"Maximum Reward: {max_reward}")
    print(f"Minimum Reward: {min_reward}")
    print(f"Number of Successful Episodes (Reward >= 1000): {success_count}/{num_tests}")
    print(f"Action Distribution: {action_counts}")

    plot_rewards = True
    if plot_rewards:
        try:
            plt.figure(figsize=(10,5))
            plt.plot(range(1, num_tests + 1), total_rewards, marker='o')
            plt.xlabel('Test Episode')
            plt.ylabel('Total Reward')
            plt.title('DQN Agent Performance on Test Episodes')
            plt.grid(True)
            plt.savefig('test_performance.png')  # Save the plot
            plt.show()
        except Exception as e:
            print(f"Error plotting rewards: {e}")

    # Optional: Plotting Action Distribution
    plot_action_distribution = True
    if plot_action_distribution:
        try:
            plt.figure(figsize=(8,6))
            actions = list(range(action_size))
            counts = [action_counts.get(action, 0) for action in actions]
            plt.bar(actions, counts, color='skyblue')
            plt.xlabel('Actions')
            plt.ylabel('Frequency')
            plt.title('Action Distribution During Testing')
            plt.xticks(actions)
            plt.grid(axis='y')
            plt.savefig('action_distribution.png')
            plt.show()
        except Exception as e:
            print(f"Error plotting action distribution: {e}")

if __name__ == "__main__":
    test_model(
        model_path=r'Lab\Final_Run\all_pth\policy_net_episode_5000.pth',
        num_tests=10,
        max_steps_per_episode=100000,
        save_gifs=False,           # Set to True to enable GIF saving
        render_episodes=[1, 5]     # Specify which episodes to render
    )