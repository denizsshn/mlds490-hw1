import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
import ale_py, shimmy


LEARNING_RATE = 1e-3
GAMMA = 0.99
NUM_EPISODES = 5000
UPDATE_EVERY_EPISODES = 5     
EVAL_EPISODES = 500
MOVING_AVG_WINDOW = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTION_MAP = {0: 2, 1: 3}      # 2: RIGHT, 3: LEFT

def preprocess(image: np.ndarray) -> np.ndarray:
    """210x160x3 uint8 -> (1,80,80) float32 {0,1}"""
    image = image[35:195]              
    image = image[::2, ::2, 0]         
    image[image == 144] = 0            
    image[image == 109] = 0            
    image[image != 0] = 1
    return image.astype(np.float32).reshape(1, 80, 80)

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1   = nn.Linear(32 * 8 * 8, 256)
        self.fc2   = nn.Linear(256, 2)  

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def _reset(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def _step(env, action):
    out = env.step(action)
    if len(out) == 5:  
        obs, reward, terminated, truncated, _ = out
        done = terminated or truncated
    else:              
        obs, reward, done, _ = out
    return obs, float(reward), bool(done)

def train():
    env = gym.make("ALE/Pong-v5")
    policy_net = Policy().to(DEVICE)
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    episode_rewards = []

    for episode_batch_start in range(0, NUM_EPISODES, UPDATE_EVERY_EPISODES):
        batch_log_probs = []
        batch_returns   = []
        batch_ep_rewards = []

        for _ in range(UPDATE_EVERY_EPISODES):
            state = _reset(env)
            prev_frame = preprocess(state)
            ep_rewards, ep_log_probs = [], []
            done = False

            while not done:
                current_frame = preprocess(state)
                state_diff = current_frame - prev_frame
                prev_frame = current_frame

                st = torch.as_tensor(state_diff, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                logits = policy_net(st)
                dist = Categorical(logits=logits)
                a_idx = dist.sample()
                ep_log_probs.append(dist.log_prob(a_idx).squeeze(0))

                gym_action = ACTION_MAP[a_idx.item()]
                state, reward, done = _step(env, gym_action)
                ep_rewards.append(reward)

            batch_ep_rewards.append(sum(ep_rewards))

            G = 0.0
            ep_returns = []
            for r in reversed(ep_rewards):
                G = r + GAMMA * G
                ep_returns.insert(0, G)

            batch_returns.extend(ep_returns)
            batch_log_probs.extend(ep_log_probs)

        episode_rewards.extend(batch_ep_rewards)

        R = torch.as_tensor(batch_returns, dtype=torch.float32, device=DEVICE)

        logp = torch.stack(batch_log_probs)        
        loss = -(logp * R).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_ep = episode_batch_start + UPDATE_EVERY_EPISODES

    print("Training finished.")
    torch.save(policy_net.state_dict(), "part1_pong_policy.pth")
    env.close()
    return episode_rewards, policy_net

def evaluate_and_plot(episode_rewards, trained_policy):
    env = gym.make("ALE/Pong-v5")
    trained_policy.eval()

    eval_rewards = []
    for _ in range(EVAL_EPISODES):
        state = _reset(env)
        prev_frame = preprocess(state)
        total, done = 0.0, False
        while not done:
            with torch.no_grad():
                cur = preprocess(state)
                diff = cur - prev_frame
                prev_frame = cur
                st = torch.as_tensor(diff, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                logits = trained_policy(st)
                dist = Categorical(logits=logits)
                a_idx = dist.sample()
            state, reward, done = _step(env, ACTION_MAP[a_idx.item()])
            total += reward
        eval_rewards.append(total)
    env.close()

    mean_r, std_r = float(np.mean(eval_rewards)), float(np.std(eval_rewards))
    print(f"Evaluation Mean Reward: {mean_r:.2f}")
    print(f"Evaluation Std Deviation: {std_r:.2f}")

    # Plot 1: training rewards
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(episode_rewards, label='Episode Reward')
    if len(episode_rewards) >= MOVING_AVG_WINDOW:
        mv = np.convolve(episode_rewards, np.ones(MOVING_AVG_WINDOW)/MOVING_AVG_WINDOW, mode='valid')
        plt.plot(range(MOVING_AVG_WINDOW-1, len(episode_rewards)), mv,
                 label=f'{MOVING_AVG_WINDOW}-Episode Moving Avg')
    plt.title('Part 1: Pong Training Rewards')
    plt.xlabel('Episode'); plt.ylabel('Total Reward'); plt.legend(); plt.grid(True)

    # Plot 2: eval histogram
    plt.subplot(1,2,2)
    plt.hist(eval_rewards, bins=range(-21, 22), edgecolor='black')
    plt.axvline(mean_r, linestyle='dashed', linewidth=2, label=f'Mean: {mean_r:.2f}')
    plt.title(f'Part 1: Pong Evaluation Rewards ({EVAL_EPISODES} Episodes)')
    plt.xlabel('Total Reward'); plt.ylabel('Frequency'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig('part1_pong_results.png')
    plt.show()

if __name__ == "__main__":
    training_rewards, final_policy = train()
    evaluate_and_plot(training_rewards, final_policy)
