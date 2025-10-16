import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym  

LEARNING_RATE = 1e-2
GAMMA = 0.95
NUM_EPISODES = 10000
EVAL_EPISODES = 500
MOVING_AVG_WINDOW = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # left/right
        )

    def forward(self, x):
        return self.net(x)

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
    env = gym.make("CartPole-v1")
    policy_net = Policy().to(DEVICE)
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    episode_rewards = []

    for episode in range(NUM_EPISODES):
        state = _reset(env)
        log_probs, rewards = [], []
        done = False

        while not done:
            st = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits = policy_net(st)
            dist = Categorical(logits=logits)
            action = dist.sample()                    
            log_probs.append(dist.log_prob(action).squeeze(0))

            state, reward, done = _step(env, action.item())
            rewards.append(reward)

        episode_rewards.append(sum(rewards))

        returns, G = [], 0.0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=DEVICE)
        returns = returns - returns.mean()

        loss_terms = [-lp * R for lp, R in zip(log_probs, returns)]
        optimizer.zero_grad()
        torch.stack(loss_terms).sum().backward()
        optimizer.step()

    print("Training finished.")
    torch.save(policy_net.state_dict(), "part3_cartpole_policy.pth")
    env.close()
    return episode_rewards, policy_net

def evaluate_and_plot(episode_rewards, trained_policy):
    env = gym.make("CartPole-v1")
    trained_policy.eval()

    eval_rewards = []
    for _ in range(EVAL_EPISODES):
        state = _reset(env)           
        done, total = False, 0.0
        while not done:
            with torch.no_grad():
                st = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                action = Categorical(logits=trained_policy(st)).sample().item()
            state, reward, done = _step(env, action)
            total += reward
        eval_rewards.append(total)
    env.close()

    mean_reward = float(np.mean(eval_rewards))
    std_reward = float(np.std(eval_rewards))
    print(f"Evaluation Mean Reward: {mean_reward:.2f}")
    print(f"Evaluation Std Deviation: {std_reward:.2f}")

    # Plot 1: Training rewards
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, label="Episode Reward")
    if len(episode_rewards) >= MOVING_AVG_WINDOW:
        mv = np.convolve(
            episode_rewards,
            np.ones(MOVING_AVG_WINDOW) / MOVING_AVG_WINDOW,
            mode="valid"
        )
        plt.plot(range(MOVING_AVG_WINDOW - 1, len(episode_rewards)), mv,
                 label=f"{MOVING_AVG_WINDOW}-Episode Moving Avg")
    plt.title("Part 1: CartPole Training Rewards")
    plt.xlabel("Episode"); plt.ylabel("Total Reward"); plt.legend(); plt.grid(True)

    # Plot 2: Histogram of evaluation rewards
    plt.subplot(1, 2, 2)
    plt.hist(eval_rewards, bins=30, edgecolor="black")
    plt.axvline(mean_reward, linestyle="dashed", linewidth=2,
                label=f"Mean: {mean_reward:.2f}")
    plt.title(f"Evaluation Rewards ({EVAL_EPISODES} Episodes)")
    plt.xlabel("Total Reward"); plt.ylabel("Frequency"); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig("part3_cartpole_results.png")
    plt.show()

if __name__ == "__main__":
    training_rewards, final_policy = train()
    evaluate_and_plot(training_rewards, final_policy)

