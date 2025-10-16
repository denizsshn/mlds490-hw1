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

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(128, 2)  
        self.value_head  = nn.Linear(128, 1)  

    def forward(self, x):
        x = self.shared_layer(x)
        action_logits = self.policy_head(x)
        state_value   = self.value_head(x)
        return action_logits, state_value

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
    env = gym.make('CartPole-v1')
    model = ActorCritic().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    episode_rewards = []

    for episode in range(NUM_EPISODES):
        state = _reset(env)
        log_probs = []
        rewards = []
        state_values = []
        done = False

        while not done:
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action_logits, value = model(state_tensor)          # value: (1,1)
            state_values.append(value.squeeze(-1).squeeze(0))   # -> scalar

            action_dist = Categorical(logits=action_logits)
            action = action_dist.sample()
            log_probs.append(action_dist.log_prob(action).squeeze(0))  # scalar

            state, reward, done = _step(env, action.item())
            rewards.append(reward)

        episode_rewards.append(sum(rewards))

        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=DEVICE)  

        log_probs    = torch.stack(log_probs)          
        state_values = torch.stack(state_values)       

        advantage = returns - state_values.detach()   

        actor_loss  = -(log_probs * advantage).mean()
        critic_loss = nn.functional.mse_loss(state_values, returns)
        total_loss  = actor_loss + critic_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print("Training finished.")
    torch.save(model.state_dict(), 'part2_cartpole_ac_policy.pth')
    env.close()
    return episode_rewards, model

def evaluate_and_plot(episode_rewards, trained_model):
    env = gym.make('CartPole-v1')
    trained_model.eval()

    eval_rewards = []
    for _ in range(EVAL_EPISODES):
        state = _reset(env)   
        done = False
        total_reward = 0.0
        while not done:
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                action_logits, _ = trained_model(state_tensor)
                action = Categorical(logits=action_logits).sample().item()
            state, reward, done = _step(env, action)
            total_reward += reward
        eval_rewards.append(total_reward)
    env.close()

    mean_reward = float(np.mean(eval_rewards))
    std_reward  = float(np.std(eval_rewards))
    print(f"Evaluation Mean Reward: {mean_reward:.2f}")
    print(f"Evaluation Std Deviation: {std_reward:.2f}")

    # Plot 1: Training rewards
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, label='Episode Reward')
    if len(episode_rewards) >= MOVING_AVG_WINDOW:
        moving_avg = np.convolve(
            episode_rewards, np.ones(MOVING_AVG_WINDOW)/MOVING_AVG_WINDOW, mode='valid'
        )
        plt.plot(range(MOVING_AVG_WINDOW-1, len(episode_rewards)),
                 moving_avg, label=f'{MOVING_AVG_WINDOW}-Episode Moving Average')
    plt.title('Part 2: CartPole w/ Baseline — Training Rewards')
    plt.xlabel('Episode'); plt.ylabel('Total Reward'); plt.legend(); plt.grid(True)

    # Plot 2: Histogram of evaluation rewards
    plt.subplot(1, 2, 2)
    plt.hist(eval_rewards, bins=30, edgecolor='black')
    plt.axvline(mean_reward, linestyle='dashed', linewidth=2,
                label=f'Mean: {mean_reward:.2f}')
    plt.title(f'Part 2: Baseline — Evaluation ({EVAL_EPISODES} Episodes)')
    plt.xlabel('Total Reward'); plt.ylabel('Frequency'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig('part2_cartpole_baseline_results.png')
    plt.show()

if __name__ == '__main__':
    training_rewards, final_model = train()
    evaluate_and_plot(training_rewards, final_model)
    


